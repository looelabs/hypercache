#!/usr/bin/env python3
"""
Measures what HyperCachePipelinedLoader does to time to first token (TTFT).

A forward pass needs every layer, so overlapping compute with I/O can save at most
min(T_io, T_compute). This compares, on a real model and the same read path:

  eager:     read and transfer every layer, then run the forward pass
  pipelined: start the forward pass after the first layer; each layer waits for its own weights

The weights are rewritten into a layer-ordered object first. HyperCache's own loader splits by
byte range only, so this script gives it true layer boundaries and does the model wiring.

Timing starts just before the loader is created. CUDA start-up and building the empty model
happen earlier and are excluded for both modes.

    python3 benchmarks/pipelined_ttft.py --model-dir /path/to/snapshot --socket /run/hc/hc.sock \
        --output benchmarks/results/pipelined_ttft.json
"""

import argparse
import ctypes
import gc
import glob
import json
import os
import platform
import statistics
import subprocess
import time
import warnings
import zlib

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockWriter
from hypercache.pipeline import HyperCachePipelinedLoader, LayerDescriptor

warnings.filterwarnings("ignore", message=".*non-writable.*")

URI = "hypercache://models/ttft/ordered-weights.bin"
PROMPT_SENTENCE = "The quick brown fox jumps over the lazy dog near the quiet river bank. "


def layer_group(name):
    if name.startswith("model.embed_tokens"):
        return 0
    if name.startswith("model.layers."):
        return int(name.split(".")[2]) + 1
    return 10_000  # final norm and lm_head run last


def build_manifest(model_dir):
    tensors = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                tensors.append((layer_group(name), name, path))
    tensors.sort(key=lambda t: (t[0], t[1]))

    entries, offset = [], 0
    for group, name, path in tensors:
        with safe_open(path, framework="pt") as f:
            piece = f.get_slice(name)
            shape, dtype = piece.get_shape(), piece.get_dtype()
        itemsize = {"F16": 2, "BF16": 2, "F32": 4}[dtype]
        length = itemsize * max(1, int(torch.tensor(shape, dtype=torch.long).prod().item()))
        entries.append(
            {"group": group, "name": name, "path": path, "offset": offset, "length": length, "shape": shape, "dtype": dtype}
        )
        offset += length
    return entries, offset


def ingest(client, entries, total):
    location = client.locate(URI, auto_ingest=False)
    if location.is_hit and location.sync_status == "LOCAL_ONLY" and location.object_size == total:
        print("Ordered weights already in HyperCache", flush=True)
        return
    print(f"Writing {total / 1e9:.1f} GB of layer-ordered weights to HyperCache", flush=True)
    data = bytearray(total)
    view = memoryview(data)
    for entry in entries:
        with safe_open(entry["path"], framework="pt") as f:
            tensor = f.get_tensor(entry["name"])
            raw = ctypes.string_at(tensor.data_ptr(), entry["length"])  # numpy has no bfloat16
        view[entry["offset"] : entry["offset"] + entry["length"]] = raw
    staged = client.stage_write(URI, total, tier="nvme", auto_flush_s3=False)
    if not staged.is_staged:
        raise RuntimeError(staged.error_message)
    writer = DirectBlockWriter()
    writer.write_chunks(staged.chunk_layout, view)
    writer.close()
    crcs, position = [], 0
    for chunk in staged.chunk_layout:
        crcs.append(zlib.crc32(view[position : position + chunk.length]) & 0xFFFFFFFF)
        position += chunk.length
    if not client.commit_write(URI, local_only=True, chunk_checksums=crcs):
        raise RuntimeError("commit failed")


def descriptors(entries):
    groups = sorted({e["group"] for e in entries})
    result = []
    for index, group in enumerate(groups):
        members = [e for e in entries if e["group"] == group]
        start = members[0]["offset"]
        length = members[-1]["offset"] + members[-1]["length"] - start
        result.append(LayerDescriptor(layer_idx=index, name=str(group), start_offset=start, length=length))
    return groups, result


class Run:
    """One empty model plus the wiring to give each layer its weights."""

    def __init__(self, config, entries, args, dtype):
        self.entries = entries
        self.dtype = dtype
        self.device = args.device
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="sdpa")
        self.model.eval()
        # Rotary embeddings are computed buffers, not weights, so meta construction leaves them empty
        rotary = self.model.model.rotary_emb
        self.model.model.rotary_emb = type(rotary)(config=config, device=args.device)
        self.groups, layers = descriptors(entries)
        self.t0 = time.perf_counter()
        self.loader = HyperCachePipelinedLoader(
            URI, socket_path=args.socket, prefer_gds=False, prefetch_ahead=args.prefetch_ahead
        )
        self.loader.layers = layers
        self.first_layer = None
        self.wait = 0.0

    def materialize(self, index, data):
        group = self.groups[index]
        base = self.loader.layers[index].start_offset
        for entry in (e for e in self.entries if e["group"] == group):
            source_dtype = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}[entry["dtype"]]
            count = entry["length"] // torch.empty(0, dtype=source_dtype).element_size()
            tensor = torch.frombuffer(data, dtype=source_dtype, count=count, offset=entry["offset"] - base)
            tensor = tensor.view(entry["shape"]).to(self.device, dtype=self.dtype)
            owner, _, leaf = entry["name"].rpartition(".")
            self.model.get_submodule(owner)._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)

    def install_hooks(self):
        inner = self.model.model
        modules = [inner.embed_tokens, *inner.layers, inner.norm]
        for index, module in enumerate(modules):
            module.register_forward_pre_hook(self._hook(index))

    def _hook(self, index):
        def hook(_module, _args):
            start = time.perf_counter()
            if index == 0 and self.first_layer is not None:
                data, self.first_layer = self.first_layer, None
            else:
                _, data = self.loader.get_layer(index)
            self.materialize(index, data)
            self.wait += time.perf_counter() - start

        return hook

    def forward(self, ids):
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        token = int(logits[0, -1].argmax())
        if self.device == "cuda":
            torch.cuda.synchronize()
        return token

    def close(self):
        self.loader.close()
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def run_eager(config, entries, args, ids, dtype):
    run = Run(config, entries, args, dtype)
    first_bytes = None
    for layer, data in run.loader.stream_layers():
        if first_bytes is None:
            first_bytes = time.perf_counter() - run.t0
        run.materialize(layer.layer_idx, data)
    loaded = time.perf_counter() - run.t0
    token = run.forward(ids)
    ttft = time.perf_counter() - run.t0
    # T_compute alone: the same pass again, with everything already on the GPU
    start = time.perf_counter()
    run.forward(ids)
    compute = time.perf_counter() - start
    run.close()
    return {"token": token, "first_layer_s": first_bytes, "t_io_s": loaded, "ttft_s": ttft, "t_compute_s": compute}


def run_pipelined(config, entries, args, ids, dtype):
    run = Run(config, entries, args, dtype)
    run.install_hooks()
    _, run.first_layer = run.loader.load_initial_layer()
    first_bytes = time.perf_counter() - run.t0
    token = run.forward(ids)
    ttft = time.perf_counter() - run.t0
    wait = run.wait
    run.close()
    return {"token": token, "first_layer_s": first_bytes, "ttft_s": ttft, "weight_wait_s": wait}


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def machine_info(device):
    gpu = ""
    if device == "cuda":
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
    return {"device": device, "gpu": gpu, "cpu_count": os.cpu_count(), "torch": torch.__version__, "python": platform.python_version(), "kernel": platform.release()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefetch-ahead", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[6, 2048, 8192])
    args = parser.parse_args()

    torch.zeros(1).to(args.device)  # CUDA start-up is not part of what's being measured
    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    entries, total = build_manifest(args.model_dir)
    client = HyperCacheClient(socket_path=args.socket, timeout=600)
    ingest(client, entries, total)
    location = client.locate(URI, auto_ingest=False)
    print(f"Object: {location.object_size} bytes, {len(location.chunk_layout)} chunks on {location.chunk_layout[0].device}")
    client.close()

    dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[entries[0]["dtype"]]
    long_ids = tokenizer(PROMPT_SENTENCE * 1500, return_tensors="pt").input_ids
    results = []
    for tokens in args.prompt_tokens:
        if tokens <= 6:
            ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
        else:
            ids = long_ids[:, :tokens]
        ids = ids.to(args.device)
        eager_rows, pipelined_rows = [], []
        for repeat in range(args.repeats):
            eager_rows.append(run_eager(config, entries, args, ids, dtype))
            pipelined_rows.append(run_pipelined(config, entries, args, ids, dtype))
            print(f"tokens={ids.shape[1]} repeat={repeat} eager={eager_rows[-1]} pipelined={pipelined_rows[-1]}", flush=True)
        tokens_match = all(e["token"] == p["token"] for e, p in zip(eager_rows, pipelined_rows))
        row = {
            "prompt_tokens": int(ids.shape[1]),
            "next_token": tokenizer.decode([eager_rows[0]["token"]]),
            "tokens_match": tokens_match,
            "first_layer_ms": 1000 * median(pipelined_rows, "first_layer_s"),
            "t_io_s": median(eager_rows, "t_io_s"),
            "t_compute_s": median(eager_rows, "t_compute_s"),
            "eager_ttft_s": median(eager_rows, "ttft_s"),
            "pipelined_ttft_s": median(pipelined_rows, "ttft_s"),
            "raw": {"eager": eager_rows, "pipelined": pipelined_rows},
        }
        row["saving_s"] = row["eager_ttft_s"] - row["pipelined_ttft_s"]
        row["saving_pct"] = 100 * row["saving_s"] / row["eager_ttft_s"]
        row["bound_s"] = min(row["t_io_s"], row["t_compute_s"])
        results.append(row)

    output = {
        "machine": machine_info(args.device),
        "model": os.path.basename(args.model_dir.rstrip("/")),
        "object_bytes": total,
        "repeats": args.repeats,
        "prefetch_ahead": args.prefetch_ahead,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print("tokens  first_layer  T_io   T_compute  eager_TTFT  pipelined_TTFT  saving         bound  match")
    for r in results:
        print(
            f"{r['prompt_tokens']:>6}  {r['first_layer_ms']:>8.1f}ms  {r['t_io_s']:>5.2f}s  {r['t_compute_s']:>7.3f}s  "
            f"{r['eager_ttft_s']:>8.2f}s  {r['pipelined_ttft_s']:>12.2f}s  {r['saving_s']:>6.2f}s {r['saving_pct']:>5.1f}%  "
            f"{r['bound_s']:>5.2f}s  {r['tokens_match']}"
        )


if __name__ == "__main__":
    main()
