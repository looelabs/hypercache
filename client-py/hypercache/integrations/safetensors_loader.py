"""
Native Safetensors, Hugging Face, and vLLM Zero-Copy Loader for HyperCache.

Parses the Safetensors binary header directly from HyperCache's striped NVMe or
POSIX shared memory blocks and reconstructs PyTorch tensors with zero intermediate
serialization overhead.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hypercache.fsspec_fs import HyperCacheFileSystem

_DTYPE_MAP = {
    "F64": ("float64", 8),
    "F32": ("float32", 4),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "I64": ("int64", 8),
    "I32": ("int32", 4),
    "I16": ("int16", 2),
    "I8": ("int8", 1),
    "U8": ("uint8", 1),
    "BOOL": ("bool", 1),
}


def parse_safetensors_header(raw_header_and_len: bytes) -> Tuple[int, Dict[str, Any]]:
    """
    Parses the 8-byte little-endian header size and JSON metadata from a Safetensors stream.
    Returns (data_start_offset, header_dict).
    """
    if len(raw_header_and_len) < 8:
        raise ValueError("Invalid safetensors file: fewer than 8 bytes")
    (header_len,) = struct.unpack("<Q", raw_header_and_len[:8])
    header_json = raw_header_and_len[8 : 8 + header_len].decode("utf-8")
    return 8 + header_len, json.loads(header_json)


def load_safetensors_state_dict(
    uri: str,
    device: str = "cpu",
    socket_path: Optional[str] = None,
    prefer_gds: bool = True,
    pinned: bool = False,
    tensor_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Loads a `.safetensors` checkpoint from `hypercache://` into a dictionary of PyTorch tensors
    (or raw memoryviews if PyTorch is not installed) using zero-copy DMA reads.

    If `tensor_names` is specified, only the requested tensors are read from the striped
    NVMe/RAM pool (selective tensor slicing for tensor-parallel / pipeline-parallel workers).
    """
    fs = HyperCacheFileSystem(socket_path=socket_path, prefer_gds=prefer_gds)
    state_dict: Dict[str, Any] = {}

    try:
        import torch
        has_torch = True
    except ImportError:
        torch = None  # type: ignore
        has_torch = False

    with fs.open(uri, "rb", pinned=pinned) as f:
        len_bytes = bytes(f.read(8))
        if len(len_bytes) < 8:
            raise ValueError(f"Safetensors file {uri} is truncated (< 8 bytes)")
        (header_len,) = struct.unpack("<Q", len_bytes)
        header_bytes = bytes(f.read(header_len))
        header = json.loads(header_bytes.decode("utf-8"))
        data_base_offset = 8 + header_len

        selected_keys = [
            k for k in header.keys()
            if k != "__metadata__" and (tensor_names is None or k in tensor_names)
        ]

        # If loading all tensors, perform a single contiguous high-queue-depth DMA read
        if tensor_names is None:
            f.seek(data_base_offset)
            payload_buf = f.read()
            payload_view = memoryview(payload_buf)

            for name in selected_keys:
                meta = header[name]
                dtype_code = meta["dtype"]
                shape = meta["shape"]
                start_rel, end_rel = meta["data_offsets"]
                tensor_slice = payload_view[start_rel:end_rel]

                if has_torch:
                    torch_dtype_name, _ = _DTYPE_MAP.get(dtype_code, ("uint8", 1))
                    torch_dtype = getattr(torch, torch_dtype_name)
                    if len(tensor_slice) == 0:
                        t = torch.empty(shape, dtype=torch_dtype, device=device)
                    else:
                        t = torch.frombuffer(bytearray(tensor_slice), dtype=torch_dtype).reshape(shape)
                        if device != "cpu":
                            t = t.to(device)
                    state_dict[name] = t
                else:
                    state_dict[name] = bytes(tensor_slice)
        else:
            # Selective byte-range read for only the requested tensors
            for name in selected_keys:
                meta = header[name]
                dtype_code = meta["dtype"]
                shape = meta["shape"]
                start_rel, end_rel = meta["data_offsets"]
                byte_len = end_rel - start_rel

                f.seek(data_base_offset + start_rel)
                raw_slice = f.read(byte_len)

                if has_torch:
                    torch_dtype_name, _ = _DTYPE_MAP.get(dtype_code, ("uint8", 1))
                    torch_dtype = getattr(torch, torch_dtype_name)
                    if byte_len == 0:
                        t = torch.empty(shape, dtype=torch_dtype, device=device)
                    else:
                        t = torch.frombuffer(bytearray(raw_slice), dtype=torch_dtype).reshape(shape)
                        if device != "cpu":
                            t = t.to(device)
                    state_dict[name] = t
                else:
                    state_dict[name] = bytes(raw_slice)

    return state_dict


def save_safetensors_state_dict(
    state_dict: Dict[str, Any],
    uri: str,
    metadata: Optional[Dict[str, str]] = None,
    socket_path: Optional[str] = None,
    tier: Optional[str] = None,
    write_through: bool = False,
    pinned: bool = False,
) -> int:
    """
    Serializes a dictionary of PyTorch tensors (or raw byte buffers) into Safetensors format
    and streams it in 4 MiB chunks directly to HyperCache.
    """
    try:
        import torch
        has_torch = True
    except ImportError:
        torch = None  # type: ignore
        has_torch = False

    reverse_dtype = {v[0]: k for k, v in _DTYPE_MAP.items()}
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    tensor_payloads: List[bytes] = []
    current_offset = 0

    for name, val in state_dict.items():
        if has_torch and isinstance(val, torch.Tensor):
            t_cpu = val.detach().contiguous().cpu()
            dtype_str = str(t_cpu.dtype).replace("torch.", "")
            st_dtype = reverse_dtype.get(dtype_str, "U8")
            shape = list(t_cpu.shape)
            raw_bytes = bytes(
                 (ctypes_buf := t_cpu.numpy().tobytes())
                if t_cpu.dtype != torch.bfloat16
                else t_cpu.view(torch.uint16).numpy().tobytes()
            )
        elif isinstance(val, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(val)
            st_dtype = "U8"
            shape = [len(raw_bytes)]
        else:
            raise TypeError(f"Unsupported value type for safetensors export: {type(val)}")

        end_offset = current_offset + len(raw_bytes)
        header[name] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [current_offset, end_offset],
        }
        tensor_payloads.append(raw_bytes)
        current_offset = end_offset

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment per Safetensors spec
    pad_len = (8 - (len(header_bytes) % 8)) % 8
    if pad_len > 0:
        header_bytes += b" " * pad_len

    prefix = struct.pack("<Q", len(header_bytes)) + header_bytes

    fs = HyperCacheFileSystem(socket_path=socket_path, prefer_gds=False)
    total_written = 0
    with fs.open(uri, "wb", tier=tier, write_through=write_through, pinned=pinned) as wf:
        wf.write(prefix)
        total_written += len(prefix)
        for buf in tensor_payloads:
            wf.write(buf)
            total_written += len(buf)

    return total_written


class HyperCachevLLMWeightLoader:
    """
    High-throughput weight shard loader for vLLM and Hugging Face multi-shard checkpoints.
    Supports concurrent shard pre-warming, cache pinning, and selective tensor-parallel slicing.
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        prefer_gds: bool = True,
        pin_base_weights: bool = True,
    ):
        self.socket_path = socket_path
        self.prefer_gds = prefer_gds
        self.pin_base_weights = pin_base_weights
        self.fs = HyperCacheFileSystem(socket_path=socket_path, prefer_gds=prefer_gds)

    def iter_shard_tensors(
        self,
        shard_uris: List[str],
        device: str = "cpu",
        tensor_filter: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, Any]]:
        """
        Iterates over one or more `.safetensors` shards in HyperCache, yielding `(name, tensor)`
        pairs directly for vLLM `model.load_weights(...)`.
        """
        for uri in shard_uris:
            tensors = load_safetensors_state_dict(
                uri=uri,
                device=device,
                socket_path=self.socket_path,
                prefer_gds=self.prefer_gds,
                pinned=self.pin_base_weights,
                tensor_names=tensor_filter,
            )
            for name, tensor in tensors.items():
                yield name, tensor
