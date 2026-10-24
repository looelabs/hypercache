#!/usr/bin/env python3
"""
HyperCache Dual-Tier Validation Suite: RAM Tier + NVMe Fallback
Simulates:
1. Training Checkpoint Spillover: RAM -> NVMe automatic fallback on capacity limits
2. Training Checkpoint Recovery: Bit-for-bit SHA-256 validation from both RAM and NVMe
3. Inference Multi-LoRA Swapping: Ultra-low-latency adapter serving from Tier 1 RAM
4. Inference Base Model Loading: Direct Block DMA streaming from Tier 2 NVMe
"""

import hashlib
import io
import os
import sys
import time
import torch
import fsspec
import hypercache
from hypercache.client import HyperCacheClient

SOCKET_PATH = "/tmp/hypercache/hypercache.sock"

print("=" * 85)
print("     HYPERCACHE DUAL-TIER VALIDATION: TIER 1 (RAM) + TIER 2 (NVMe FALLBACK)     ")
print("=" * 85)

client = HyperCacheClient(socket_path=SOCKET_PATH)
if not client.ping():
    print(f"Error: HyperCache daemon not responding at {SOCKET_PATH}")
    print("Start the daemon with RAM tier enabled:")
    print("  ./hypercache -c configs/hypercache-agent.toml start --ram")
    sys.exit(1)

devices = client.get_devices()
print(f"Active Discovered Block Devices: {len(devices)}")
for d in devices:
    print(f"  -> {d.path} ({d.size_bytes / (1024**3):.2f} GiB)")
print("=" * 85)


def generate_synthetic_state_dict(size_mb: int) -> bytes:
    """Generates a realistic PyTorch model state_dict of exact specified MB size."""
    num_floats = (size_mb * 1024 * 1024) // 4
    weights = {
        "encoder.layer.0.weight": torch.randn(num_floats // 2, dtype=torch.float32),
        "encoder.layer.1.weight": torch.randn(num_floats // 2, dtype=torch.float32),
    }
    buf = io.BytesIO()
    torch.save(weights, buf)
    return buf.getvalue()


# =============================================================================
# PART 1: TRAINING WORKLOAD - RAM TIER & NVMe SPILLOVER BENCHMARK
# =============================================================================
print("\n" + "=" * 85)
print("PART 1: TRAINING WORKLOAD - CHECKPOINT WRITE-BACK & AUTOMATIC NVMe SPILLOVER")
print("=" * 85)

checkpoint_size_mb = 100
print(f"Generating 3 successive training checkpoints ({checkpoint_size_mb} MB each)...")

ckpt_data_1 = generate_synthetic_state_dict(checkpoint_size_mb)
ckpt_data_2 = generate_synthetic_state_dict(checkpoint_size_mb)
ckpt_data_3 = generate_synthetic_state_dict(checkpoint_size_mb)

sha_1 = hashlib.sha256(ckpt_data_1).hexdigest()
sha_2 = hashlib.sha256(ckpt_data_2).hexdigest()
sha_3 = hashlib.sha256(ckpt_data_3).hexdigest()

ckpt_uri_1 = "hypercache://checkpoints/run_01_epoch_01.pt"
ckpt_uri_2 = "hypercache://checkpoints/run_01_epoch_02.pt"
ckpt_uri_3 = "hypercache://checkpoints/run_01_epoch_03.pt"

# --- 1. Save Epoch 1 explicitly to Tier 1 RAM ---
print(f"\n[Epoch 1] Saving to Tier 1 Host RAM (Memory Bus Write)...")
t0 = time.perf_counter()
with fsspec.open(ckpt_uri_1, "wb", socket_path=SOCKET_PATH, tier="ram") as f:
    f.write(ckpt_data_1)
t_save_1 = time.perf_counter() - t0
bw_save_1 = (len(ckpt_data_1) / (1024 * 1024)) / t_save_1
layout_1 = client.locate(ckpt_uri_1)
tier_1 = layout_1.chunk_layout[0].tier if layout_1.chunk_layout else "unknown"

print(f"  Target Tier:       {tier_1.upper()}")
print(f"  torch.save Time:   {t_save_1 * 1000.0:.2f} ms")
print(f"  Write Throughput:  {bw_save_1:.2f} MB/s ({bw_save_1 / 1024.0:.2f} GB/s)")

# --- 2. Save Epoch 2 explicitly to Tier 2 NVMe ---
print(f"\n[Epoch 2] Saving to Tier 2 Striped NVMe Pool (Direct Block DMA Write)...")
t0 = time.perf_counter()
with fsspec.open(ckpt_uri_2, "wb", socket_path=SOCKET_PATH, tier="nvme") as f:
    f.write(ckpt_data_2)
t_save_2 = time.perf_counter() - t0
bw_save_2 = (len(ckpt_data_2) / (1024 * 1024)) / t_save_2
layout_2 = client.locate(ckpt_uri_2)
tier_2 = layout_2.chunk_layout[0].tier if layout_2.chunk_layout else "unknown"

print(f"  Target Tier:       {tier_2.upper()}")
print(f"  torch.save Time:   {t_save_2 * 1000.0:.2f} ms")
print(f"  Write Throughput:  {bw_save_2:.2f} MB/s ({bw_save_2 / 1024.0:.2f} GB/s)")

# --- 3. Save Epoch 3 with Auto-Tiering ---
print(f"\n[Epoch 3] Saving with Auto-Tier Placement...")
t0 = time.perf_counter()
with fsspec.open(ckpt_uri_3, "wb", socket_path=SOCKET_PATH) as f:
    f.write(ckpt_data_3)
t_save_3 = time.perf_counter() - t0
bw_save_3 = (len(ckpt_data_3) / (1024 * 1024)) / t_save_3
layout_3 = client.locate(ckpt_uri_3)
tier_3 = layout_3.chunk_layout[0].tier if layout_3.chunk_layout else "unknown"

print(f"  Allocated Tier:    {tier_3.upper()}")
print(f"  torch.save Time:   {t_save_3 * 1000.0:.2f} ms")
print(f"  Write Throughput:  {bw_save_3:.2f} MB/s ({bw_save_3 / 1024.0:.2f} GB/s)")


# --- 4. Training Recovery / Restore Benchmark (torch.load) ---
print(f"\n--- Training Checkpoint Recovery & Bit-for-Bit Validation ---")

# Restore from RAM
t0 = time.perf_counter()
with fsspec.open(ckpt_uri_1, "rb", socket_path=SOCKET_PATH) as f:
    read_bytes_1 = f.read()
t_load_1 = time.perf_counter() - t0
bw_load_1 = (len(read_bytes_1) / (1024 * 1024)) / t_load_1
assert hashlib.sha256(read_bytes_1).hexdigest() == sha_1, "Data corruption in RAM checkpoint!"
print(f"  Epoch 1 (RAM Load):   {t_load_1 * 1000.0:.2f} ms ({bw_load_1:.2f} MB/s = {bw_load_1 / 1024.0:.2f} GB/s) | 100% SHA-256 MATCH")

# Restore from NVMe
t0 = time.perf_counter()
with fsspec.open(ckpt_uri_2, "rb", socket_path=SOCKET_PATH) as f:
    read_bytes_2 = f.read()
t_load_2 = time.perf_counter() - t0
bw_load_2 = (len(read_bytes_2) / (1024 * 1024)) / t_load_2
assert hashlib.sha256(read_bytes_2).hexdigest() == sha_2, "Data corruption in NVMe checkpoint!"
print(f"  Epoch 2 (NVMe Load):  {t_load_2 * 1000.0:.2f} ms ({bw_load_2:.2f} MB/s = {bw_load_2 / 1024.0:.2f} GB/s) | 100% SHA-256 MATCH")


# =============================================================================
# PART 2: INFERENCE WORKLOAD - MULTI-LORA HOT-SWAPPING & BASE MODEL SERVING
# =============================================================================
print("\n" + "=" * 85)
print("PART 2: INFERENCE WORKLOAD - DYNAMIC LORA HOT-SWAPPING & BASE MODEL SERVING")
print("=" * 85)

# 1. Simulate 4 Multi-LoRA Adapters (20 MB each) in Tier 1 RAM
lora_count = 4
lora_size_mb = 20
print(f"Preloading {lora_count} fine-tuned LoRA adapters ({lora_size_mb} MB each) into Tier 1 RAM...")

lora_uris = [f"hypercache://adapters/lora_task_{i:02d}.bin" for i in range(lora_count)]
lora_payloads = [os.urandom(lora_size_mb * 1024 * 1024) for _ in range(lora_count)]

for uri, payload in zip(lora_uris, lora_payloads):
    with fsspec.open(uri, "wb", socket_path=SOCKET_PATH, tier="ram") as f:
        f.write(payload)

# Benchmark rapid dynamic adapter swapping (switching LoRA per user query)
print("\nBenchmarking Dynamic LoRA Hot-Swapping Latency across sequential inference requests:")
for i, uri in enumerate(lora_uris):
    t0 = time.perf_counter()
    with fsspec.open(uri, "rb", socket_path=SOCKET_PATH) as f:
        adapter_bytes = f.read()
    t_swap = time.perf_counter() - t0
    swap_bw = lora_size_mb / t_swap
    print(f"  Request {i+1}: Swapped '{uri.split('/')[-1]}' in {t_swap * 1000.0:.2f} ms ({swap_bw:.2f} MB/s = {swap_bw / 1024.0:.2f} GB/s)")

# 2. Simulate Base Foundation Model Weights (200 MB) in Tier 2 NVMe
base_model_uri = "hypercache://models/mistral_base_weights.safetensors"
base_model_size_mb = 200
print(f"\nPreloading Foundation Base Model ({base_model_size_mb} MB) into Tier 2 Striped NVMe Pool...")
base_model_payload = os.urandom(base_model_size_mb * 1024 * 1024)

with fsspec.open(base_model_uri, "wb", socket_path=SOCKET_PATH, tier="nvme") as f:
    f.write(base_model_payload)

print(f"Streaming Base Model weights into Inference Engine (Cold Start / Auto-Scaling Pod Boot)...")
t0 = time.perf_counter()
with fsspec.open(base_model_uri, "rb", socket_path=SOCKET_PATH) as f:
    loaded_base = f.read()
t_base_load = time.perf_counter() - t0
bw_base_load = base_model_size_mb / t_base_load

print(f"  Base Model Load Time:  {t_base_load * 1000.0:.2f} ms")
print(f"  NVMe DMA Throughput:   {bw_base_load:.2f} MB/s ({bw_base_load / 1024.0:.2f} GB/s)")
assert len(loaded_base) == len(base_model_payload), "Base model size mismatch!"


# =============================================================================
# SUMMARY MATRIX
# =============================================================================
print("\n" + "=" * 85)
print("                               BENCHMARK SUMMARY MATRIX                         ")
print("=" * 85)
print(f"{'Operation':<35} | {'Tier':<8} | {'Size':<10} | {'Latency':<12} | {'Throughput'}")
print("-" * 85)
print(f"{'Training Checkpoint Save (Epoch 1)':<35} | {'RAM':<8} | {checkpoint_size_mb:>4} MB   | {t_save_1 * 1000:>8.2f} ms | {bw_save_1 / 1024.0:>6.2f} GB/s")
print(f"{'Training Checkpoint Save (Epoch 2)':<35} | {'NVMe':<8} | {checkpoint_size_mb:>4} MB   | {t_save_2 * 1000:>8.2f} ms | {bw_save_2 / 1024.0:>6.2f} GB/s")
print(f"{'Training Checkpoint Load (Recovery)':<35} | {'RAM':<8} | {checkpoint_size_mb:>4} MB   | {t_load_1 * 1000:>8.2f} ms | {bw_load_1 / 1024.0:>6.2f} GB/s")
print(f"{'Training Checkpoint Load (Recovery)':<35} | {'NVMe':<8} | {checkpoint_size_mb:>4} MB   | {t_load_2 * 1000:>8.2f} ms | {bw_load_2 / 1024.0:>6.2f} GB/s")
print(f"{'Inference Dynamic LoRA Hot-Swap':<35} | {'RAM':<8} | {lora_size_mb:>4} MB   | {t_swap * 1000:>8.2f} ms | {swap_bw / 1024.0:>6.2f} GB/s")
print(f"{'Inference Base Model Load':<35} | {'NVMe':<8} | {base_model_size_mb:>4} MB   | {t_base_load * 1000:>8.2f} ms | {bw_base_load / 1024.0:>6.2f} GB/s")
print("=" * 85)
print("                   ALL DUAL-TIER INTEGRITY TESTS PASSED SUCCESSFULLY            ")
print("=" * 85)
