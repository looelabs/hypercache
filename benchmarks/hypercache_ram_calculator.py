#!/usr/bin/env python3
"""
HyperCache Host RAM Consumption Calculator & Node Memory Profiler.

Calculates the exact Host RAM footprint of HyperCache on a GPU compute node:
1. Control Plane Daemon Baseline RSS (Tokio runtime, UDS listener, metrics).
2. Metadata Store (Moka W-TinyLFU cache + chunk routing map).
3. Tier 1 Host RAM / CXL Shared Memory Cache (/dev/shm).
4. Multi-Tenant MAP_SHARED memory deduplication savings vs. Linux Page Cache.
"""

import argparse
import sys
from typing import Dict, Any


def calculate_ram_footprint(
    nvme_pool_tb: float = 4.0,
    ram_tier_gb: float = 32.0,
    ram_tier_enabled: bool = True,
    num_models: int = 50,
    chunk_size_mb: float = 4.0,
    concurrent_streams: int = 4,
    num_tenants: int = 4,
) -> Dict[str, Any]:
    """
    Computes exact host RAM requirements for HyperCache components.
    """
    # 1. Base Rust Control Plane Daemon RSS
    # Tokio async runtime threadpool (16 threads * 2MB stack + heap) + Axum UDS listener + Prometheus registry
    base_daemon_mb = 28.0

    # 2. Metadata Catalog (Moka W-TinyLFU + Chunk Routing Map)
    # StoredMetadata struct: ~128 bytes per object
    # ChunkDescriptor: ~80 bytes per chunk (4 MiB default)
    total_nvme_bytes = nvme_pool_tb * (1024 ** 4)
    total_chunks = total_nvme_bytes / (chunk_size_mb * 1024 * 1024)
    chunk_descriptor_bytes = 80.0
    object_metadata_bytes = 128.0

    chunk_map_mb = (total_chunks * chunk_descriptor_bytes) / (1024 * 1024)
    object_meta_mb = (num_models * object_metadata_bytes) / (1024 * 1024)
    moka_filter_overhead_mb = 16.0 # TinyLFU CountMinSketch + frequency sketches

    total_metadata_mb = chunk_map_mb + object_meta_mb + moka_filter_overhead_mb

    # 3. Transient Streaming / Direct DMA Buffers
    # O_DIRECT bypasses page cache for NVMe; S3/NFS ingestion workers use fixed 4 MiB ring buffers
    transient_buffers_mb = concurrent_streams * chunk_size_mb

    # 4. Tier 1 Host RAM Data Cache (/dev/shm)
    active_ram_tier_mb = (ram_tier_gb * 1024) if ram_tier_enabled else 0.0

    # Total HyperCache Memory Footprint
    control_plane_and_meta_mb = base_daemon_mb + total_metadata_mb + transient_buffers_mb
    total_max_host_ram_mb = control_plane_and_meta_mb + active_ram_tier_mb
    total_max_host_ram_gb = total_max_host_ram_mb / 1024.0

    # Multi-Tenant Savings Analysis
    # Standard PyTorch / HuggingFace page cache copies vs. HyperCache MAP_SHARED
    traditional_ram_used_mb = active_ram_tier_mb * num_tenants
    hypercache_ram_used_mb = active_ram_tier_mb
    ram_saved_gb = max(0.0, (traditional_ram_used_mb - hypercache_ram_used_mb) / 1024.0)

    return {
        "base_daemon_mb": base_daemon_mb,
        "total_chunks": int(total_chunks),
        "chunk_map_mb": chunk_map_mb,
        "object_meta_mb": object_meta_mb,
        "moka_filter_overhead_mb": moka_filter_overhead_mb,
        "total_metadata_mb": total_metadata_mb,
        "transient_buffers_mb": transient_buffers_mb,
        "control_plane_and_meta_mb": control_plane_and_meta_mb,
        "active_ram_tier_mb": active_ram_tier_mb,
        "active_ram_tier_gb": active_ram_tier_mb / 1024.0,
        "total_max_host_ram_mb": total_max_host_ram_mb,
        "total_max_host_ram_gb": total_max_host_ram_gb,
        "metadata_to_storage_ratio": (total_metadata_mb / (nvme_pool_tb * 1024 * 1024)) * 100,
        "num_tenants": num_tenants,
        "ram_saved_gb": ram_saved_gb,
    }


def print_report(res: Dict[str, Any], nvme_pool_tb: float, ram_tier_gb: float, ram_tier_enabled: bool, num_tenants: int):
    print("=" * 80)
    print("           HYPERCACHE HOST RAM CONSUMPTION & SIZING BREAKDOWN            ")
    print("=" * 80)
    print(f"Node Configuration Profile:")
    print(f"  - NVMe Striped Storage Pool:       {nvme_pool_tb:.2f} TB ({res['total_chunks']:,} x 4 MiB chunks)")
    print(f"  - Tier 1 Host RAM Cache:           {'ENABLED (' + str(ram_tier_gb) + ' GB)' if ram_tier_enabled else 'DISABLED (0 GB)'}")
    print(f"  - Concurrent Container Tenants:    {num_tenants} processes / inference workers")
    print("-" * 80)
    print("Component Breakdown:                                 Allocated Host RAM")
    print("-" * 80)
    print(f"1. Control Plane Daemon (Rust/Tokio Base RSS):       {res['base_daemon_mb']:>10.2f} MB")
    print(f"2. Metadata Store (Moka W-TinyLFU Catalog):          {res['total_metadata_mb']:>10.2f} MB")
    print(f"     - Physical Chunk Routing Table:                 {res['chunk_map_mb']:>10.2f} MB")
    print(f"     - Object Descriptors & Sync Headers:            {res['object_meta_mb']:>10.2f} MB")
    print(f"     - Moka Frequency & LRU Overhead:                {res['moka_filter_overhead_mb']:>10.2f} MB")
    print(f"3. Direct DMA & Transient Stream Buffers:            {res['transient_buffers_mb']:>10.2f} MB")
    print("-" * 80)
    print(f"-> TOTAL CONTROL PLANE & METADATA OVERHEAD:          {res['control_plane_and_meta_mb']:>10.2f} MB ({res['control_plane_and_meta_mb'] / 1024.0:.3f} GB)")
    print(f"   (Metadata-to-Storage Ratio: Only {res['metadata_to_storage_ratio']:.4f}% of NVMe capacity)")
    print("-" * 80)
    if ram_tier_enabled:
        print(f"4. Tier 1 Shared Memory Cache Pool (/dev/shm):       {res['active_ram_tier_mb']:>10.2f} MB ({res['active_ram_tier_gb']:.2f} GB)")
    else:
        print(f"4. Tier 1 Shared Memory Cache Pool:                  {0.0:>10.2f} MB (Disabled)")
    print("=" * 80)
    print(f"TOTAL MAXIMUM HOST RAM FOOTPRINT:                    {res['total_max_host_ram_gb']:>10.2f} GB ({res['total_max_host_ram_mb']:,.2f} MB)")
    print("=" * 80)
    if ram_tier_enabled and num_tenants > 1:
        print(f"\nMulti-Tenant Zero-Copy Sharing Savings (MAP_SHARED):")
        print(f"  - Traditional File System / Page Cache ({num_tenants} tenants): {res['active_ram_tier_gb'] * num_tenants:>8.2f} GB")
        print(f"  - HyperCache Shared Memory Tier ({num_tenants} tenants):         {res['active_ram_tier_gb']:>8.2f} GB")
        print(f"  - HOST RAM SAVED (Zero Duplicate Copies):          {res['ram_saved_gb']:>8.2f} GB")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Calculate HyperCache host RAM consumption on GPU nodes.")
    parser.add_argument("--nvme-tb", type=float, default=4.0, help="Total NVMe storage capacity in Terabytes (default: 4.0)")
    parser.add_argument("--ram-tier-gb", type=float, default=32.0, help="Tier 1 Host RAM capacity in Gigabytes (default: 32.0)")
    parser.add_argument("--no-ram-tier", action="store_true", help="Disable Tier 1 Host RAM cache (NVMe-only mode)")
    parser.add_argument("--models", type=int, default=100, help="Number of cached model checkpoints/datasets (default: 100)")
    parser.add_argument("--tenants", type=int, default=4, help="Number of concurrent container inference workers (default: 4)")
    args = parser.parse_args()

    ram_tier_enabled = not args.no_ram_tier
    res = calculate_ram_footprint(
        nvme_pool_tb=args.nvme_tb,
        ram_tier_gb=args.ram_tier_gb,
        ram_tier_enabled=ram_tier_enabled,
        num_models=args.models,
        num_tenants=args.tenants,
    )
    print_report(res, args.nvme_tb, args.ram_tier_gb, ram_tier_enabled, args.tenants)


if __name__ == "__main__":
    main()
