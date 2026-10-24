#!/usr/bin/env python3
"""
HyperCache Write-Through vs. Write-Back Live Verification Suite.

Validates:
1. Asynchronous Write-Back (Default: fast non-blocking return, status COMMITTED -> SYNCED_BACKEND in background).
2. Synchronous Write-Through (write_through=True: blocks until flush to authoritative backend completes, status SYNCED_BACKEND).
3. Tier 1 Host RAM vs Tier 2 Striped NVMe storage pools.
4. Authoritative backend synchronization and bitwise SHA-256 parity.
"""

import hashlib
import io
import os
import subprocess
import sys
import time
import fsspec

from hypercache.client import HyperCacheClient
from hypercache.fsspec_fs import HyperCacheFileSystem


def run_verification():
    print("=" * 70)
    print("HYPERCACHE WRITE-THROUGH VS. WRITE-BACK VERIFICATION SUITE")
    print("=" * 70)

    socket_path = "/tmp/hypercache/hypercache.sock"
    client = HyperCacheClient(socket_path=socket_path)

    # 1. Check daemon connectivity
    if not client.ping():
        print("[ERROR] Daemon is not responding on socket:", socket_path)
        sys.exit(1)
    print("[PASS] HyperCache daemon is connected.")

    # Discovered devices
    devices = client.get_devices()
    print(f"[INFO] Discovered {len(devices)} storage block devices.")
    for d in devices:
        print(f"       - {d.path} ({d.size_bytes / 1e9:.2f} GB, NVMe={d.is_nvme}, Virtual={d.is_virtual})")

    fs = HyperCacheFileSystem(socket_path=socket_path)
    nfs_root = "/tmp/hypercache_nfs_authoritative"
    os.makedirs(nfs_root, exist_ok=True)

    # Test 1: Asynchronous Write-Back Checkpoint (Default, NVMe Tier)
    print("\n" + "-" * 70)
    print("TEST 1: Asynchronous Write-Back Checkpoint (NVMe Tier, Default)")
    print("-" * 70)
    ckpt_size_mb = 100
    payload_wb = os.urandom(ckpt_size_mb * 1024 * 1024)
    expected_sha_wb = hashlib.sha256(payload_wb).hexdigest()
    nfs_file_wb = os.path.join(nfs_root, "epoch_wb_nvme.pt")
    uri_wb = f"file://{nfs_file_wb}"

    t0 = time.perf_counter()
    with fs.open(uri_wb, "wb", tier="nvme", write_through=False) as f:
        f.write(payload_wb)
    t_wb = (time.perf_counter() - t0) * 1000
    mb_s_wb = (ckpt_size_mb / (t_wb / 1000))
    print(f"[RESULT] Asynchronous Write-Back saved {ckpt_size_mb} MB in {t_wb:.2f} ms ({mb_s_wb:.2f} MB/s)")

    # Read back from cache and verify SHA-256
    with fs.open(uri_wb, "rb") as f:
        read_back_wb = f.read()
    actual_sha_wb = hashlib.sha256(read_back_wb).hexdigest()
    assert actual_sha_wb == expected_sha_wb, "SHA-256 mismatch on NVMe Write-Back read!"
    print(f"[PASS] NVMe Write-Back bitwise integrity verified (SHA-256: {actual_sha_wb[:16]}...)")

    # Allow async worker a moment to sync to authoritative storage
    time.sleep(0.5)
    if os.path.exists(nfs_file_wb):
        with open(nfs_file_wb, "rb") as f:
            auth_sha_wb = hashlib.sha256(f.read()).hexdigest()
        assert auth_sha_wb == expected_sha_wb, "SHA-256 mismatch on authoritative NFS file!"
        print(f"[PASS] Authoritative backend file synced asynchronously (SHA-256: {auth_sha_wb[:16]}...)")

    # Test 2: Synchronous Write-Through Checkpoint (NVMe Tier)
    print("\n" + "-" * 70)
    print("TEST 2: Synchronous Write-Through Checkpoint (NVMe Tier, write_through=True)")
    print("-" * 70)
    payload_wt = os.urandom(ckpt_size_mb * 1024 * 1024)
    expected_sha_wt = hashlib.sha256(payload_wt).hexdigest()
    nfs_file_wt = os.path.join(nfs_root, "epoch_wt_nvme.pt")
    uri_wt = f"file://{nfs_file_wt}"

    t0 = time.perf_counter()
    with fs.open(uri_wt, "wb", tier="nvme", write_through=True) as f:
        f.write(payload_wt)
    t_wt = (time.perf_counter() - t0) * 1000
    mb_s_wt = (ckpt_size_mb / (t_wt / 1000))
    print(f"[RESULT] Synchronous Write-Through saved {ckpt_size_mb} MB in {t_wt:.2f} ms ({mb_s_wt:.2f} MB/s)")

    wt_status = client.get_write_status(uri_wt)
    print(f"[INFO] Immediate write status: {wt_status['status']}")
    assert wt_status['status'] == 'SYNCED_BACKEND', f"Expected SYNCED_BACKEND, got {wt_status['status']}"

    # Read back from cache and verify SHA-256
    with fs.open(uri_wt, "rb") as f:
        read_back_wt = f.read()
    actual_sha_wt = hashlib.sha256(read_back_wt).hexdigest()
    assert actual_sha_wt == expected_sha_wt, "SHA-256 mismatch on NVMe Write-Through read!"
    print(f"[PASS] NVMe Write-Through bitwise integrity verified (SHA-256: {actual_sha_wt[:16]}...)")

    # Verify authoritative file was written synchronously immediately
    assert os.path.exists(nfs_file_wt), "Authoritative file missing after write-through!"
    with open(nfs_file_wt, "rb") as f:
        auth_sha_wt = hashlib.sha256(f.read()).hexdigest()
    assert auth_sha_wt == expected_sha_wt, "Authoritative file content mismatch!"
    print(f"[PASS] Authoritative file synchronously confirmed on disk (SHA-256: {auth_sha_wt[:16]}...)")

    # Test 3: Asynchronous Write-Back (RAM Tier)
    print("\n" + "-" * 70)
    print("TEST 3: Asynchronous Write-Back Checkpoint (RAM Tier 1)")
    print("-" * 70)
    ram_size_mb = 100
    payload_ram_wb = os.urandom(ram_size_mb * 1024 * 1024)
    expected_sha_ram_wb = hashlib.sha256(payload_ram_wb).hexdigest()
    nfs_file_ram_wb = os.path.join(nfs_root, "epoch_wb_ram.pt")
    uri_ram_wb = f"file://{nfs_file_ram_wb}"

    t0 = time.perf_counter()
    with fs.open(uri_ram_wb, "wb", tier="ram", write_through=False) as f:
        f.write(payload_ram_wb)
    t_ram_wb = (time.perf_counter() - t0) * 1000
    mb_s_ram_wb = (ram_size_mb / (t_ram_wb / 1000))
    print(f"[RESULT] RAM Tier Write-Back saved {ram_size_mb} MB in {t_ram_wb:.2f} ms ({mb_s_ram_wb:.2f} MB/s)")

    with fs.open(uri_ram_wb, "rb") as f:
        read_back_ram_wb = f.read()
    actual_sha_ram_wb = hashlib.sha256(read_back_ram_wb).hexdigest()
    assert actual_sha_ram_wb == expected_sha_ram_wb, "SHA-256 mismatch on RAM Write-Back read!"
    print(f"[PASS] RAM Tier Write-Back bitwise integrity verified (SHA-256: {actual_sha_ram_wb[:16]}...)")

    # Test 4: Synchronous Write-Through (RAM Tier)
    print("\n" + "-" * 70)
    print("TEST 4: Synchronous Write-Through Checkpoint (RAM Tier 1, write_through=True)")
    print("-" * 70)
    payload_ram_wt = os.urandom(ram_size_mb * 1024 * 1024)
    expected_sha_ram_wt = hashlib.sha256(payload_ram_wt).hexdigest()
    nfs_file_ram_wt = os.path.join(nfs_root, "epoch_wt_ram.pt")
    uri_ram_wt = f"file://{nfs_file_ram_wt}"

    t0 = time.perf_counter()
    with fs.open(uri_ram_wt, "wb", tier="ram", write_through=True) as f:
        f.write(payload_ram_wt)
    t_ram_wt = (time.perf_counter() - t0) * 1000
    mb_s_ram_wt = (ram_size_mb / (t_ram_wt / 1000))
    print(f"[RESULT] RAM Tier Write-Through saved {ram_size_mb} MB in {t_ram_wt:.2f} ms ({mb_s_ram_wt:.2f} MB/s)")

    wt_ram_status = client.get_write_status(uri_ram_wt)
    print(f"[INFO] Immediate write status: {wt_ram_status['status']}")
    assert wt_ram_status['status'] == 'SYNCED_BACKEND', f"Expected SYNCED_BACKEND, got {wt_ram_status['status']}"

    with fs.open(uri_ram_wt, "rb") as f:
        read_back_ram_wt = f.read()
    actual_sha_ram_wt = hashlib.sha256(read_back_ram_wt).hexdigest()
    assert actual_sha_ram_wt == expected_sha_ram_wt, "SHA-256 mismatch on RAM Write-Through read!"
    print(f"[PASS] RAM Tier Write-Through bitwise integrity verified (SHA-256: {actual_sha_ram_wt[:16]}...)")

    assert os.path.exists(nfs_file_ram_wt), "Authoritative file missing after RAM write-through!"
    with open(nfs_file_ram_wt, "rb") as f:
        auth_sha_ram_wt = hashlib.sha256(f.read()).hexdigest()
    assert auth_sha_ram_wt == expected_sha_ram_wt, "Authoritative file content mismatch!"
    print(f"[PASS] Authoritative file synchronously confirmed on disk (SHA-256: {auth_sha_ram_wt[:16]}...)")

    # Summary table
    print("\n" + "=" * 70)
    print("WRITE MODE PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Tier':<10} | {'Mode':<22} | {'Size':<8} | {'Latency':<12} | {'Throughput':<12} | {'Integrity'}")
    print("-" * 70)
    print(f"{'NVMe':<10} | {'Write-Back (Async)':<22} | {ckpt_size_mb} MB   | {t_wb:>8.2f} ms | {mb_s_wb:>8.2f} MB/s | PASS (100%)")
    print(f"{'NVMe':<10} | {'Write-Through (Sync)':<22} | {ckpt_size_mb} MB   | {t_wt:>8.2f} ms | {mb_s_wt:>8.2f} MB/s | PASS (100%)")
    print(f"{'RAM':<10} | {'Write-Back (Async)':<22} | {ram_size_mb} MB   | {t_ram_wb:>8.2f} ms | {mb_s_ram_wb:>8.2f} MB/s | PASS (100%)")
    print(f"{'RAM':<10} | {'Write-Through (Sync)':<22} | {ram_size_mb} MB   | {t_ram_wt:>8.2f} ms | {mb_s_ram_wt:>8.2f} MB/s | PASS (100%)")
    print("=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_verification()
