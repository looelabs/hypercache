#!/usr/bin/env bash
set -euo pipefail

# HyperCache: Setup mock raw block files for testing/development
DEVICE_COUNT=4
DEVICE_SIZE_MB=100

echo "=== HyperCache: Provisioning ${DEVICE_COUNT} mock block devices (${DEVICE_SIZE_MB}MB each) ==="

for i in $(seq 0 $((DEVICE_COUNT - 1))); do
    DEV_PATH="/tmp/hypercache_nvme${i}.raw"
    if [ -f "$DEV_PATH" ]; then
        echo "Device $DEV_PATH already exists ($(ls -lh "$DEV_PATH" | awk '{print $5}'))"
    else
        echo "Creating sparse raw device: $DEV_PATH"
        # Create sparse file
        dd if=/dev/zero of="$DEV_PATH" bs=1M count=0 seek=$DEVICE_SIZE_MB 2>/dev/null || truncate -s ${DEVICE_SIZE_MB}M "$DEV_PATH"
        chmod 666 "$DEV_PATH"
    fi
done

echo "=== Mock block devices ready in /tmp/ ==="
ls -lh /tmp/hypercache_nvme*.raw
