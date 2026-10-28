//! Synthetic tensor payloads for offline benchmarks and demos.
//!
//! Synthetic data is only ever served when explicitly allowed: either globally via
//! `allow_synthetic_s3_fallback = true`, or per request for URIs under
//! [`SYNTHETIC_URI_PREFIX`] when the client sets `allow_synthetic` (as `hypercache bench` does).

/// Reserved URI namespace used by the built-in benchmark for synthetic payloads.
pub const SYNTHETIC_URI_PREFIX: &str = "hypercache://models/benchmark/";

/// Extracts the target benchmark size from a URI pattern
/// (e.g. "weights_500mb.safetensors" -> 500 MiB, "weights_2gb" -> 2 GiB). Defaults to 20 MiB.
pub fn extract_size_from_uri(uri: &str) -> usize {
    if let Some(pos) = uri.find("weights_") {
        let after = &uri[pos + 8..];
        if let Some(end) = after.find("mb") {
            if let Ok(mb) = after[..end].parse::<usize>() {
                return mb * 1024 * 1024;
            }
        } else if let Some(end) = after.find("gb") {
            if let Ok(gb) = after[..end].parse::<usize>() {
                return gb * 1024 * 1024 * 1024;
            }
        }
    }
    20 * 1024 * 1024
}

/// Generates deterministic synthetic tensor bytes for offline demos.
pub fn generate_synthetic_payload(uri: &str, size: usize) -> Vec<u8> {
    let header = format!("HYPERCACHE_SYNTHETIC_TENSOR_HEADER:{}\n", uri);
    let mut data = Vec::with_capacity(size);
    data.extend_from_slice(&header.as_bytes()[..header.len().min(size)]);
    let pattern = b"0123456789ABCDEF_HYPERCACHE_BLOCK_DMA_";

    let mut block = Vec::with_capacity(1024 * 1024);
    while block.len() < 1024 * 1024 {
        block.extend_from_slice(pattern);
    }
    block.truncate(1024 * 1024);

    while data.len() + block.len() <= size {
        data.extend_from_slice(&block);
    }
    let remainder = size.saturating_sub(data.len());
    data.extend_from_slice(&block[..remainder]);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_sizes() {
        assert_eq!(
            extract_size_from_uri("hypercache://models/benchmark/nvme_weights_5mb.safetensors"),
            5 * 1024 * 1024
        );
        assert_eq!(extract_size_from_uri("x/weights_1gb.bin"), 1024 * 1024 * 1024);
        assert_eq!(extract_size_from_uri("x/model.bin"), 20 * 1024 * 1024);
        for size in [0usize, 10, 1024 * 1024 + 3] {
            assert_eq!(generate_synthetic_payload("u", size).len(), size);
        }
    }
}
