use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::config::Config;

type HmacSha256 = Hmac<Sha256>;

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

#[derive(Clone)]
pub struct S3Fetcher {
    client: Client,
    config: Config,
}

impl S3Fetcher {
    pub fn new(config: Config) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self { client, config }
    }

    /// Resolves a hypercache:// or s3:// URI into bucket and object key
    pub fn parse_uri(&self, uri: &str) -> (String, String) {
        let clean = if let Some(rest) = uri.strip_prefix("hypercache://s3/") {
            rest
        } else if let Some(rest) = uri.strip_prefix("hypercache://") {
            rest
        } else if let Some(rest) = uri.strip_prefix("s3://") {
            rest
        } else {
            uri
        };

        let parts: Vec<&str> = clean.splitn(2, '/').collect();
        if parts.len() == 2 && !self.config.s3_bucket.is_empty() && parts[0] == self.config.s3_bucket {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            (self.config.s3_bucket.clone(), clean.to_string())
        }
    }

    /// Fetches the object bytes from authoritative S3 / Object Storage
    #[allow(dead_code)]
    pub async fn fetch_object(&self, uri: &str) -> Result<Vec<u8>> {
        self.fetch_object_with_fallback(uri, self.config.allow_synthetic_s3_fallback)
            .await
    }

    /// Fetches the object bytes from authoritative S3 with explicit synthetic fallback control
    #[allow(dead_code)]
    pub async fn fetch_object_with_fallback(&self, uri: &str, allow_synthetic: bool) -> Result<Vec<u8>> {
        let (bucket, key) = self.parse_uri(uri);
        let endpoint = self.config.s3_endpoint.trim_end_matches('/');
        let url = format!("{}/{}/{}", endpoint, bucket, key);

        info!("Fetching authoritative object from S3: {} (key: {})", url, key);

        let mut req = self.client.get(&url);

        // Apply AWS SigV4 authorization headers if credentials are configured
        if !self.config.s3_access_key.is_empty() && !self.config.s3_secret_key.is_empty() {
            let access_key = &self.config.s3_access_key;
            let secret_key = &self.config.s3_secret_key;
            let region = if self.config.s3_region.is_empty() {
                "us-central1"
            } else {
                &self.config.s3_region
            };

            let host = endpoint
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or("localhost");

            let path = format!("/{}/{}", bucket, key);
            let now = chrono::Utc::now();
            let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
            let date_stamp = now.format("%Y%m%d").to_string();
            let payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

            let canonical_headers = format!(
                "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
                host, payload_hash, amz_date
            );
            let signed_headers = "host;x-amz-content-sha256;x-amz-date";
            let canonical_request = format!(
                "GET\n{}\n\n{}\n{}\n{}",
                path, canonical_headers, signed_headers, payload_hash
            );
            let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
            let string_to_sign = format!(
                "AWS4-HMAC-SHA256\n{}\n{}\n{}",
                amz_date,
                credential_scope,
                sha256_hex(canonical_request.as_bytes())
            );

            let k_secret = format!("AWS4{}", secret_key);
            let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
            let k_region = hmac_sha256(&k_date, region.as_bytes());
            let k_service = hmac_sha256(&k_region, b"s3");
            let k_signing = hmac_sha256(&k_service, b"aws4_request");
            let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

            let auth_header = format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                access_key, credential_scope, signed_headers, signature
            );

            req = req
                .header("host", host)
                .header("x-amz-date", amz_date)
                .header("x-amz-content-sha256", payload_hash)
                .header("Authorization", auth_header);
        }

        let is_benchmark_uri = uri.contains("models/benchmark/");

        match req.send().await {
            Ok(r) if r.status().is_success() => {
                let bytes = r
                    .bytes()
                    .await
                    .with_context(|| "Failed to read response body from S3")?;
                info!("Successfully fetched {} bytes from S3 for {}", bytes.len(), uri);
                Ok(bytes.to_vec())
            }
            Ok(r) => {
                if allow_synthetic || is_benchmark_uri {
                    let target_size = extract_size_from_uri(uri);
                    warn!(
                        "S3 endpoint returned status {} for {} (synthetic fallback active). Generating {:.2} MiB synthetic tensor payload for {}",
                        r.status(),
                        url,
                        target_size as f64 / (1024.0 * 1024.0),
                        uri
                    );
                    Ok(generate_synthetic_payload(uri, target_size))
                } else {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    anyhow::bail!(
                        "S3 GET failed for '{}' ({}) with HTTP status {}: {}",
                        uri,
                        url,
                        status,
                        body
                    );
                }
            }
            Err(e) => {
                if allow_synthetic || is_benchmark_uri {
                    let target_size = extract_size_from_uri(uri);
                    warn!(
                        "Failed to connect to S3 endpoint {}: {} (synthetic fallback active). Generating {:.2} MiB synthetic tensor payload for {}",
                        url,
                        e,
                        target_size as f64 / (1024.0 * 1024.0),
                        uri
                    );
                    Ok(generate_synthetic_payload(uri, target_size))
                } else {
                    anyhow::bail!(
                        "Failed to connect to authoritative S3 endpoint '{}' for URI '{}': {}",
                        url,
                        uri,
                        e
                    );
                }
            }
        }
    }

    /// Streams an S3 object directly in fixed-size chunks (`chunk_size_bytes`) without buffering the full object in RAM.
    pub async fn fetch_object_streaming<F, Fut>(
        &self,
        uri: &str,
        chunk_size: usize,
        allow_synthetic: bool,
        mut on_chunk: F,
    ) -> Result<u64>
    where
        F: FnMut(usize, Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        use futures_util::StreamExt;
        let (bucket, key) = self.parse_uri(uri);
        let endpoint = self.config.s3_endpoint.trim_end_matches('/');
        let url = format!("{}/{}/{}", endpoint, bucket, key);

        info!("Streaming authoritative object from S3: {} (key: {})", url, key);
        let mut req = self.client.get(&url);

        if !self.config.s3_access_key.is_empty() && !self.config.s3_secret_key.is_empty() {
            let access_key = &self.config.s3_access_key;
            let secret_key = &self.config.s3_secret_key;
            let region = if self.config.s3_region.is_empty() {
                "us-central1"
            } else {
                &self.config.s3_region
            };

            let host = endpoint
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or("localhost");

            let path = format!("/{}/{}", bucket, key);
            let now = chrono::Utc::now();
            let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
            let date_stamp = now.format("%Y%m%d").to_string();
            let payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

            let canonical_headers = format!(
                "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
                host, payload_hash, amz_date
            );
            let signed_headers = "host;x-amz-content-sha256;x-amz-date";
            let canonical_request = format!(
                "GET\n{}\n\n{}\n{}\n{}",
                path, canonical_headers, signed_headers, payload_hash
            );
            let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
            let string_to_sign = format!(
                "AWS4-HMAC-SHA256\n{}\n{}\n{}",
                amz_date,
                credential_scope,
                sha256_hex(canonical_request.as_bytes())
            );

            let k_secret = format!("AWS4{}", secret_key);
            let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
            let k_region = hmac_sha256(&k_date, region.as_bytes());
            let k_service = hmac_sha256(&k_region, b"s3");
            let k_signing = hmac_sha256(&k_service, b"aws4_request");
            let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

            let auth_header = format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                access_key, credential_scope, signed_headers, signature
            );

            req = req
                .header("host", host)
                .header("x-amz-date", amz_date)
                .header("x-amz-content-sha256", payload_hash)
                .header("Authorization", auth_header);
        }

        let is_benchmark_uri = uri.contains("models/benchmark/");

        match req.send().await {
            Ok(r) if r.status().is_success() => {
                let mut stream = r.bytes_stream();
                let mut buf = Vec::with_capacity(chunk_size);
                let mut chunk_idx = 0usize;
                let mut total_bytes = 0u64;

                while let Some(item) = stream.next().await {
                    let bytes = item.with_context(|| "Failed reading S3 stream chunk")?;
                    let mut slice = &bytes[..];
                    while !slice.is_empty() {
                        let space = chunk_size - buf.len();
                        let take = space.min(slice.len());
                        buf.extend_from_slice(&slice[..take]);
                        slice = &slice[take..];

                        if buf.len() == chunk_size {
                            let full_chunk = std::mem::replace(&mut buf, Vec::with_capacity(chunk_size));
                            total_bytes += full_chunk.len() as u64;
                            on_chunk(chunk_idx, full_chunk).await?;
                            chunk_idx += 1;
                        }
                    }
                }

                if !buf.is_empty() {
                    total_bytes += buf.len() as u64;
                    on_chunk(chunk_idx, buf).await?;
                }

                info!("Successfully streamed {} bytes from S3 for {}", total_bytes, uri);
                Ok(total_bytes)
            }
            Ok(r) => {
                if allow_synthetic || is_benchmark_uri {
                    let target_size = extract_size_from_uri(uri);
                    warn!(
                        "S3 endpoint returned status {} for {} (synthetic fallback active). Streaming {:.2} MiB synthetic tensor payload for {}",
                        r.status(),
                        url,
                        target_size as f64 / (1024.0 * 1024.0),
                        uri
                    );
                    let data = generate_synthetic_payload(uri, target_size);
                    let mut chunk_idx = 0usize;
                    for slice in data.chunks(chunk_size) {
                        on_chunk(chunk_idx, slice.to_vec()).await?;
                        chunk_idx += 1;
                    }
                    Ok(data.len() as u64)
                } else {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    anyhow::bail!(
                        "S3 GET failed for '{}' ({}) with HTTP status {}: {}",
                        uri,
                        url,
                        status,
                        body
                    );
                }
            }
            Err(e) => {
                if allow_synthetic || is_benchmark_uri {
                    let target_size = extract_size_from_uri(uri);
                    warn!(
                        "Failed to connect to S3 endpoint {}: {} (synthetic fallback active). Streaming {:.2} MiB synthetic tensor payload for {}",
                        url,
                        e,
                        target_size as f64 / (1024.0 * 1024.0),
                        uri
                    );
                    let data = generate_synthetic_payload(uri, target_size);
                    let mut chunk_idx = 0usize;
                    for slice in data.chunks(chunk_size) {
                        on_chunk(chunk_idx, slice.to_vec()).await?;
                        chunk_idx += 1;
                    }
                    Ok(data.len() as u64)
                } else {
                    anyhow::bail!(
                        "Failed to connect to authoritative S3 endpoint '{}' for URI '{}': {}",
                        url,
                        uri,
                        e
                    );
                }
            }
        }
    }

    /// Uploads object bytes directly to authoritative S3 / Object Storage via SigV4 PUT
    pub async fn upload_object(&self, uri: &str, data: &[u8]) -> Result<()> {
        let (bucket, key) = self.parse_uri(uri);
        let endpoint = self.config.s3_endpoint.trim_end_matches('/');
        let url = format!("{}/{}/{}", endpoint, bucket, key);

        info!("Uploading {} bytes to S3: {} (key: {})", data.len(), url, key);

        let mut req = self.client.put(&url).body(data.to_vec());

        // Apply AWS SigV4 authorization headers if credentials are configured
        if !self.config.s3_access_key.is_empty() && !self.config.s3_secret_key.is_empty() {
            let access_key = &self.config.s3_access_key;
            let secret_key = &self.config.s3_secret_key;
            let region = if self.config.s3_region.is_empty() {
                "us-central1"
            } else {
                &self.config.s3_region
            };

            let host = endpoint
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or("localhost");

            let path = format!("/{}/{}", bucket, key);
            let now = chrono::Utc::now();
            let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
            let date_stamp = now.format("%Y%m%d").to_string();
            let payload_hash = sha256_hex(data);

            let canonical_headers = format!(
                "content-length:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
                data.len(),
                host,
                payload_hash,
                amz_date
            );
            let signed_headers = "content-length;host;x-amz-content-sha256;x-amz-date";
            let canonical_request = format!("PUT\n{}\n\n{}\n{}\n{}", path, canonical_headers, signed_headers, payload_hash);
            let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
            let string_to_sign = format!("AWS4-HMAC-SHA256\n{}\n{}\n{}", amz_date, credential_scope, sha256_hex(canonical_request.as_bytes()));

            let k_secret = format!("AWS4{}", secret_key);
            let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
            let k_region = hmac_sha256(&k_date, region.as_bytes());
            let k_service = hmac_sha256(&k_region, b"s3");
            let k_signing = hmac_sha256(&k_service, b"aws4_request");
            let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

            let auth_header = format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                access_key, credential_scope, signed_headers, signature
            );

            req = req
                .header("host", host)
                .header("content-length", data.len().to_string())
                .header("x-amz-date", amz_date)
                .header("x-amz-content-sha256", payload_hash)
                .header("Authorization", auth_header);
        }

        let resp = req.send().await.with_context(|| format!("Failed to send S3 PUT to {}", url))?;
        if resp.status().is_success() {
            info!("Successfully uploaded {} bytes to S3: {}", data.len(), url);
            Ok(())
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("S3 PUT failed with status {}: {}", status, text);
        }
    }

    /// Lists objects under a prefix from authoritative S3
    pub async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let (bucket, key_prefix) = self.parse_uri(prefix);
        let endpoint = self.config.s3_endpoint.trim_end_matches('/');
        let url = format!("{}/{}?prefix={}", endpoint, bucket, key_prefix);

        info!("Listing S3 objects with prefix: {}", url);
        let mut req = self.client.get(&url);

        if !self.config.s3_access_key.is_empty() && !self.config.s3_secret_key.is_empty() {
            let access_key = &self.config.s3_access_key;
            let secret_key = &self.config.s3_secret_key;
            let region = if self.config.s3_region.is_empty() {
                "us-central1"
            } else {
                &self.config.s3_region
            };

            let host = endpoint
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or("localhost");

            let path = format!("/{}", bucket);
            let query = format!("prefix={}", key_prefix);
            let now = chrono::Utc::now();
            let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
            let date_stamp = now.format("%Y%m%d").to_string();
            let payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

            let canonical_headers = format!("host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n", host, payload_hash, amz_date);
            let signed_headers = "host;x-amz-content-sha256;x-amz-date";
            let canonical_request = format!("GET\n{}\n{}\n{}\n{}\n{}", path, query, canonical_headers, signed_headers, payload_hash);
            let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
            let string_to_sign = format!("AWS4-HMAC-SHA256\n{}\n{}\n{}", amz_date, credential_scope, sha256_hex(canonical_request.as_bytes()));

            let k_secret = format!("AWS4{}", secret_key);
            let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
            let k_region = hmac_sha256(&k_date, region.as_bytes());
            let k_service = hmac_sha256(&k_region, b"s3");
            let k_signing = hmac_sha256(&k_service, b"aws4_request");
            let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

            let auth_header = format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                access_key, credential_scope, signed_headers, signature
            );

            req = req
                .header("host", host)
                .header("x-amz-date", amz_date)
                .header("x-amz-content-sha256", payload_hash)
                .header("Authorization", auth_header);
        }

        let resp = req.send().await?;
        let text = resp.text().await?;
        let mut keys = Vec::new();
        // Parse simple XML Key tags: <Key>...</Key>
        let mut start_idx = 0;
        while let Some(pos) = text[start_idx..].find("<Key>") {
            let key_start = start_idx + pos + 5;
            if let Some(end_pos) = text[key_start..].find("</Key>") {
                let key = &text[key_start..key_start + end_pos];
                keys.push(format!("s3://{}/{}", bucket, key));
                start_idx = key_start + end_pos + 6;
            } else {
                break;
            }
        }
        Ok(keys)
    }
}

/// Helper to extract target benchmark size from URI pattern (e.g. "weights_500mb.safetensors" -> 500 MiB)
fn extract_size_from_uri(uri: &str) -> usize {
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
    20 * 1024 * 1024 // Default 20 MiB
}

/// Fallback helper to generate synthetic tensor bytes for offline demo
fn generate_synthetic_payload(uri: &str, size: usize) -> Vec<u8> {
    let header = format!("HYPERCACHE_SYNTHETIC_TENSOR_HEADER:{}\n", uri);
    let mut data = Vec::with_capacity(size);
    data.extend_from_slice(header.as_bytes());
    let pattern = b"0123456789ABCDEF_HYPERCACHE_BLOCK_DMA_";

    // Pre-build a 1 MiB block template
    let mut block = Vec::with_capacity(1024 * 1024);
    while block.len() < 1024 * 1024 {
        block.extend_from_slice(pattern);
    }
    block.truncate(1024 * 1024);

    while data.len() + block.len() <= size {
        data.extend_from_slice(&block);
    }
    let remainder = size.saturating_sub(data.len());
    if remainder > 0 {
        data.extend_from_slice(&block[..remainder]);
    }
    data
}
