use anyhow::{Context, Result};
use futures_util::{StreamExt, TryStreamExt};
use hmac::{Hmac, Mac};
use reqwest::{Client, Method, RequestBuilder, Url};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tracing::{info, warn};

use super::synthetic::{extract_size_from_uri, generate_synthetic_payload};
use crate::config::Config;

type HmacSha256 = Hmac<Sha256>;

const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";
const MAX_RANGE_ATTEMPTS: u32 = 4;
/// S3 rejects multipart uploads with more than 10,000 parts and parts below 5 MiB (except the last).
const MAX_MULTIPART_PARTS: u64 = 10_000;
const MIN_PART_SIZE: u64 = 8 * 1024 * 1024;

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

/// AWS SigV4 URI encoding: unreserved characters pass through, everything else is %XX.
/// `/` is kept only when encoding a path.
pub fn uri_encode(input: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(input.len() * 3 / 2);
    for b in input.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b'/' if keep_slash => out.push('/'),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Returns the (unescaped) text content of every `<tag>...</tag>` occurrence.
fn xml_tag_values(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let mut values = Vec::new();
    let mut pos = 0;
    while let Some(start) = text[pos..].find(&open) {
        let value_start = pos + start + open.len();
        match text[value_start..].find(&close) {
            Some(end) => {
                values.push(xml_unescape(&text[value_start..value_start + end]));
                pos = value_start + end + close.len();
            }
            None => break,
        }
    }
    values
}

#[derive(Clone)]
pub struct S3Fetcher {
    client: Client,
    config: Config,
}

impl S3Fetcher {
    pub fn new(config: Config) -> Self {
        // No whole-request timeout: multi-GB bodies legitimately take longer than any fixed
        // bound. A stalled connection is caught by the per-read idle timeout instead.
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(config.s3_read_timeout_secs.max(1)))
            .tcp_keepalive(Duration::from_secs(30))
            .pool_max_idle_per_host(64)
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

    fn region(&self) -> &str {
        if self.config.s3_region.is_empty() {
            "us-east-1"
        } else {
            &self.config.s3_region
        }
    }

    /// Builds the request URL and its SigV4 canonical path (path-style addressing).
    /// Any path prefix on the configured endpoint is preserved.
    fn object_url(&self, bucket: &str, key: Option<&str>) -> Result<(Url, String)> {
        let endpoint = Url::parse(self.config.s3_endpoint.trim_end_matches('/'))
            .with_context(|| format!("Invalid s3_endpoint '{}'", self.config.s3_endpoint))?;
        let base_path = endpoint.path().trim_end_matches('/');
        let mut canonical_path = format!("{}/{}", base_path, uri_encode(bucket, false));
        if let Some(k) = key {
            canonical_path.push('/');
            canonical_path.push_str(&uri_encode(k, true));
        }
        let mut url = endpoint.clone();
        url.set_path(&canonical_path);
        url.set_query(None);
        Ok((url, canonical_path))
    }

    /// Creates a SigV4-signed request (signing is skipped when no credentials are configured,
    /// for public buckets and local S3-compatible stores).
    fn signed_request(
        &self,
        method: Method,
        url: &Url,
        canonical_path: &str,
        query: &[(String, String)],
        payload_hash: &str,
    ) -> RequestBuilder {
        let mut sorted_query: Vec<(String, String)> = query
            .iter()
            .map(|(k, v)| (uri_encode(k, false), uri_encode(v, false)))
            .collect();
        sorted_query.sort();
        let canonical_query = sorted_query
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");

        let mut full_url = url.clone();
        full_url.set_query(if canonical_query.is_empty() {
            None
        } else {
            Some(&canonical_query)
        });

        let mut req = self.client.request(method.clone(), full_url);

        if self.config.s3_access_key.is_empty() || self.config.s3_secret_key.is_empty() {
            return req;
        }

        let host = match (url.host_str(), url.port()) {
            (Some(h), Some(p)) => format!("{}:{}", h, p),
            (Some(h), None) => h.to_string(),
            _ => "localhost".to_string(),
        };
        let now = chrono::Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();
        let session_token = self.config.s3_session_token.as_deref().filter(|t| !t.trim().is_empty());

        let mut canonical_headers = format!(
            "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
            host, payload_hash, amz_date
        );
        let mut signed_headers = "host;x-amz-content-sha256;x-amz-date".to_string();
        if let Some(token) = session_token {
            canonical_headers.push_str(&format!("x-amz-security-token:{}\n", token.trim()));
            signed_headers.push_str(";x-amz-security-token");
        }

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method.as_str(),
            canonical_path,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash
        );
        let region = self.region();
        let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            amz_date,
            credential_scope,
            sha256_hex(canonical_request.as_bytes())
        );

        let k_secret = format!("AWS4{}", self.config.s3_secret_key);
        let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
        let k_region = hmac_sha256(&k_date, region.as_bytes());
        let k_service = hmac_sha256(&k_region, b"s3");
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

        let auth_header = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.config.s3_access_key, credential_scope, signed_headers, signature
        );

        req = req
            .header("x-amz-date", amz_date)
            .header("x-amz-content-sha256", payload_hash)
            .header("Authorization", auth_header);
        if let Some(token) = session_token {
            req = req.header("x-amz-security-token", token.trim());
        }
        req
    }

    /// Returns the object size via HEAD, or None if HEAD failed or reported no length.
    pub async fn head_object(&self, uri: &str) -> Result<Option<u64>> {
        let (bucket, key) = self.parse_uri(uri);
        let (url, path) = self.object_url(&bucket, Some(&key))?;
        let resp = self
            .signed_request(Method::HEAD, &url, &path, &[], UNSIGNED_PAYLOAD)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        Ok(resp
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok()))
    }

    /// Fetches bytes [start, start+len) with retries and strict length validation.
    async fn fetch_range(&self, url: &Url, path: &str, start: u64, len: u64) -> Result<Vec<u8>> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let result: Result<Vec<u8>> = async {
                let resp = self
                    .signed_request(Method::GET, url, path, &[], UNSIGNED_PAYLOAD)
                    .header(reqwest::header::RANGE, format!("bytes={}-{}", start, start + len - 1))
                    .send()
                    .await?;
                let status = resp.status();
                if status != reqwest::StatusCode::PARTIAL_CONTENT && !(status == reqwest::StatusCode::OK && start == 0)
                {
                    anyhow::bail!("ranged GET returned HTTP {}", status);
                }
                let mut bytes = resp.bytes().await?.to_vec();
                if status == reqwest::StatusCode::OK {
                    // Server ignored the Range header and sent the whole object
                    bytes.truncate(len as usize);
                }
                if bytes.len() as u64 != len {
                    anyhow::bail!("ranged GET returned {} bytes, expected {}", bytes.len(), len);
                }
                Ok(bytes)
            }
            .await;

            match result {
                Ok(b) => return Ok(b),
                Err(e) if attempt < MAX_RANGE_ATTEMPTS => {
                    let backoff = Duration::from_millis(200 * (1 << (attempt - 1)));
                    warn!(
                        "S3 range {}-{} of {} failed (attempt {}/{}): {}. Retrying in {:?}",
                        start,
                        start + len - 1,
                        url,
                        attempt,
                        MAX_RANGE_ATTEMPTS,
                        e,
                        backoff
                    );
                    tokio::time::sleep(backoff).await;
                }
                Err(e) => {
                    return Err(e.context(format!(
                        "S3 range {}-{} of {} failed after {} attempts",
                        start,
                        start + len - 1,
                        url,
                        MAX_RANGE_ATTEMPTS
                    )))
                }
            }
        }
    }

    /// Fetches the object bytes from authoritative S3 (buffers the whole object; tests and small objects only)
    #[allow(dead_code)]
    pub async fn fetch_object(&self, uri: &str) -> Result<Vec<u8>> {
        self.fetch_object_with_fallback(uri, self.config.allow_synthetic_s3_fallback)
            .await
    }

    #[allow(dead_code)]
    pub async fn fetch_object_with_fallback(&self, uri: &str, allow_synthetic: bool) -> Result<Vec<u8>> {
        let collected = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<(usize, Vec<u8>)>::new()));
        let sink = std::sync::Arc::clone(&collected);
        self.fetch_object_streaming(uri, self.config.chunk_size_bytes, allow_synthetic, move |idx, data| {
            let sink = std::sync::Arc::clone(&sink);
            async move {
                sink.lock().await.push((idx, data));
                Ok(())
            }
        })
        .await?;
        let mut parts = std::sync::Arc::try_unwrap(collected)
            .map_err(|_| anyhow::anyhow!("chunk sink still shared"))?
            .into_inner();
        parts.sort_by_key(|(i, _)| *i);
        Ok(parts.into_iter().flat_map(|(_, d)| d).collect())
    }

    /// Streams an S3 object in fixed-size chunks (`chunk_size_bytes`) without buffering the full object.
    ///
    /// When the object size is known (HEAD), chunks are fetched with up to `s3_parallel_streams`
    /// concurrent ranged GETs and `on_chunk` runs concurrently for different chunk indices.
    /// Otherwise a single streaming GET is used. `on_chunk` may be called out of order.
    pub async fn fetch_object_streaming<F, Fut>(
        &self,
        uri: &str,
        chunk_size: usize,
        allow_synthetic: bool,
        on_chunk: F,
    ) -> Result<u64>
    where
        F: Fn(usize, Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let (bucket, key) = self.parse_uri(uri);
        let (url, path) = self.object_url(&bucket, Some(&key))?;
        let parallel = self.config.s3_parallel_streams.max(1);

        if parallel > 1 {
            if let Ok(Some(size)) = self.head_object(uri).await {
                let chunk = chunk_size as u64;
                let num_chunks = size.div_ceil(chunk);
                info!(
                    "Streaming authoritative object from S3: {} ({} bytes, {} ranged GETs, {} parallel streams)",
                    url, size, num_chunks, parallel
                );
                let on_chunk = &on_chunk;
                let url_ref = &url;
                let path_ref = path.as_str();
                futures_util::stream::iter(0..num_chunks)
                    .map(|idx| async move {
                        let start = idx * chunk;
                        let len = chunk.min(size - start);
                        let data = self.fetch_range(url_ref, path_ref, start, len).await?;
                        on_chunk(idx as usize, data).await
                    })
                    .buffer_unordered(parallel)
                    .try_collect::<Vec<()>>()
                    .await?;
                info!("Successfully streamed {} bytes from S3 for {}", size, uri);
                return Ok(size);
            }
        }

        info!("Streaming authoritative object from S3: {} (key: {})", url, key);
        let req = self.signed_request(Method::GET, &url, &path, &[], UNSIGNED_PAYLOAD);

        let failure = match req.send().await {
            Ok(r) if r.status().is_success() => {
                let mut stream = r.bytes_stream();
                let mut buf = Vec::with_capacity(chunk_size);
                let mut chunk_idx = 0usize;
                let mut total_bytes = 0u64;

                while let Some(item) = stream.next().await {
                    let bytes = item.with_context(|| "Failed reading S3 stream chunk")?;
                    let mut slice = &bytes[..];
                    while !slice.is_empty() {
                        let take = (chunk_size - buf.len()).min(slice.len());
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
                return Ok(total_bytes);
            }
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                format!(
                    "S3 GET failed for '{}' ({}) with HTTP status {}: {}",
                    uri, url, status, body
                )
            }
            Err(e) => format!(
                "Failed to connect to authoritative S3 endpoint '{}' for URI '{}': {}",
                url, uri, e
            ),
        };

        if !allow_synthetic {
            anyhow::bail!(failure);
        }

        let target_size = extract_size_from_uri(uri);
        warn!(
            "{} (synthetic fallback active). Streaming {:.2} MiB synthetic tensor payload for {}",
            failure,
            target_size as f64 / (1024.0 * 1024.0),
            uri
        );
        let data = generate_synthetic_payload(uri, target_size);
        for (chunk_idx, slice) in data.chunks(chunk_size).enumerate() {
            on_chunk(chunk_idx, slice.to_vec()).await?;
        }
        Ok(data.len() as u64)
    }

    /// Uploads object bytes to authoritative S3 via a single SigV4 PUT
    pub async fn upload_object(&self, uri: &str, data: &[u8]) -> Result<()> {
        let (bucket, key) = self.parse_uri(uri);
        let (url, path) = self.object_url(&bucket, Some(&key))?;

        info!("Uploading {} bytes to S3: {} (key: {})", data.len(), url, key);

        let payload_hash = sha256_hex(data);
        let resp = self
            .signed_request(Method::PUT, &url, &path, &[], &payload_hash)
            .body(data.to_vec())
            .send()
            .await
            .with_context(|| format!("Failed to send S3 PUT to {}", url))?;
        if resp.status().is_success() {
            info!("Successfully uploaded {} bytes to S3: {}", data.len(), url);
            Ok(())
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("S3 PUT failed with status {}: {}", status, text);
        }
    }

    /// Chooses a multipart part size: a multiple of `chunk_size`, at least 8 MiB, and small
    /// enough to stay within the 10,000-part limit.
    pub fn multipart_part_size(total_size: u64, chunk_size: u64) -> u64 {
        let min_for_limit = total_size.div_ceil(MAX_MULTIPART_PARTS);
        let target = min_for_limit.max(MIN_PART_SIZE);
        target.div_ceil(chunk_size.max(1)) * chunk_size.max(1)
    }

    /// Streams an object to S3 with a multipart upload. `read_part(start, len)` supplies the
    /// bytes of each part, so the full object is never held in memory. The upload is
    /// aborted on failure so no orphaned parts are billed.
    pub async fn upload_object_multipart<F, Fut>(
        &self,
        uri: &str,
        total_size: u64,
        part_size: u64,
        read_part: F,
    ) -> Result<()>
    where
        F: Fn(u64, u64) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<u8>>>,
    {
        let (bucket, key) = self.parse_uri(uri);
        let (url, path) = self.object_url(&bucket, Some(&key))?;

        let init_query = vec![("uploads".to_string(), String::new())];
        let resp = self
            .signed_request(Method::POST, &url, &path, &init_query, UNSIGNED_PAYLOAD)
            .send()
            .await
            .with_context(|| format!("Failed to initiate multipart upload to {}", url))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("S3 CreateMultipartUpload failed with status {}: {}", status, text);
        }
        let upload_id = xml_tag_values(&text, "UploadId")
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("CreateMultipartUpload response had no UploadId"))?;

        let num_parts = total_size.div_ceil(part_size);
        info!(
            "Multipart upload of {} bytes to {} in {} parts of {} bytes (upload id {})",
            total_size, url, num_parts, part_size, upload_id
        );

        let parallel = self.config.s3_parallel_streams.clamp(1, 16);
        let read_part = &read_part;
        let (url_ref, path_ref, upload_id_ref) = (&url, path.as_str(), upload_id.as_str());
        let upload_result: Result<Vec<(u64, String)>> = futures_util::stream::iter(0..num_parts)
            .map(|part_idx| async move {
                let start = part_idx * part_size;
                let len = part_size.min(total_size - start);
                let data = read_part(start, len).await?;
                let part_number = part_idx + 1;
                let query = vec![
                    ("partNumber".to_string(), part_number.to_string()),
                    ("uploadId".to_string(), upload_id_ref.to_string()),
                ];
                let mut attempt = 0;
                loop {
                    attempt += 1;
                    let res = self
                        .signed_request(Method::PUT, url_ref, path_ref, &query, UNSIGNED_PAYLOAD)
                        .body(data.clone())
                        .send()
                        .await;
                    let err = match res {
                        Ok(r) if r.status().is_success() => {
                            let etag = r
                                .headers()
                                .get(reqwest::header::ETAG)
                                .and_then(|v| v.to_str().ok())
                                .map(|s| s.to_string())
                                .ok_or_else(|| anyhow::anyhow!("UploadPart {} returned no ETag", part_number))?;
                            return Ok((part_number, etag));
                        }
                        Ok(r) => format!("HTTP {}: {}", r.status(), r.text().await.unwrap_or_default()),
                        Err(e) => e.to_string(),
                    };
                    if attempt >= MAX_RANGE_ATTEMPTS {
                        anyhow::bail!("UploadPart {} failed after {} attempts: {}", part_number, attempt, err);
                    }
                    tokio::time::sleep(Duration::from_millis(200 * (1 << (attempt - 1)))).await;
                }
            })
            .buffer_unordered(parallel)
            .try_collect()
            .await;

        let mut parts = match upload_result {
            Ok(p) => p,
            Err(e) => {
                self.abort_multipart(&url, &path, &upload_id).await;
                return Err(e);
            }
        };
        parts.sort_by_key(|(n, _)| *n);

        let mut body = String::from("<CompleteMultipartUpload>");
        for (n, etag) in &parts {
            body.push_str(&format!(
                "<Part><PartNumber>{}</PartNumber><ETag>{}</ETag></Part>",
                n,
                xml_escape(etag)
            ));
        }
        body.push_str("</CompleteMultipartUpload>");

        let complete_query = vec![("uploadId".to_string(), upload_id.clone())];
        let payload_hash = sha256_hex(body.as_bytes());
        let resp = self
            .signed_request(Method::POST, &url, &path, &complete_query, &payload_hash)
            .body(body)
            .send()
            .await;
        match resp {
            // CompleteMultipartUpload can return 200 with an <Error> body
            Ok(r) if r.status().is_success() => {
                let text = r.text().await.unwrap_or_default();
                if text.contains("<Error>") {
                    self.abort_multipart(&url, &path, &upload_id).await;
                    anyhow::bail!("S3 CompleteMultipartUpload failed: {}", text);
                }
                info!("Completed multipart upload of {} bytes to {}", total_size, url);
                Ok(())
            }
            Ok(r) => {
                let status = r.status();
                let text = r.text().await.unwrap_or_default();
                self.abort_multipart(&url, &path, &upload_id).await;
                anyhow::bail!("S3 CompleteMultipartUpload failed with status {}: {}", status, text)
            }
            Err(e) => {
                self.abort_multipart(&url, &path, &upload_id).await;
                Err(anyhow::Error::from(e).context("Failed to send CompleteMultipartUpload"))
            }
        }
    }

    async fn abort_multipart(&self, url: &Url, path: &str, upload_id: &str) {
        let query = vec![("uploadId".to_string(), upload_id.to_string())];
        match self
            .signed_request(Method::DELETE, url, path, &query, UNSIGNED_PAYLOAD)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                warn!("Aborted multipart upload {} to {}", upload_id, url)
            }
            Ok(r) => warn!("Failed to abort multipart upload {}: HTTP {}", upload_id, r.status()),
            Err(e) => warn!("Failed to abort multipart upload {}: {}", upload_id, e),
        }
    }

    /// Lists all objects under a prefix (ListObjectsV2, following continuation tokens)
    pub async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let (bucket, key_prefix) = self.parse_uri(prefix);
        let (url, path) = self.object_url(&bucket, None)?;

        info!("Listing S3 objects in bucket '{}' with prefix '{}'", bucket, key_prefix);
        let mut keys = Vec::new();
        let mut continuation: Option<String> = None;

        loop {
            let mut query = vec![
                ("list-type".to_string(), "2".to_string()),
                ("prefix".to_string(), key_prefix.clone()),
            ];
            if let Some(ref token) = continuation {
                query.push(("continuation-token".to_string(), token.clone()));
            }

            let resp = self
                .signed_request(Method::GET, &url, &path, &query, UNSIGNED_PAYLOAD)
                .send()
                .await
                .with_context(|| format!("Failed to list S3 prefix at {}", url))?;
            let status = resp.status();
            let text = resp.text().await?;
            if !status.is_success() {
                anyhow::bail!("S3 ListObjectsV2 failed with status {}: {}", status, text);
            }

            for key in xml_tag_values(&text, "Key") {
                keys.push(format!("s3://{}/{}", bucket, key));
            }

            let truncated = xml_tag_values(&text, "IsTruncated")
                .first()
                .is_some_and(|v| v.trim() == "true");
            continuation = xml_tag_values(&text, "NextContinuationToken").into_iter().next();
            if !truncated || continuation.is_none() {
                break;
            }
        }

        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_encode_follows_sigv4_rules() {
        assert_eq!(uri_encode("models/llama 3/w+1.bin", true), "models/llama%203/w%2B1.bin");
        assert_eq!(uri_encode("a/b", false), "a%2Fb");
        assert_eq!(uri_encode("A-z_0.9~", false), "A-z_0.9~");
        assert_eq!(uri_encode("\u{e9}", false), "%C3%A9");
    }

    #[test]
    fn test_xml_helpers() {
        let xml = "<ListBucketResult><IsTruncated>true</IsTruncated><Contents><Key>a&amp;b.bin</Key></Contents>\
                   <Contents><Key>c&lt;d</Key></Contents><NextContinuationToken>tok/+=</NextContinuationToken></ListBucketResult>";
        assert_eq!(xml_tag_values(xml, "Key"), vec!["a&b.bin", "c<d"]);
        assert_eq!(xml_tag_values(xml, "IsTruncated"), vec!["true"]);
        assert_eq!(xml_tag_values(xml, "NextContinuationToken"), vec!["tok/+="]);
        assert_eq!(xml_escape("\"etag&\""), "&quot;etag&amp;&quot;");
    }

    #[test]
    fn test_multipart_part_size_respects_limits() {
        let chunk = 4 * 1024 * 1024;
        // Small objects use the 8 MiB minimum (chunk aligned)
        assert_eq!(
            S3Fetcher::multipart_part_size(100 * 1024 * 1024, chunk),
            8 * 1024 * 1024
        );
        // 1 TiB object must fit in 10,000 parts
        let tib = 1024u64 * 1024 * 1024 * 1024;
        let part = S3Fetcher::multipart_part_size(tib, chunk);
        assert!(tib.div_ceil(part) <= MAX_MULTIPART_PARTS);
        assert_eq!(part % chunk, 0);
    }

    #[test]
    fn test_object_url_preserves_endpoint_prefix_and_encodes_key() {
        let config = Config {
            s3_endpoint: "https://storage.example.com/s3-gw/".to_string(),
            s3_bucket: "bkt".to_string(),
            ..Config::default()
        };
        let f = S3Fetcher::new(config);
        let (url, path) = f.object_url("bkt", Some("dir/my file.bin")).unwrap();
        assert_eq!(path, "/s3-gw/bkt/dir/my%20file.bin");
        assert_eq!(url.as_str(), "https://storage.example.com/s3-gw/bkt/dir/my%20file.bin");
    }

    /// Known-answer test for the SigV4 signer using the AWS documentation example
    /// (GET /test.txt on examplebucket with a Range header excluded from signing).
    #[test]
    fn test_sigv4_signing_key_derivation() {
        let k_secret = "AWS4wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        let k_date = hmac_sha256(k_secret.as_bytes(), b"20130524");
        let k_region = hmac_sha256(&k_date, b"us-east-1");
        let k_service = hmac_sha256(&k_region, b"s3");
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        let canonical_request = "GET\n/test.txt\n\nhost:examplebucket.s3.amazonaws.com\nrange:bytes=0-9\n\
x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n\
x-amz-date:20130524T000000Z\n\nhost;range;x-amz-content-sha256;x-amz-date\n\
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n20130524T000000Z\n20130524/us-east-1/s3/aws4_request\n{}",
            sha256_hex(canonical_request.as_bytes())
        );
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));
        assert_eq!(
            signature,
            "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
    }
}
