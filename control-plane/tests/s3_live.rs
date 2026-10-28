//! Live S3 integration tests against a real S3-compatible endpoint (versitygw, MinIO, AWS).
//!
//! These are `#[ignore]`d so a plain `cargo test` never needs network access. Run them with:
//!
//! ```bash
//! HYPERCACHE_TEST_S3_ENDPOINT=http://127.0.0.1:9000 \
//! HYPERCACHE_TEST_S3_ACCESS_KEY=minioadmin HYPERCACHE_TEST_S3_SECRET_KEY=minioadmin \
//! HYPERCACHE_TEST_S3_BUCKET=hypercache-e2e \
//!   cargo test --test s3_live -- --ignored --test-threads=1
//! ```
//!
//! The bucket must already exist (scripts/verify_release.sh creates it).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use hypercache_control_plane::config::Config;
use hypercache_control_plane::ingestion::s3::S3Fetcher;
use hypercache_control_plane::ingestion::IngestionEngine;
use hypercache_control_plane::metadata::MetadataStore;

fn s3_config() -> Option<Config> {
    let endpoint = std::env::var("HYPERCACHE_TEST_S3_ENDPOINT").ok()?;
    Some(Config {
        authoritative_backend: "s3".to_string(),
        s3_endpoint: endpoint,
        s3_region: "us-east-1".to_string(),
        s3_bucket: std::env::var("HYPERCACHE_TEST_S3_BUCKET").unwrap_or_else(|_| "hypercache-e2e".to_string()),
        s3_access_key: std::env::var("HYPERCACHE_TEST_S3_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".to_string()),
        s3_secret_key: std::env::var("HYPERCACHE_TEST_S3_SECRET_KEY").unwrap_or_else(|_| "minioadmin".to_string()),
        s3_session_token: None,
        allow_synthetic_s3_fallback: false,
        ..Config::default()
    })
}

macro_rules! require_s3 {
    () => {
        match s3_config() {
            Some(c) => c,
            None => {
                eprintln!("HYPERCACHE_TEST_S3_ENDPOINT not set; skipping live S3 test");
                return;
            }
        }
    };
}

fn unique(tag: &str) -> String {
    format!(
        "it-{}-{}-{}",
        tag,
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    )
}

fn pattern(len: usize, seed: u64) -> Vec<u8> {
    // xorshift so every byte position differs between objects and parts
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

#[tokio::test]
#[ignore]
async fn list_prefix_paginates_past_1000_keys() {
    let config = require_s3!();
    let bucket = config.s3_bucket.clone();
    let fetcher = S3Fetcher::new(config);
    let prefix = unique("list");
    let count = 1005usize;

    let results: Vec<anyhow::Result<()>> = futures_util::stream::iter(0..count)
        .map(|i| {
            let f = fetcher.clone();
            let uri = format!("s3://{}/{}/shard-{:05}.bin", bucket, prefix, i);
            async move { f.upload_object(&uri, format!("{}", i).as_bytes()).await }
        })
        .buffer_unordered(32)
        .collect()
        .await;
    for r in results {
        r.expect("upload failed");
    }

    let listed = fetcher
        .list_prefix(&format!("s3://{}/{}/", bucket, prefix))
        .await
        .expect("list_prefix failed");
    assert_eq!(listed.len(), count, "pagination lost keys");
    let mut uniq = listed.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), count, "duplicate keys across pages");
    assert!(listed.iter().all(|u| u.contains(&prefix)));
}

#[tokio::test]
#[ignore]
async fn multipart_upload_and_parallel_ranged_download_round_trip() {
    let config = require_s3!();
    let bucket = config.s3_bucket.clone();
    let fetcher = S3Fetcher::new(Config {
        s3_parallel_streams: 8,
        ..config
    });
    let uri = format!("s3://{}/{}/big.bin", bucket, unique("mpu"));
    let total = 70 * 1024 * 1024 + 12345;
    let data = Arc::new(pattern(total, 42));
    let chunk = 4 * 1024 * 1024u64;
    let part = S3Fetcher::multipart_part_size(total as u64, chunk);

    let src = Arc::clone(&data);
    fetcher
        .upload_object_multipart(&uri, total as u64, part, move |start, len| {
            let src = Arc::clone(&src);
            async move { Ok(src[start as usize..(start + len) as usize].to_vec()) }
        })
        .await
        .expect("multipart upload failed");

    assert_eq!(fetcher.head_object(&uri).await.unwrap(), Some(total as u64));

    let parts: Arc<Mutex<BTreeMap<usize, Vec<u8>>>> = Arc::default();
    let sink = Arc::clone(&parts);
    let fetched = fetcher
        .fetch_object_streaming(&uri, chunk as usize, false, move |idx, bytes| {
            let sink = Arc::clone(&sink);
            async move {
                assert!(
                    sink.lock().unwrap().insert(idx, bytes).is_none(),
                    "chunk {} delivered twice",
                    idx
                );
                Ok(())
            }
        })
        .await
        .expect("streaming fetch failed");
    assert_eq!(fetched, total as u64);
    let assembled: Vec<u8> = parts.lock().unwrap().values().flatten().copied().collect();
    assert!(assembled == *data, "downloaded bytes differ from uploaded bytes");
}

#[tokio::test]
#[ignore]
async fn keys_needing_uri_encoding_are_signed_correctly() {
    let config = require_s3!();
    let bucket = config.s3_bucket.clone();
    let fetcher = S3Fetcher::new(config);
    let base = unique("enc");
    for key in [
        "dir with space/file name.bin",
        "a+b=c&d/x~y_z.bin",
        "unicode/\u{00e9}t\u{00e9}.bin",
    ] {
        let uri = format!("s3://{}/{}/{}", bucket, base, key);
        let body = pattern(1000, key.len() as u64);
        fetcher
            .upload_object(&uri, &body)
            .await
            .unwrap_or_else(|e| panic!("PUT {}: {:#}", key, e));
        assert_eq!(fetcher.head_object(&uri).await.unwrap(), Some(1000), "HEAD {}", key);
    }
    let listed = fetcher
        .list_prefix(&format!("s3://{}/{}/", bucket, base))
        .await
        .unwrap();
    assert_eq!(listed.len(), 3, "listed: {:?}", listed);
}

#[tokio::test]
#[ignore]
async fn missing_object_fails_without_synthetic_data() {
    let config = require_s3!();
    let bucket = config.s3_bucket.clone();
    let fetcher = S3Fetcher::new(config);
    // Even benchmark-looking keys must not fabricate data when synthetic is not allowed
    let uri = format!("s3://{}/models/benchmark/{}_8mb.bin", bucket, unique("missing"));
    let res = fetcher
        .fetch_object_streaming(&uri, 4 * 1024 * 1024, false, |_, _| async { Ok(()) })
        .await;
    assert!(res.is_err(), "missing object must be an error");
}

#[tokio::test]
#[ignore]
async fn engine_ingests_and_flushes_through_s3() {
    let config = require_s3!();
    let dir = std::env::temp_dir().join(unique("engine"));
    std::fs::create_dir_all(&dir).unwrap();
    let devs: Vec<String> = (0..2)
        .map(|i| {
            let p = dir.join(format!("dev{}.raw", i));
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(128 * 1024 * 1024).unwrap();
            p.to_string_lossy().to_string()
        })
        .collect();
    let config = Config {
        block_devices: devs,
        chunk_size_bytes: 4 * 1024 * 1024,
        device_start_offset_bytes: 4096,
        auto_create_mock_devices: false,
        s3_multipart_threshold_bytes: 8 * 1024 * 1024,
        ..config
    };
    let bucket = config.s3_bucket.clone();
    let metadata = MetadataStore::new(1 << 34, None, None);
    let engine = IngestionEngine::new(config.clone(), metadata.clone(), None);
    let fetcher = S3Fetcher::new(config);

    let src_uri = format!("s3://{}/{}/model.bin", bucket, unique("src"));
    let body = pattern(21 * 1024 * 1024 + 7, 7);
    fetcher.upload_object(&src_uri, &body).await.unwrap();

    let obj = engine.ingest_object(&src_uri).await.expect("ingest failed");
    assert_eq!(obj.object_size, body.len() as u64);
    assert!(obj.layout.iter().all(|c| c.checksum.is_some()));
    let (digest, _) = engine.digest_layout(&obj.layout).await.unwrap();
    use sha2::Digest;
    assert_eq!(digest, hex::encode(sha2::Sha256::digest(&body)));

    // Flush the cached layout to a new key: above the threshold, so multipart is used
    let dst_uri = format!("s3://{}/{}/copy.bin", bucket, unique("dst"));
    engine
        .flush_to_backend(&dst_uri, &obj.layout)
        .await
        .expect("flush failed");
    assert_eq!(fetcher.head_object(&dst_uri).await.unwrap(), Some(body.len() as u64));
    let again = engine.ingest_object(&dst_uri).await.unwrap();
    let (digest2, _) = engine.digest_layout(&again.layout).await.unwrap();
    assert_eq!(digest2, digest);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
#[ignore]
async fn server_enforces_signatures() {
    // Guards the other tests: if the endpoint accepted unsigned or wrongly signed requests,
    // the URI-encoding / SigV4 tests would pass vacuously.
    let config = require_s3!();
    let bucket = config.s3_bucket.clone();
    let bad = S3Fetcher::new(Config {
        s3_secret_key: format!("{}-wrong", config.s3_secret_key),
        ..config
    });
    let uri = format!("s3://{}/{}/x.bin", bucket, unique("sig"));
    assert!(
        bad.upload_object(&uri, b"x").await.is_err(),
        "endpoint accepted a bad signature"
    );
}
