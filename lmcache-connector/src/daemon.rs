//! Minimal client for the daemon's JSON-lines socket. One per worker thread, so a slow commit on
//! one connection can't hold up the others.

use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
pub struct Chunk {
    #[serde(default = "default_tier")]
    pub tier: String,
    pub device: String,
    pub offset: u64,
    pub length: u64,
    #[serde(default)]
    pub checksum: Option<u32>,
}

fn default_tier() -> String {
    "nvme".to_string()
}

impl Chunk {
    pub fn is_ram(&self) -> bool {
        self.tier.eq_ignore_ascii_case("ram")
    }

    pub fn is_shm(&self) -> bool {
        self.is_ram() || self.device.ends_with(".shm")
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct Response {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub object_size: Option<u64>,
    #[serde(default)]
    pub chunk_layout: Option<Vec<Chunk>>,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub sync_status: Option<String>,
    #[serde(default)]
    pub error_message: Option<String>,
}

impl Response {
    pub fn layout(&self) -> &[Chunk] {
        self.chunk_layout.as_deref().unwrap_or(&[])
    }

    /// True only for a committed local-only block with a checksum on every chunk. Anything else
    /// could be stale, unverifiable or on its way to a backend, so it counts as a miss.
    pub fn is_valid_block(&self) -> bool {
        self.status == "HIT"
            && self.sync_status.as_deref() == Some("LOCAL_ONLY")
            && !self.layout().is_empty()
            && self.layout().iter().all(|chunk| chunk.checksum.is_some())
    }

    fn message(&self) -> &str {
        self.error_message.as_deref().unwrap_or("unknown error")
    }
}

pub struct DaemonClient {
    socket_path: PathBuf,
    timeout: Duration,
    commit_timeout: Duration,
    stream: Option<BufReader<UnixStream>>,
    line: String,
}

impl DaemonClient {
    pub fn new(socket_path: PathBuf, timeout: Duration) -> Self {
        // The daemon syncs the device during a commit, which can take longer than a lookup
        let commit_timeout = timeout.max(Duration::from_secs(300));
        DaemonClient {
            socket_path,
            timeout,
            commit_timeout,
            stream: None,
            line: String::new(),
        }
    }

    fn request(&mut self, body: &Value, timeout: Duration) -> io::Result<Response> {
        let result = self.exchange(body, timeout);
        if result.is_err() {
            // A timeout or reset leaves the stream at an unknown position: start clean next time
            self.stream = None;
        }
        result
    }

    fn exchange(&mut self, body: &Value, timeout: Duration) -> io::Result<Response> {
        if self.stream.is_none() {
            self.stream = Some(BufReader::new(UnixStream::connect(&self.socket_path)?));
        }
        let reader = self.stream.as_mut().expect("connected above");
        reader.get_ref().set_read_timeout(Some(timeout))?;
        reader.get_ref().set_write_timeout(Some(timeout))?;

        let mut message = serde_json::to_vec(body)?;
        message.push(b'\n');
        reader.get_mut().write_all(&message)?;

        loop {
            self.line.clear();
            if reader.read_line(&mut self.line)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "daemon closed the connection",
                ));
            }
            if !self.line.trim().is_empty() {
                return Ok(serde_json::from_str(&self.line)?);
            }
        }
    }

    pub fn locate(&mut self, uri: &str) -> io::Result<Response> {
        let body = json!({"action": "locate", "uri": uri, "auto_ingest": false, "pinned": false});
        self.request(&body, self.timeout)
    }

    pub fn stage_write(&mut self, uri: &str, size: u64, tier: &str) -> io::Result<Response> {
        let body = json!({
            "action": "stage_write",
            "uri": uri,
            "size_bytes": size,
            "tier": tier,
            "is_leader": false,
            "auto_flush_s3": false,
            "write_through": false,
            "pinned": false,
        });
        let response = self.request(&body, self.timeout)?;
        if response.status != "WRITESTAGED" && response.status != "WRITE_STAGED" {
            return Err(io::Error::other(format!("stage_write failed: {}", response.message())));
        }
        Ok(response)
    }

    pub fn commit_local_only(&mut self, uri: &str, checksums: &[u32]) -> io::Result<()> {
        let body = json!({
            "action": "commit_write",
            "uri": uri,
            "write_through": false,
            "local_only": true,
            "chunk_checksums": checksums,
        });
        let response = self.request(&body, self.commit_timeout)?;
        if response.status != "WRITECOMMITTED" && response.status != "WRITE_COMMITTED" {
            return Err(io::Error::other(format!("commit_write failed: {}", response.message())));
        }
        Ok(())
    }

    pub fn evict(&mut self, uri: &str) -> io::Result<bool> {
        let response = self.request(&json!({"action": "evict", "uri": uri}), self.timeout)?;
        Ok(matches!(response.status.as_str(), "EVICTED" | "PONG" | "OK"))
    }
}
