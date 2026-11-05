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

fn connection_is_dead(error: &io::Error) -> bool {
    matches!(error.kind(), io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset)
}

/// Limits on one write of lookups. The daemon answers each line before it reads the next, so these
/// requests have to fit in the socket buffer, or both sides could block on a full buffer.
const LOOKUP_BATCH_KEYS: usize = 64;
const LOOKUP_BATCH_BYTES: usize = 64 * 1024;

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
        let mut message = serde_json::to_vec(body)?;
        message.push(b'\n');
        self.send_reconnecting(&message, timeout)?;
        self.read_response()
    }

    /// A kept connection dies when the daemon restarts, and the next write is what finds out. The
    /// daemon can't have received that request, so resending it is safe. A failure after this
    /// point is never resent, because the daemon may already have acted on the request.
    fn send_reconnecting(&mut self, message: &[u8], timeout: Duration) -> io::Result<()> {
        let reused = self.stream.is_some();
        match self.send(message, timeout) {
            Err(error) if reused && connection_is_dead(&error) => {
                self.stream = None;
                self.send(message, timeout)
            }
            other => other,
        }
    }

    fn read_response(&mut self) -> io::Result<Response> {
        let reader = self.stream.as_mut().expect("connected by send");
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

    fn send(&mut self, message: &[u8], timeout: Duration) -> io::Result<()> {
        if self.stream.is_none() {
            self.stream = Some(BufReader::new(UnixStream::connect(&self.socket_path)?));
        }
        let reader = self.stream.as_mut().expect("connected above");
        // macOS refuses these on a socket whose peer has gone, before anything is written
        let dead = |error| io::Error::new(io::ErrorKind::BrokenPipe, error);
        reader.get_ref().set_read_timeout(Some(timeout)).map_err(dead)?;
        reader.get_ref().set_write_timeout(Some(timeout)).map_err(dead)?;
        reader.get_mut().write_all(message)
    }

    pub fn locate(&mut self, uri: &str) -> io::Result<Response> {
        let body = json!({"action": "locate", "uri": uri, "auto_ingest": false, "pinned": false});
        self.request(&body, self.timeout)
    }

    /// Looks up many objects, one reply per URI in the same order. The requests go out together,
    /// which avoids most of the cost of a round trip per key. The daemon answers in order on one
    /// connection, so a reply can't be matched to the wrong URI. Any failure fails the whole call.
    pub fn locate_many(&mut self, uris: &[String]) -> io::Result<Vec<Response>> {
        let mut responses = Vec::with_capacity(uris.len());
        let mut rest = uris;
        while !rest.is_empty() {
            let (message, count) = lookup_message(rest)?;
            let result = self.locate_group(&message, count, &mut responses);
            if result.is_err() {
                // A timeout or reset leaves the stream at an unknown position: start clean next time
                self.stream = None;
                return result.map(|_| Vec::new());
            }
            rest = &rest[count..];
        }
        Ok(responses)
    }

    fn locate_group(&mut self, message: &[u8], count: usize, responses: &mut Vec<Response>) -> io::Result<()> {
        self.send_reconnecting(message, self.timeout)?;
        for _ in 0..count {
            responses.push(self.read_response()?);
        }
        Ok(())
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

/// Builds as many locate requests as fit the limits, always at least one.
fn lookup_message(uris: &[String]) -> io::Result<(Vec<u8>, usize)> {
    let mut message = Vec::new();
    let mut count = 0;
    for uri in uris.iter().take(LOOKUP_BATCH_KEYS) {
        let before = message.len();
        let body = json!({"action": "locate", "uri": uri, "auto_ingest": false, "pinned": false});
        serde_json::to_writer(&mut message, &body)?;
        message.push(b'\n');
        if count > 0 && message.len() > LOOKUP_BATCH_BYTES {
            message.truncate(before);
            break;
        }
        count += 1;
    }
    Ok((message, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const TIMEOUT: Duration = Duration::from_secs(5);

    fn socket_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hc-client-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("d.sock")
    }

    /// Answers one request on one connection, then closes it.
    fn answer_one(listener: &UnixListener, requests: &AtomicUsize) {
        // Gives up rather than blocking, so a failed assertion can't hang the test run
        listener.set_nonblocking(true).unwrap();
        let deadline = std::time::Instant::now() + TIMEOUT;
        let stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(_) if std::time::Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                Err(error) => panic!("no connection: {error}"),
            }
        };
        stream.set_nonblocking(false).unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        requests.fetch_add(1, Ordering::SeqCst);
        reader.get_mut().write_all(b"{\"status\":\"MISS\"}\n").unwrap();
    }

    #[test]
    fn request_survives_a_daemon_restart_on_a_kept_connection() {
        let socket = socket_path("restart");
        let listener = UnixListener::bind(&socket).unwrap();
        let requests = AtomicUsize::new(0);
        let mut client = DaemonClient::new(socket.clone(), TIMEOUT);

        std::thread::scope(|scope| {
            scope.spawn(|| answer_one(&listener, &requests));
            assert_eq!(client.locate("hypercache://a").unwrap().status, "MISS");
        });
        // The old connection is closed now. The next request should reconnect and go through once.
        std::thread::scope(|scope| {
            scope.spawn(|| answer_one(&listener, &requests));
            assert_eq!(client.locate("hypercache://a").unwrap().status, "MISS");
        });
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(socket.parent().unwrap());
    }

    #[test]
    fn request_the_daemon_received_is_never_resent() {
        let socket = socket_path("received");
        let listener = UnixListener::bind(&socket).unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut client = DaemonClient::new(socket.clone(), TIMEOUT);

        // A kept connection whose daemon reads the request, then dies before it replies
        let (near, far) = UnixStream::pair().unwrap();
        client.stream = Some(BufReader::new(near));
        let daemon = std::thread::spawn(move || {
            let mut line = String::new();
            BufReader::new(far).read_line(&mut line).unwrap();
        });

        assert!(client.request(&json!({"action": "commit"}), TIMEOUT).is_err());
        daemon.join().unwrap();
        // A resend would have had to reconnect through this listener
        assert_eq!(listener.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock);
        let _ = std::fs::remove_dir_all(socket.parent().unwrap());
    }

    #[test]
    fn a_daemon_that_is_really_down_is_still_an_error() {
        let socket = socket_path("down");
        let listener = UnixListener::bind(&socket).unwrap();
        let requests = AtomicUsize::new(0);
        let mut client = DaemonClient::new(socket.clone(), TIMEOUT);
        std::thread::scope(|scope| {
            scope.spawn(|| answer_one(&listener, &requests));
            client.locate("hypercache://a").unwrap();
        });
        drop(listener);
        std::fs::remove_file(&socket).unwrap();
        assert!(client.locate("hypercache://a").is_err());
        let _ = std::fs::remove_dir_all(socket.parent().unwrap());
    }

    fn uris(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("hypercache://object/{i}")).collect()
    }

    fn lines_in(message: &[u8]) -> usize {
        message.iter().filter(|b| **b == b'\n').count()
    }

    #[test]
    fn lookup_message_respects_its_limits() {
        let (message, count) = lookup_message(&uris(LOOKUP_BATCH_KEYS + 5)).unwrap();
        assert_eq!(count, LOOKUP_BATCH_KEYS);
        assert_eq!(lines_in(&message), LOOKUP_BATCH_KEYS);

        let long: Vec<String> = (0..10)
            .map(|i| format!("hypercache://{i}/{}", "x".repeat(20_000)))
            .collect();
        let (message, count) = lookup_message(&long).unwrap();
        assert_eq!(count, 3);
        assert_eq!(lines_in(&message), 3);
        assert!(message.len() <= LOOKUP_BATCH_BYTES);

        // One URI over the limit still goes out, on its own
        let huge = vec!["x".repeat(LOOKUP_BATCH_BYTES * 2), "y".into()];
        assert_eq!(lookup_message(&huge).unwrap().1, 1);
    }

    /// Answers every line with its own URI, in order, until the client hangs up.
    fn echo_uris(stream: UnixStream) {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            let request: Value = serde_json::from_str(&line).unwrap();
            let reply = json!({"status": "MISS", "error_message": request["uri"]});
            reader.get_mut().write_all(format!("{reply}\n").as_bytes()).unwrap();
            line.clear();
        }
    }

    #[test]
    fn locate_many_keeps_replies_in_order_across_groups() {
        let socket = socket_path("many");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || echo_uris(listener.accept().unwrap().0));
        let mut client = DaemonClient::new(socket.clone(), TIMEOUT);
        let wanted = uris(LOOKUP_BATCH_KEYS * 3 + 7);
        let replies = client.locate_many(&wanted).unwrap();
        let got: Vec<&str> = replies.iter().map(|r| r.message()).collect();
        assert_eq!(got, wanted.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(client.locate_many(&[]).unwrap().is_empty());
        drop(client);
        server.join().unwrap();
        let _ = std::fs::remove_dir_all(socket.parent().unwrap());
    }

    #[test]
    fn locate_many_survives_a_daemon_restart_on_a_kept_connection() {
        let socket = socket_path("many-restart");
        let listener = UnixListener::bind(&socket).unwrap();
        let requests = AtomicUsize::new(0);
        let mut client = DaemonClient::new(socket.clone(), TIMEOUT);
        std::thread::scope(|scope| {
            scope.spawn(|| answer_one(&listener, &requests));
            client.locate("hypercache://a").unwrap();
        });
        // The old connection is closed now, so the whole write should go out again on a new one
        listener.set_nonblocking(false).unwrap();
        let server = std::thread::spawn(move || echo_uris(listener.accept().unwrap().0));
        assert_eq!(client.locate_many(&uris(3)).unwrap().len(), 3);
        drop(client);
        server.join().unwrap();
        let _ = std::fs::remove_dir_all(socket.parent().unwrap());
    }

    #[test]
    fn locate_many_fails_whole_when_the_daemon_dies_part_way() {
        let (near, far) = UnixStream::pair().unwrap();
        let socket = socket_path("many-dies");
        let listener = UnixListener::bind(&socket).unwrap();
        let mut client = DaemonClient::new(socket.clone(), TIMEOUT);
        client.stream = Some(BufReader::new(near));
        let daemon = std::thread::spawn(move || {
            let mut reader = BufReader::new(far);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            reader.get_mut().write_all(b"{\"status\":\"HIT\"}\n").unwrap();
        });
        assert!(client.locate_many(&uris(4)).is_err());
        daemon.join().unwrap();
        assert!(
            client.stream.is_none(),
            "the stream position is unknown, so it must not be reused"
        );

        // The next call starts clean, and no reply from the old connection can be read by it
        let server = std::thread::spawn(move || echo_uris(listener.accept().unwrap().0));
        let replies = client.locate_many(&uris(2)).unwrap();
        assert_eq!(replies[1].message(), "hypercache://object/1");
        drop(client);
        server.join().unwrap();
        let _ = std::fs::remove_dir_all(socket.parent().unwrap());
    }
}
