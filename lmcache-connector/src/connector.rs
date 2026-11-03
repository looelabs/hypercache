//! The connector core: queues, worker threads and the block rules. No Python in here.
//!
//! LMCache's wrapper submits batches and must not block, so submit only queues work. Workers do
//! the I/O and report through a pollable fd. Rules:
//!
//! - Any doubt about a block is a miss. Lookup only accepts committed local-only blocks with a
//!   checksum for every chunk, and load verifies every chunk before reporting success.
//! - Blocks are never flushed to a backend. If the daemon can't prove it understands local-only
//!   commits, no data is written, because an older daemon would upload it.
//! - Checksum failures evict the block, unless it was replaced during the read.

use crate::daemon::{Chunk, DaemonClient, Response};
use crate::io::{BounceBuffer, ChunkIo, ReadError};
use crate::key::{key_to_uri, quote};
use std::any::Any;
use std::collections::{HashSet, VecDeque};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::PathBuf;
use std::sync::atomic::{
    AtomicBool, AtomicU64, AtomicU8, AtomicUsize,
    Ordering::{Relaxed, SeqCst},
};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const LOOKUP_WORKERS: usize = 2;

#[derive(Clone, Debug)]
pub struct Config {
    pub namespace: String,
    pub socket_path: PathBuf,
    pub tier: String,
    pub num_workers: usize,
    pub timeout: Duration,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.namespace.is_empty() {
            return Err(
                "namespace is required: use a value that changes when the model weights or dtype change".into(),
            );
        }
        if self.tier != "nvme" && self.tier != "ram" {
            return Err(format!("tier must be 'nvme' or 'ram', not '{}'", self.tier));
        }
        if self.num_workers < 1 {
            return Err("num_workers must be at least 1".into());
        }
        if self.timeout.is_zero() {
            return Err("timeout must be above zero".into());
        }
        Ok(())
    }
}

/// A caller's buffer. The submitter guarantees it stays valid and unused by anyone else until
/// its batch completes, by placing what keeps it alive in the batch's `keep_alive`.
#[derive(Clone, Copy)]
pub struct RawBuffer {
    pub ptr: *mut u8,
    pub len: usize,
}

unsafe impl Send for RawBuffer {}

impl RawBuffer {
    unsafe fn as_slice<'a>(self) -> &'a [u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }

    unsafe fn as_mut_slice<'a>(self) -> &'a mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }
}

pub struct Completion {
    pub future_id: u64,
    pub ok: bool,
    pub error: String,
    pub results: Vec<bool>,
    /// Whatever kept the batch's buffers alive. The caller decides where this gets dropped,
    /// which matters for Python buffers, since releasing those needs the interpreter lock.
    pub keep_alive: Box<dyn Any + Send>,
}

struct Batch {
    future_id: u64,
    remaining: AtomicUsize,
    results: Vec<AtomicU8>,
    first_error: Mutex<Option<String>>,
    failed: AtomicBool,
    keep_alive: Mutex<Option<Box<dyn Any + Send>>>,
}

enum Job {
    Store {
        batch: Arc<Batch>,
        index: usize,
        key: String,
        buffer: RawBuffer,
    },
    Load {
        batch: Arc<Batch>,
        index: usize,
        key: String,
        buffer: RawBuffer,
    },
    Exists {
        batch: Arc<Batch>,
        keys: Vec<String>,
    },
    Delete {
        batch: Arc<Batch>,
        keys: Vec<String>,
    },
}

struct Queue {
    jobs: Mutex<VecDeque<Job>>,
    ready: Condvar,
}

impl Queue {
    fn new() -> Self {
        Queue {
            jobs: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
        }
    }

    fn push_all(&self, jobs: impl IntoIterator<Item = Job>) {
        self.jobs.lock().expect("queue lock").extend(jobs);
        self.ready.notify_all();
    }
}

#[derive(Default)]
pub struct Counters {
    pub checksum_failures: AtomicU64,
    pub load_failures: AtomicU64,
    pub store_failures: AtomicU64,
}

/// The fd LMCache polls: a non-blocking pipe, readable while completions are waiting.
struct Notifier {
    read: OwnedFd,
    write: OwnedFd,
}

impl Notifier {
    fn new() -> io::Result<Self> {
        let mut fds: [RawFd; 2] = [0; 2];
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let (read, write) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
        for fd in fds {
            unsafe {
                libc::fcntl(fd, libc::F_SETFL, libc::fcntl(fd, libc::F_GETFL) | libc::O_NONBLOCK);
                libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
            }
        }
        Ok(Notifier { read, write })
    }

    fn signal(&self) {
        // A full pipe is fine: it is already readable, so the poller will wake
        let _ = unsafe { libc::write(self.write.as_raw_fd(), [1u8].as_ptr().cast(), 1) };
    }

    fn drain(&self) {
        let mut sink = [0u8; 256];
        while unsafe { libc::read(self.read.as_raw_fd(), sink.as_mut_ptr().cast(), sink.len()) } > 0 {}
    }
}

struct Shared {
    config: Config,
    namespace: String,
    store_queue: Queue,
    load_queue: Queue,
    lookup_queue: Queue,
    completions: Mutex<Vec<Completion>>,
    notifier: Notifier,
    storing: Mutex<HashSet<String>>,
    local_only_confirmed: AtomicBool,
    closed: AtomicBool,
    io: ChunkIo,
    counters: Counters,
}

pub struct Connector {
    shared: Arc<Shared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    next_future: AtomicU64,
    owner_pid: u32,
}

impl Connector {
    pub fn new(config: Config) -> io::Result<Self> {
        config
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let num_workers = config.num_workers;
        let shared = Arc::new(Shared {
            namespace: quote(&config.namespace),
            config,
            store_queue: Queue::new(),
            load_queue: Queue::new(),
            lookup_queue: Queue::new(),
            completions: Mutex::new(Vec::new()),
            notifier: Notifier::new()?,
            storing: Mutex::new(HashSet::new()),
            local_only_confirmed: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            io: ChunkIo::new(),
            counters: Counters::default(),
        });
        let mut workers = Vec::new();
        for (name, count, which) in [
            ("store", num_workers, 0usize),
            ("load", num_workers, 1),
            ("lookup", LOOKUP_WORKERS, 2),
        ] {
            for index in 0..count {
                let shared = shared.clone();
                workers.push(
                    std::thread::Builder::new()
                        .name(format!("hc-l2-{name}-{index}"))
                        .spawn(move || worker_loop(shared, which))?,
                );
            }
        }
        Ok(Connector {
            shared,
            workers: Mutex::new(workers),
            next_future: AtomicU64::new(1),
            owner_pid: std::process::id(),
        })
    }

    pub fn event_fd(&self) -> RawFd {
        self.shared.notifier.read.as_raw_fd()
    }

    pub fn uri(&self, key: &str) -> Result<String, String> {
        key_to_uri(&self.shared.namespace, key)
    }

    pub fn counters(&self) -> &Counters {
        &self.shared.counters
    }

    pub fn io_stats(&self) -> &crate::io::IoStats {
        &self.shared.io.stats
    }

    fn new_batch(&self, count: usize, jobs: usize, keep_alive: Box<dyn Any + Send>) -> io::Result<Arc<Batch>> {
        if self.shared.closed.load(SeqCst) {
            return Err(io::Error::other("connector is closed"));
        }
        // Worker threads don't survive a fork, so a child's work would never complete
        if std::process::id() != self.owner_pid {
            return Err(io::Error::other("connector was created in another process"));
        }
        let future_id = self.next_future.fetch_add(1, Relaxed);
        Ok(Arc::new(Batch {
            future_id,
            remaining: AtomicUsize::new(jobs),
            results: (0..count).map(|_| AtomicU8::new(0)).collect(),
            first_error: Mutex::new(None),
            failed: AtomicBool::new(false),
            keep_alive: Mutex::new(Some(keep_alive)),
        }))
    }

    fn check_lengths(keys: &[String], buffers: &[RawBuffer]) -> io::Result<()> {
        if keys.len() != buffers.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "keys and buffers differ in length",
            ));
        }
        Ok(())
    }

    /// # Safety
    /// Every buffer must stay valid and unmodified until this batch's completion is drained, and
    /// `keep_alive` must be what guarantees that.
    pub unsafe fn submit_set(
        &self,
        keys: Vec<String>,
        buffers: Vec<RawBuffer>,
        keep_alive: Box<dyn Any + Send>,
    ) -> io::Result<u64> {
        Self::check_lengths(&keys, &buffers)?;
        let batch = self.new_batch(keys.len(), keys.len(), keep_alive)?;
        let future_id = batch.future_id;
        let jobs: Vec<Job> = keys
            .into_iter()
            .zip(buffers)
            .enumerate()
            .map(|(index, (key, buffer))| Job::Store {
                batch: batch.clone(),
                index,
                key,
                buffer,
            })
            .collect();
        self.dispatch(&self.shared.store_queue, batch, jobs);
        Ok(future_id)
    }

    /// # Safety
    /// Every buffer must stay valid and unused by anyone else until this batch's completion is
    /// drained, and `keep_alive` must be what guarantees that.
    pub unsafe fn submit_get(
        &self,
        keys: Vec<String>,
        buffers: Vec<RawBuffer>,
        keep_alive: Box<dyn Any + Send>,
    ) -> io::Result<u64> {
        Self::check_lengths(&keys, &buffers)?;
        let batch = self.new_batch(keys.len(), keys.len(), keep_alive)?;
        let future_id = batch.future_id;
        let jobs: Vec<Job> = keys
            .into_iter()
            .zip(buffers)
            .enumerate()
            .map(|(index, (key, buffer))| Job::Load {
                batch: batch.clone(),
                index,
                key,
                buffer,
            })
            .collect();
        self.dispatch(&self.shared.load_queue, batch, jobs);
        Ok(future_id)
    }

    pub fn submit_exists(&self, keys: Vec<String>) -> io::Result<u64> {
        let batch = self.new_batch(keys.len(), 1, Box::new(()))?;
        let future_id = batch.future_id;
        let job = Job::Exists {
            batch: batch.clone(),
            keys,
        };
        self.dispatch(&self.shared.lookup_queue, batch, vec![job]);
        Ok(future_id)
    }

    pub fn submit_delete(&self, keys: Vec<String>) -> io::Result<u64> {
        let batch = self.new_batch(keys.len(), 1, Box::new(()))?;
        let future_id = batch.future_id;
        let job = Job::Delete {
            batch: batch.clone(),
            keys,
        };
        self.dispatch(&self.shared.lookup_queue, batch, vec![job]);
        Ok(future_id)
    }

    fn dispatch(&self, queue: &Queue, batch: Arc<Batch>, jobs: Vec<Job>) {
        if jobs.is_empty() {
            batch.remaining.store(1, SeqCst);
            finish_one(&self.shared, &batch);
        } else {
            queue.push_all(jobs);
        }
    }

    pub fn drain_completions(&self) -> Vec<Completion> {
        // Drain first. A completion pushed after the take below signals again, so none is missed.
        self.shared.notifier.drain();
        std::mem::take(&mut *self.shared.completions.lock().expect("completions lock"))
    }

    /// Stops the workers. Queued work fails fast, and I/O already running finishes first. After
    /// this returns, no worker touches any caller's buffer.
    pub fn close(&self) {
        self.shared.closed.store(true, SeqCst);
        for queue in [
            &self.shared.store_queue,
            &self.shared.load_queue,
            &self.shared.lookup_queue,
        ] {
            let _guard = queue.jobs.lock().expect("queue lock");
            queue.ready.notify_all();
        }
        let workers = std::mem::take(&mut *self.workers.lock().expect("workers lock"));
        let was_running = !workers.is_empty();
        for worker in workers {
            let _ = worker.join();
        }
        if was_running {
            self.shared.io.sync();
        }
    }
}

impl Drop for Connector {
    fn drop(&mut self) {
        self.close();
    }
}

struct Worker {
    shared: Arc<Shared>,
    client: DaemonClient,
    bounce: BounceBuffer,
}

fn worker_loop(shared: Arc<Shared>, which: usize) {
    let client = DaemonClient::new(shared.config.socket_path.clone(), shared.config.timeout);
    let mut worker = Worker {
        shared: shared.clone(),
        client,
        bounce: BounceBuffer::new(),
    };
    let queue = match which {
        0 => &shared.store_queue,
        1 => &shared.load_queue,
        _ => &shared.lookup_queue,
    };
    loop {
        let job = {
            let mut jobs = queue.jobs.lock().expect("queue lock");
            loop {
                if let Some(job) = jobs.pop_front() {
                    break job;
                }
                if shared.closed.load(SeqCst) {
                    return;
                }
                jobs = queue.ready.wait(jobs).expect("queue lock");
            }
        };
        worker.run(job);
    }
}

fn finish_one(shared: &Shared, batch: &Batch) {
    if batch.remaining.fetch_sub(1, SeqCst) != 1 {
        return;
    }
    let completion = Completion {
        future_id: batch.future_id,
        ok: !batch.failed.load(SeqCst),
        error: batch.first_error.lock().expect("error lock").take().unwrap_or_default(),
        results: batch.results.iter().map(|r| r.load(SeqCst) != 0).collect(),
        keep_alive: batch
            .keep_alive
            .lock()
            .expect("keep_alive lock")
            .take()
            .unwrap_or_else(|| Box::new(())),
    };
    shared.completions.lock().expect("completions lock").push(completion);
    shared.notifier.signal();
}

fn record_failure(batch: &Batch, message: String) {
    batch.failed.store(true, SeqCst);
    let mut first = batch.first_error.lock().expect("error lock");
    if first.is_none() {
        *first = Some(message);
    }
}

/// Removes a URI from the in-flight store set, however the store ends.
struct StoreGuard<'a> {
    shared: &'a Shared,
    uri: &'a str,
}

impl Drop for StoreGuard<'_> {
    fn drop(&mut self) {
        self.shared.storing.lock().expect("storing lock").remove(self.uri);
    }
}

fn layout_covers(layout: &[Chunk], size: usize) -> bool {
    layout.iter().map(|chunk| chunk.length).sum::<u64>() == size as u64
}

impl Worker {
    fn run(&mut self, job: Job) {
        let shared = self.shared.clone();
        let closed = shared.closed.load(SeqCst);
        match job {
            Job::Store {
                batch,
                index,
                key,
                buffer,
            } => {
                let result = if closed {
                    Err(io::Error::other("connector is closed"))
                } else {
                    self.store_one(&key, buffer)
                };
                match result {
                    Ok(()) => batch.results[index].store(1, SeqCst),
                    Err(error) => {
                        shared.counters.store_failures.fetch_add(1, Relaxed);
                        record_failure(&batch, format!("store failed for {key}: {error}"));
                    }
                }
                finish_one(&shared, &batch);
            }
            Job::Load {
                batch,
                index,
                key,
                buffer,
            } => {
                if !closed && self.load_one(&key, buffer) {
                    batch.results[index].store(1, SeqCst);
                } else {
                    record_failure(&batch, format!("load missed for {key}"));
                }
                finish_one(&shared, &batch);
            }
            Job::Exists { batch, keys } => {
                if closed {
                    record_failure(&batch, "connector is closed".into());
                } else {
                    self.exists(&batch, &keys);
                }
                finish_one(&shared, &batch);
            }
            Job::Delete { batch, keys } => {
                if !closed {
                    self.delete(&batch, &keys);
                }
                finish_one(&shared, &batch);
            }
        }
    }

    fn store_one(&mut self, key: &str, buffer: RawBuffer) -> io::Result<()> {
        let shared = self.shared.clone();
        let uri = shared.uri_for(key)?;
        if buffer.len == 0 {
            return Err(io::Error::other("empty object"));
        }
        if !shared.storing.lock().expect("storing lock").insert(uri.clone()) {
            return Ok(()); // already being stored
        }
        let _guard = StoreGuard {
            shared: &shared,
            uri: &uri,
        };

        if self.client.locate(&uri)?.is_valid_block() {
            return Ok(());
        }
        let staged = self.client.stage_write(&uri, buffer.len as u64, &shared.config.tier)?;
        let result = self.write_and_commit(&uri, staged.layout(), buffer);
        if result.is_err() {
            if let Err(error) = self.client.evict(&uri) {
                eprintln!("hypercache connector: could not clean up {uri}: {error}");
            }
        }
        result
    }

    fn write_and_commit(&mut self, uri: &str, layout: &[Chunk], buffer: RawBuffer) -> io::Result<()> {
        self.require_local_only_support(uri)?;
        if !layout_covers(layout, buffer.len) {
            return Err(io::Error::other("staged layout does not match the object size"));
        }
        let source = unsafe { buffer.as_slice() };
        let mut checksums = Vec::with_capacity(layout.len());
        let mut position = 0;
        for chunk in layout {
            let data = &source[position..position + chunk.length as usize];
            self.shared.io.write_chunk(chunk, data, &mut self.bounce)?;
            checksums.push(crc32fast::hash(data));
            position += chunk.length as usize;
        }
        self.client.commit_local_only(uri, &checksums)
    }

    /// An older daemon ignores local_only and would upload the block to its backend, so this is
    /// settled before the first commit. Only newer daemons report status on locate.
    fn require_local_only_support(&mut self, uri: &str) -> io::Result<()> {
        if self.shared.local_only_confirmed.load(SeqCst) {
            return Ok(());
        }
        if self.client.locate(uri)?.sync_status.as_deref() != Some("STAGED") {
            return Err(io::Error::other(
                "the HyperCache daemon has no local-only commit support; upgrade it",
            ));
        }
        self.shared.local_only_confirmed.store(true, SeqCst);
        Ok(())
    }

    fn load_one(&mut self, key: &str, buffer: RawBuffer) -> bool {
        let shared = self.shared.clone();
        let Ok(uri) = shared.uri_for(key) else {
            shared.counters.load_failures.fetch_add(1, Relaxed);
            return false;
        };
        let location = match self.client.locate(&uri) {
            Ok(location) => location,
            Err(error) => {
                eprintln!("hypercache connector: load failed for {uri}: {error}");
                shared.counters.load_failures.fetch_add(1, Relaxed);
                return false;
            }
        };
        if !location.is_valid_block() {
            return false;
        }
        if location.object_size != Some(buffer.len as u64) || !layout_covers(location.layout(), buffer.len) {
            eprintln!(
                "hypercache connector: size mismatch for {uri}: buffer is {} bytes",
                buffer.len
            );
            return false;
        }
        let target = unsafe { buffer.as_mut_slice() };
        let mut position = 0;
        for chunk in location.layout() {
            let length = chunk.length as usize;
            match shared
                .io
                .read_chunk(chunk, &mut target[position..position + length], &mut self.bounce)
            {
                Ok(()) => position += length,
                Err(ReadError::Checksum(mismatch)) => {
                    eprintln!(
                        "hypercache connector: treating {uri} as a miss: CRC32 at {}+{}: expected {:08x}, got {:08x}",
                        chunk.device, chunk.offset, mismatch.expected, mismatch.actual
                    );
                    shared.counters.checksum_failures.fetch_add(1, Relaxed);
                    self.evict_if_unchanged(&uri, &location);
                    return false;
                }
                Err(ReadError::Io(error)) => {
                    eprintln!("hypercache connector: load failed for {uri}: {error}");
                    shared.counters.load_failures.fetch_add(1, Relaxed);
                    return false;
                }
            }
        }
        true
    }

    /// A changed generation means the block was replaced during the read, so it isn't corrupt.
    fn evict_if_unchanged(&mut self, uri: &str, seen: &Response) {
        let current = self.client.locate(uri).map(|location| location.generation);
        let result = match current {
            Ok(generation) if generation == seen.generation => self.client.evict(uri).map(|_| ()),
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            eprintln!("hypercache connector: could not evict {uri}: {error}");
        }
    }

    /// A daemon failure fails the whole batch, which LMCache treats as all misses.
    fn exists(&mut self, batch: &Batch, keys: &[String]) {
        let shared = self.shared.clone();
        for (index, key) in keys.iter().enumerate() {
            let Ok(uri) = shared.uri_for(key) else { continue };
            match self.client.locate(&uri) {
                Ok(location) if location.is_valid_block() => batch.results[index].store(1, SeqCst),
                Ok(_) => {}
                Err(error) => {
                    record_failure(batch, format!("lookup failed: {error}"));
                    return;
                }
            }
        }
    }

    fn delete(&mut self, batch: &Batch, keys: &[String]) {
        let shared = self.shared.clone();
        for (index, key) in keys.iter().enumerate() {
            let Ok(uri) = shared.uri_for(key) else { continue };
            let deleted = self.client.locate(&uri).and_then(|location| {
                if location.status == "HIT" {
                    self.client.evict(&uri)
                } else {
                    Ok(false)
                }
            });
            match deleted {
                Ok(true) => batch.results[index].store(1, SeqCst),
                Ok(false) => {}
                Err(error) => eprintln!("hypercache connector: delete failed for {uri}: {error}"),
            }
        }
    }
}

impl Shared {
    fn uri_for(&self, key: &str) -> io::Result<String> {
        key_to_uri(&self.namespace, key).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::fs::FileExt;
    use std::os::unix::net::UnixListener;
    use std::path::Path;
    use std::time::Instant;

    const CHUNK: u64 = 4 * 1024 * 1024;

    #[derive(Clone)]
    struct Stored {
        layout: Vec<Value>,
        size: u64,
        generation: u64,
        status: &'static str,
    }

    #[derive(Default)]
    struct DaemonState {
        objects: HashMap<String, Stored>,
        next_offset: u64,
        commits: usize,
        requests: usize,
        requested_flush: usize,
    }

    /// Speaks the real wire format with file-backed storage.
    struct FakeDaemon {
        socket: PathBuf,
        device: PathBuf,
        state: Arc<Mutex<DaemonState>>,
        /// An older daemon reports no sync_status on locate
        legacy: bool,
    }

    impl FakeDaemon {
        fn start(name: &str, legacy: bool) -> FakeDaemon {
            let dir = std::env::temp_dir().join(format!("hc-conn-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let device = dir.join("device.bin");
            std::fs::File::create(&device)
                .unwrap()
                .set_len(256 * 1024 * 1024)
                .unwrap();
            let socket = dir.join("d.sock");
            let listener = UnixListener::bind(&socket).unwrap();
            let state = Arc::new(Mutex::new(DaemonState::default()));
            let daemon = FakeDaemon {
                socket,
                device: device.clone(),
                state: state.clone(),
                legacy,
            };
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let stream = stream.unwrap();
                    let state = state.clone();
                    let device = device.clone();
                    std::thread::spawn(move || serve(stream, state, device, legacy));
                }
            });
            daemon
        }

        fn connector(&self, namespace: &str) -> Connector {
            Connector::new(Config {
                namespace: namespace.into(),
                socket_path: self.socket.clone(),
                tier: "nvme".into(),
                num_workers: 4,
                timeout: Duration::from_secs(5),
            })
            .unwrap()
        }
    }

    impl Drop for FakeDaemon {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.device.parent().unwrap());
        }
    }

    fn serve(stream: std::os::unix::net::UnixStream, state: Arc<Mutex<DaemonState>>, device: PathBuf, legacy: bool) {
        let mut writer = stream.try_clone().unwrap();
        for line in BufReader::new(stream).lines() {
            let Ok(line) = line else { return };
            let request: Value = serde_json::from_str(&line).unwrap();
            let response = handle(&request, &state, &device, legacy);
            writer.write_all(format!("{response}\n").as_bytes()).unwrap();
        }
    }

    fn handle(request: &Value, state: &Mutex<DaemonState>, device: &Path, legacy: bool) -> Value {
        let mut state = state.lock().unwrap();
        state.requests += 1;
        let uri = request["uri"].as_str().unwrap_or_default().to_string();
        match request["action"].as_str().unwrap() {
            "locate" => match state.objects.get(&uri) {
                None => json!({"status": "MISS"}),
                Some(o) if o.status == "STAGED" => {
                    if legacy {
                        json!({"status": "MISS"})
                    } else {
                        json!({"status": "MISS", "sync_status": "STAGED"})
                    }
                }
                Some(o) => json!({
                    "status": "HIT", "object_size": o.size, "chunk_layout": o.layout,
                    "generation": o.generation, "sync_status": o.status,
                }),
            },
            "stage_write" => {
                let size = request["size_bytes"].as_u64().unwrap();
                if request["auto_flush_s3"].as_bool() != Some(false) {
                    state.requested_flush += 1;
                }
                let mut layout = Vec::new();
                let mut remaining = size;
                while remaining > 0 {
                    let length = remaining.min(CHUNK);
                    let offset = state.next_offset;
                    state.next_offset += CHUNK;
                    layout.push(json!({
                        "tier": "nvme", "device": device.to_str().unwrap(), "offset": offset, "length": length,
                    }));
                    remaining -= length;
                }
                let generation = state.next_offset;
                state.objects.insert(
                    uri,
                    Stored {
                        layout: layout.clone(),
                        size,
                        generation,
                        status: "STAGED",
                    },
                );
                json!({"status": "WRITESTAGED", "object_size": size, "chunk_layout": layout})
            }
            "commit_write" => {
                let local_only = request["local_only"].as_bool().unwrap_or(false);
                let checksums = request["chunk_checksums"].as_array().cloned().unwrap_or_default();
                state.commits += 1;
                let Some(object) = state.objects.get_mut(&uri) else {
                    return json!({"status": "ERROR", "error_message": "not staged"});
                };
                for (chunk, checksum) in object.layout.iter_mut().zip(&checksums) {
                    chunk["checksum"] = checksum.clone();
                }
                object.status = if local_only { "LOCAL_ONLY" } else { "COMMITTED" };
                json!({"status": "WRITECOMMITTED"})
            }
            "evict" => {
                if state.objects.remove(&uri).is_some() {
                    json!({"status": "EVICTED"})
                } else {
                    json!({"status": "ERROR", "error_message": "no such object"})
                }
            }
            other => panic!("unexpected action {other}"),
        }
    }

    /// A page-aligned test buffer.
    struct Aligned {
        storage: Vec<u8>,
        start: usize,
        len: usize,
    }

    impl Aligned {
        fn new(data: &[u8]) -> Aligned {
            let mut storage = vec![0u8; data.len() + 4096];
            let start = (4096 - storage.as_ptr() as usize % 4096) % 4096;
            storage[start..start + data.len()].copy_from_slice(data);
            Aligned {
                storage,
                start,
                len: data.len(),
            }
        }

        fn raw(&mut self) -> RawBuffer {
            RawBuffer {
                ptr: self.storage[self.start..].as_mut_ptr(),
                len: self.len,
            }
        }

        fn bytes(&self) -> &[u8] {
            &self.storage[self.start..self.start + self.len]
        }
    }

    fn pattern(length: usize, seed: u8) -> Vec<u8> {
        (0..length).map(|i| ((i / 7) as u8).wrapping_add(seed)).collect()
    }

    fn key(n: u8) -> String {
        format!("Qwen/test@00000000@0@{n:02x}ab")
    }

    fn wait(connector: &Connector, future_id: u64) -> Completion {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut pending = Vec::new();
        loop {
            pending.extend(connector.drain_completions());
            if let Some(position) = pending.iter().position(|c| c.future_id == future_id) {
                return pending.swap_remove(position);
            }
            assert!(Instant::now() < deadline, "timed out waiting for {future_id}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn store(connector: &Connector, keys: &[String], buffers: &mut [Aligned]) -> Completion {
        let raws: Vec<RawBuffer> = buffers.iter_mut().map(|b| b.raw()).collect();
        let id = unsafe { connector.submit_set(keys.to_vec(), raws, Box::new(())) }.unwrap();
        wait(connector, id)
    }

    fn load(connector: &Connector, keys: &[String], buffers: &mut [Aligned]) -> Completion {
        let raws: Vec<RawBuffer> = buffers.iter_mut().map(|b| b.raw()).collect();
        let id = unsafe { connector.submit_get(keys.to_vec(), raws, Box::new(())) }.unwrap();
        wait(connector, id)
    }

    fn exists(connector: &Connector, keys: &[String]) -> Vec<bool> {
        let id = connector.submit_exists(keys.to_vec()).unwrap();
        wait(connector, id).results
    }

    const SIZE: usize = 14 * 1024 * 1024 + 123;

    #[test]
    fn round_trip_with_partial_last_block() {
        let daemon = FakeDaemon::start("roundtrip", false);
        let connector = daemon.connector("ns");
        let keys: Vec<String> = (0..4).map(key).collect();
        let datas: Vec<Vec<u8>> = (0..4).map(|i| pattern(SIZE, i)).collect();
        let mut sources: Vec<Aligned> = datas.iter().map(|d| Aligned::new(d)).collect();
        let done = store(&connector, &keys, &mut sources);
        assert!(done.ok, "{}", done.error);
        assert_eq!(done.results, [true; 4]);

        let mut query = keys.clone();
        query.push(key(99));
        assert_eq!(exists(&connector, &query), [true, true, true, true, false]);

        let mut targets: Vec<Aligned> = (0..4).map(|_| Aligned::new(&vec![0u8; SIZE])).collect();
        let done = load(&connector, &keys, &mut targets);
        assert!(done.ok, "{}", done.error);
        for (target, data) in targets.iter().zip(&datas) {
            assert_eq!(target.bytes(), &data[..]);
        }
        let state = daemon.state.lock().unwrap();
        assert_eq!(state.requested_flush, 0, "must never ask for a backend flush");
    }

    #[test]
    fn repeat_store_writes_nothing() {
        let daemon = FakeDaemon::start("repeat", false);
        let connector = daemon.connector("ns");
        let keys = vec![key(1)];
        let mut source = vec![Aligned::new(&pattern(SIZE, 1))];
        assert!(store(&connector, &keys, &mut source).ok);
        let commits = daemon.state.lock().unwrap().commits;
        assert!(store(&connector, &keys, &mut source).ok);
        assert_eq!(daemon.state.lock().unwrap().commits, commits);
    }

    #[test]
    fn older_daemon_gets_no_data_and_no_commit() {
        let daemon = FakeDaemon::start("legacy", true);
        assert!(daemon.legacy);
        let connector = daemon.connector("ns");
        let keys = vec![key(1)];
        let data = pattern(SIZE, 1);
        let mut source = vec![Aligned::new(&data)];
        let done = store(&connector, &keys, &mut source);
        assert!(!done.ok);
        assert!(done.error.contains("local-only"), "{}", done.error);
        let state = daemon.state.lock().unwrap();
        assert_eq!(state.commits, 0);
        assert!(state.objects.is_empty(), "the staged object must be cleaned up");
        let on_disk = std::fs::read(&daemon.device).unwrap();
        assert!(on_disk[..4096].iter().all(|&b| b == 0), "no data may reach the device");
        assert_eq!(connector.counters().store_failures.load(Relaxed), 1);
    }

    #[test]
    fn corrupt_block_is_a_miss_and_is_evicted() {
        let daemon = FakeDaemon::start("corrupt", false);
        let connector = daemon.connector("ns");
        let keys = vec![key(1), key(2)];
        let mut sources = vec![Aligned::new(&pattern(SIZE, 1)), Aligned::new(&pattern(SIZE, 2))];
        assert!(store(&connector, &keys, &mut sources).ok);

        // Damage the first object's last chunk
        let uri = connector.uri(&keys[0]).unwrap();
        let (device, offset) = {
            let state = daemon.state.lock().unwrap();
            let last = state.objects[&uri].layout.last().unwrap().clone();
            (
                last["device"].as_str().unwrap().to_string(),
                last["offset"].as_u64().unwrap(),
            )
        };
        std::fs::OpenOptions::new()
            .write(true)
            .open(device)
            .unwrap()
            .write_all_at(&[0xEE], offset + 9)
            .unwrap();

        let mut targets = vec![Aligned::new(&vec![0u8; SIZE]), Aligned::new(&vec![0u8; SIZE])];
        let done = load(&connector, &keys, &mut targets);
        assert_eq!(done.results, [false, true]);
        assert!(!done.ok);
        assert_eq!(connector.counters().checksum_failures.load(Relaxed), 1);
        assert_eq!(exists(&connector, &keys), [false, true]);
    }

    #[test]
    fn replaced_block_is_not_evicted_after_a_checksum_failure() {
        let daemon = FakeDaemon::start("generation", false);
        let connector = daemon.connector("ns");
        let keys = vec![key(1)];
        let mut sources = vec![Aligned::new(&pattern(SIZE, 1))];
        assert!(store(&connector, &keys, &mut sources).ok);
        let uri = connector.uri(&keys[0]).unwrap();
        let seen = {
            let state = daemon.state.lock().unwrap();
            let o = &state.objects[&uri];
            json!({"status": "HIT", "generation": o.generation - 1})
        };
        let seen: Response = serde_json::from_value(seen).unwrap();
        let mut worker = Worker {
            shared: connector.shared.clone(),
            client: DaemonClient::new(daemon.socket.clone(), Duration::from_secs(5)),
            bounce: BounceBuffer::new(),
        };
        worker.evict_if_unchanged(&uri, &seen);
        assert_eq!(exists(&connector, &keys), [true], "a replaced block must survive");
    }

    #[test]
    fn wrong_buffer_size_is_a_miss_and_leaves_the_block() {
        let daemon = FakeDaemon::start("size", false);
        let connector = daemon.connector("ns");
        let keys = vec![key(1)];
        let mut sources = vec![Aligned::new(&pattern(SIZE, 1))];
        assert!(store(&connector, &keys, &mut sources).ok);
        let mut wrong = vec![Aligned::new(&vec![7u8; SIZE - 4096])];
        let done = load(&connector, &keys, &mut wrong);
        assert_eq!(done.results, [false]);
        assert!(
            wrong[0].bytes().iter().all(|&b| b == 7),
            "the buffer must not be touched"
        );
        assert_eq!(exists(&connector, &keys), [true]);
    }

    #[test]
    fn staged_or_uncommitted_blocks_are_misses() {
        let daemon = FakeDaemon::start("staged", false);
        let connector = daemon.connector("ns");
        let uri = connector.uri(&key(1)).unwrap();
        let chunk = json!({"tier": "nvme", "device": "/x", "offset": 0, "length": 8, "checksum": 1});
        let no_sum = json!({"tier": "nvme", "device": "/x", "offset": 0, "length": 8});
        for (status, layout) in [
            ("COMMITTED", vec![chunk.clone()]),
            ("LOCAL_ONLY", vec![]),
            ("LOCAL_ONLY", vec![no_sum]),
        ] {
            daemon.state.lock().unwrap().objects.insert(
                uri.clone(),
                Stored {
                    layout,
                    size: 8,
                    generation: 1,
                    status,
                },
            );
            assert_eq!(exists(&connector, &[key(1)]), [false], "{status}");
        }
    }

    #[test]
    fn namespace_and_salt_isolate_blocks() {
        let daemon = FakeDaemon::start("isolate", false);
        let connector = daemon.connector("a");
        let mut sources = vec![Aligned::new(&pattern(SIZE, 1))];
        assert!(store(&connector, &[key(1)], &mut sources).ok);
        assert_eq!(exists(&daemon.connector("b"), &[key(1)]), [false]);
        assert_eq!(exists(&connector, &[format!("{}@tenant", key(1))]), [false]);
    }

    #[test]
    fn malformed_key_is_a_miss_on_every_path() {
        let daemon = FakeDaemon::start("badkey", false);
        let connector = daemon.connector("ns");
        let bad = vec!["no-separators".to_string()];
        assert_eq!(exists(&connector, &bad), [false]);
        let mut buffers = vec![Aligned::new(&vec![1u8; 4096])];
        assert_eq!(load(&connector, &bad, &mut buffers).results, [false]);
        assert!(!store(&connector, &bad, &mut buffers).ok);
    }

    #[test]
    fn unreachable_daemon_is_a_failed_batch_then_recovers() {
        let daemon = FakeDaemon::start("down", false);
        let connector = Connector::new(Config {
            namespace: "ns".into(),
            socket_path: "/nonexistent/socket".into(),
            tier: "nvme".into(),
            num_workers: 1,
            timeout: Duration::from_secs(1),
        })
        .unwrap();
        let id = connector.submit_exists(vec![key(1)]).unwrap();
        let done = wait(&connector, id);
        assert!(!done.ok);
        assert_eq!(done.results, [false]);
        let mut buffers = vec![Aligned::new(&vec![0u8; 4096])];
        assert_eq!(load(&connector, &[key(1)], &mut buffers).results, [false]);
        drop(daemon);
    }

    #[test]
    fn delete_removes_only_existing_blocks() {
        let daemon = FakeDaemon::start("delete", false);
        let connector = daemon.connector("ns");
        let mut sources = vec![Aligned::new(&pattern(SIZE, 1))];
        assert!(store(&connector, &[key(1)], &mut sources).ok);
        let id = connector.submit_delete(vec![key(1), key(2)]).unwrap();
        assert_eq!(wait(&connector, id).results, [true, false]);
        assert_eq!(exists(&connector, &[key(1)]), [false]);
    }

    #[test]
    fn fd_becomes_readable_and_is_drained() {
        let daemon = FakeDaemon::start("poll", false);
        let connector = daemon.connector("ns");
        let poll_once = |timeout_ms| {
            let mut fds = libc::pollfd {
                fd: connector.event_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            unsafe { libc::poll(&mut fds, 1, timeout_ms) }
        };
        assert_eq!(poll_once(0), 0);
        connector.submit_exists(vec![key(1)]).unwrap();
        assert_eq!(poll_once(5000), 1);
        assert_eq!(connector.drain_completions().len(), 1);
        assert_eq!(poll_once(0), 0, "a drained fd must not keep waking the poller");
    }

    #[test]
    fn empty_batches_complete_at_once() {
        let daemon = FakeDaemon::start("empty", false);
        let connector = daemon.connector("ns");
        let id = connector.submit_exists(vec![]).unwrap();
        let done = wait(&connector, id);
        assert!(done.ok && done.results.is_empty());
    }

    #[test]
    fn close_stops_workers_and_refuses_new_work() {
        let daemon = FakeDaemon::start("close", false);
        let connector = daemon.connector("ns");
        let mut sources: Vec<Aligned> = (0..16).map(|i| Aligned::new(&pattern(1 << 20, i))).collect();
        let keys: Vec<String> = (0..16).map(key).collect();
        let raws: Vec<RawBuffer> = sources.iter_mut().map(|b| b.raw()).collect();
        let id = unsafe { connector.submit_set(keys, raws, Box::new(())) }.unwrap();
        connector.close();
        connector.close();
        let finished = connector.drain_completions();
        assert_eq!(finished.len(), 1, "queued work must still complete");
        assert_eq!(finished[0].future_id, id);
        assert!(connector.submit_exists(vec![key(1)]).is_err());
    }

    #[test]
    fn keep_alive_comes_back_with_the_completion() {
        let daemon = FakeDaemon::start("keep", false);
        let connector = daemon.connector("ns");
        let marker = Arc::new(());
        let mut sources = [Aligned::new(&pattern(4096, 1))];
        let raws: Vec<RawBuffer> = sources.iter_mut().map(|b| b.raw()).collect();
        let id = unsafe { connector.submit_set(vec![key(1)], raws, Box::new(marker.clone())) }.unwrap();
        let done = wait(&connector, id);
        assert_eq!(Arc::strong_count(&marker), 2, "workers must not drop it");
        drop(done);
        assert_eq!(Arc::strong_count(&marker), 1);
    }

    #[test]
    fn bad_config_is_refused() {
        let base = Config {
            namespace: "ns".into(),
            socket_path: "/x".into(),
            tier: "nvme".into(),
            num_workers: 1,
            timeout: Duration::from_secs(1),
        };
        assert!(base.validate().is_ok());
        assert!(Config {
            namespace: String::new(),
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(Config {
            tier: "s3".into(),
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(Config {
            num_workers: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(Config {
            timeout: Duration::ZERO,
            ..base
        }
        .validate()
        .is_err());
    }
}
