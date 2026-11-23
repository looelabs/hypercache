//! Group commit for device flushes.
//!
//! A commit must not return before a device flush that covers its data has finished. A flush covers
//! every write that completed before the flush started, so a caller needs a flush that *starts*
//! after it arrived. Flushes are numbered 1, 2, 3, ... A caller that arrives while flush `k` is
//! running needs flush `k + 1`; one that arrives while no flush is running starts the next one
//! itself. Every caller whose need a flush meets returns with that flush's result, so many
//! concurrent commits share one flush instead of queueing one flush each.
//!
//! The flush runs in its own task, so a caller that is cancelled while waiting cannot leave the
//! gate stuck with a flush marked as running.

use std::io;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

/// The flush operation, for example `File::sync_data` on the device.
pub type FlushFn = Arc<dyn Fn() -> io::Result<()> + Send + Sync>;

#[derive(Default)]
pub struct FlushGate {
    state: Mutex<FlushState>,
    done: Notify,
}

#[derive(Default)]
struct FlushState {
    /// Number of the most recently started flush (0 = none yet).
    started: u64,
    /// Flushes `1..=completed` have finished.
    completed: u64,
    running: bool,
    /// The most recent flush that failed, and its error. A caller fails if this flush is at or
    /// after the one it needed, which can only report a failure too many, never one too few.
    last_failure: Option<(u64, String)>,
}

impl FlushGate {
    /// Returns once a flush that started after this call has finished, with that flush's result.
    pub async fn flush(self: &Arc<Self>, flush_fn: FlushFn) -> Result<(), String> {
        let need = self.state.lock().unwrap().started + 1;
        loop {
            let notified = {
                let mut st = self.state.lock().unwrap();
                if st.completed >= need {
                    return match &st.last_failure {
                        Some((k, e)) if *k >= need => Err(e.clone()),
                        _ => Ok(()),
                    };
                }
                if !st.running {
                    st.running = true;
                    st.started += 1;
                    self.spawn_flush(st.started, Arc::clone(&flush_fn));
                }
                // Created under the lock, so the completion's notify_waiters, which needs the lock
                // first, cannot be missed.
                self.done.notified()
            };
            notified.await;
        }
    }

    fn spawn_flush(self: &Arc<Self>, number: u64, flush_fn: FlushFn) {
        let gate = Arc::clone(self);
        tokio::spawn(async move {
            let result = tokio::task::spawn_blocking(move || flush_fn()).await;
            {
                let mut st = gate.state.lock().unwrap();
                st.completed = number;
                st.running = false;
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => st.last_failure = Some((number, e.to_string())),
                    Err(e) => st.last_failure = Some((number, format!("flush task panicked: {}", e))),
                }
            }
            gate.done.notify_waiters();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn counting(delay_ms: u64, count: Arc<AtomicUsize>) -> FlushFn {
        Arc::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(delay_ms));
            Ok(())
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_callers_share_flushes() {
        let gate = Arc::new(FlushGate::default());
        let count = Arc::new(AtomicUsize::new(0));
        let f = counting(30, Arc::clone(&count));
        let tasks: Vec<_> = (0..100)
            .map(|_| {
                let (g, f) = (Arc::clone(&gate), Arc::clone(&f));
                tokio::spawn(async move { g.flush(f).await })
            })
            .collect();
        for t in tasks {
            assert_eq!(t.await.unwrap(), Ok(()));
        }
        let n = count.load(Ordering::SeqCst);
        assert!((1..=3).contains(&n), "100 concurrent callers ran {} flushes", n);
    }

    /// A caller that arrives while a flush is running must wait for a flush that starts later.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn caller_arriving_mid_flush_waits_for_the_next_flush() {
        let gate = Arc::new(FlushGate::default());
        let starts = Arc::new(AtomicUsize::new(0));
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let s = Arc::clone(&starts);
        let f: FlushFn = Arc::new(move || {
            s.fetch_add(1, Ordering::SeqCst);
            release_rx.lock().unwrap().recv().unwrap();
            Ok(())
        });

        let first = tokio::spawn({
            let (g, f) = (Arc::clone(&gate), Arc::clone(&f));
            async move { g.flush(f).await }
        });
        while starts.load(Ordering::SeqCst) < 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let second = tokio::spawn({
            let (g, f) = (Arc::clone(&gate), Arc::clone(&f));
            async move { g.flush(f).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        release_tx.send(()).unwrap(); // flush 1 finishes
        assert_eq!(first.await.unwrap(), Ok(()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !second.is_finished(),
            "second caller returned on a flush that started before it arrived"
        );
        assert_eq!(starts.load(Ordering::SeqCst), 2, "the second caller must start flush 2");
        release_tx.send(()).unwrap(); // flush 2 finishes
        assert_eq!(second.await.unwrap(), Ok(()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn failed_flush_fails_every_caller_it_covered() {
        let gate = Arc::new(FlushGate::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        let f: FlushFn = Arc::new(move || {
            std::thread::sleep(Duration::from_millis(30));
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(io::Error::other("injected EIO"))
            } else {
                Ok(())
            }
        });
        let tasks: Vec<_> = (0..20)
            .map(|_| {
                let (g, f) = (Arc::clone(&gate), Arc::clone(&f));
                tokio::spawn(async move { g.flush(f).await })
            })
            .collect();
        let mut failed = 0;
        for t in tasks {
            match t.await.unwrap() {
                Ok(()) => {}
                Err(e) => {
                    assert!(e.contains("injected EIO"));
                    failed += 1;
                }
            }
        }
        // The first flush failed and at least its starter needed it. Callers that needed flush 2
        // may also see it reported (failures are reported too often, never too rarely).
        assert!(failed >= 1);
        // A caller arriving after the failure succeeds on a new flush.
        let late = gate.flush(Arc::clone(&f)).await;
        assert_eq!(late, Ok(()));
    }

    /// Dropping a waiting caller, even the one that started the flush, must not leave the gate stuck.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancelled_caller_does_not_wedge_the_gate() {
        let gate = Arc::new(FlushGate::default());
        let count = Arc::new(AtomicUsize::new(0));
        let f = counting(100, Arc::clone(&count));
        let starter = tokio::spawn({
            let (g, f) = (Arc::clone(&gate), Arc::clone(&f));
            async move { g.flush(f).await }
        });
        while count.load(Ordering::SeqCst) < 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        starter.abort();
        let r = tokio::time::timeout(Duration::from_secs(5), gate.flush(Arc::clone(&f))).await;
        assert_eq!(r.expect("gate stuck after a cancelled caller"), Ok(()));
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }
}
