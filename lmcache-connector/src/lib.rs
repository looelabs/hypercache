//! LMCache native L2 connector backed by the HyperCache daemon.
//!
//! LMCache's `native_plugin` type loads a class that exposes `event_fd`, `submit_batch_*`,
//! `drain_completions` and `close`. This crate provides that class in Rust, so everything below
//! LMCache's own wrapper runs without Python or the interpreter lock.

pub mod connector;
pub mod daemon;
pub mod io;
pub mod key;

#[cfg(feature = "python")]
mod python;
