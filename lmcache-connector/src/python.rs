//! The Python face of the connector. Keeps to what crosses the boundary: arguments, buffers and
//! completions. Workers never touch the interpreter, and buffers are released in
//! `drain_completions`, which runs with the interpreter lock held.

use crate::connector::{Config, Connector, RawBuffer};
use crate::lmcache_version::{untested_version_problem, TESTED_LMCACHE_VERSIONS};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{ffi, IntoPyObjectExt};
use std::env;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

/// A Python buffer held open until its batch completes.
struct HeldBuffer(Box<ffi::Py_buffer>);

// A Py_buffer is plain data that keeps its exporter alive. Its only unsafe use is the release in
// `drop`, which takes the interpreter lock.
unsafe impl Send for HeldBuffer {}

impl HeldBuffer {
    fn acquire(object: &Bound<'_, PyAny>, writable: bool) -> PyResult<Self> {
        let mut raw = Box::new(MaybeUninit::<ffi::Py_buffer>::zeroed());
        // PyBUF_SIMPLE asks for one contiguous run of bytes, and fails for anything else
        let flags = if writable {
            ffi::PyBUF_WRITABLE
        } else {
            ffi::PyBUF_SIMPLE
        };
        if unsafe { ffi::PyObject_GetBuffer(object.as_ptr(), raw.as_mut_ptr(), flags) } != 0 {
            return Err(PyErr::fetch(object.py()));
        }
        // PyObject_GetBuffer initialized it
        let raw = unsafe { Box::from_raw(Box::into_raw(raw).cast::<ffi::Py_buffer>()) };
        Ok(HeldBuffer(raw))
    }

    fn raw(&self) -> RawBuffer {
        RawBuffer {
            ptr: self.0.buf.cast(),
            len: self.0.len as usize,
        }
    }
}

impl Drop for HeldBuffer {
    fn drop(&mut self) {
        Python::with_gil(|_| unsafe { ffi::PyBuffer_Release(&mut *self.0) });
    }
}

fn acquire_all(memviews: &Bound<'_, PyAny>, writable: bool) -> PyResult<(Vec<HeldBuffer>, Vec<RawBuffer>)> {
    let mut held = Vec::new();
    for memview in memviews.try_iter()? {
        held.push(HeldBuffer::acquire(&memview?, writable)?);
    }
    let raws = held.iter().map(HeldBuffer::raw).collect();
    Ok((held, raws))
}

fn default_socket_path() -> PathBuf {
    // Same order as the Python client's get_default_socket_path
    if let Some(path) = env::var("HYPERCACHE_SOCKET_PATH")
        .ok()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
    {
        return PathBuf::from(path);
    }
    const PRIMARY: &str = "/run/hypercache/hypercache.sock";
    const FALLBACK: &str = "/tmp/hypercache/hypercache.sock";
    const LEGACY: &str = "/tmp/hypercache.sock";
    for path in [PRIMARY, FALLBACK, LEGACY] {
        if Path::new(path).exists() {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(if Path::new("/run").exists() { PRIMARY } else { FALLBACK })
}

fn runtime_error(error: std::io::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Why LMCache's compiled GPU copy kernels can't be used, or None when they load or don't apply.
/// Without them LMCache falls back to plain torch ops with a single log line. That fallback can
/// store stale KV data (LMCache issue #6673), which then changes answers after a cache hit, so the
/// connector says so loudly.
fn lmcache_gpu_ops_problem(py: Python<'_>) -> PyResult<Option<String>> {
    let find_spec = py.import("importlib.util")?.getattr("find_spec")?;
    if find_spec.call1(("lmcache",))?.is_none() {
        return Ok(None);
    }
    let error = match py.import("lmcache.cuda_ops") {
        Ok(_) => return Ok(None),
        Err(error) => error,
    };
    // Only CUDA builds have these kernels. Without torch or a CUDA GPU there is nothing to check.
    let Ok(torch) = py.import("torch") else {
        return Ok(None);
    };
    let cuda = torch.getattr("cuda")?.call_method0("is_available")?.extract::<bool>()?;
    let hip = !torch.getattr("version")?.getattr("hip")?.is_none();
    if !cuda || hip {
        return Ok(None);
    }
    let torch_version: String = torch.getattr("__version__")?.extract()?;
    Ok(Some(format!(
        "lmcache.cuda_ops failed to load under torch {torch_version}: {}",
        error.value(py)
    )))
}

/// Checks that LMCache's compiled GPU copy kernels load. Returns None when they do, or the reason
/// they don't. The connector runs this at startup and prints a warning on a problem.
#[pyfunction]
fn check_lmcache_gpu_ops(py: Python<'_>) -> PyResult<Option<String>> {
    lmcache_gpu_ops_problem(py)
}

fn warn_on_lmcache_gpu_ops(py: Python<'_>) {
    match lmcache_gpu_ops_problem(py) {
        Ok(None) => {}
        Ok(Some(problem)) => eprintln!(
            "hypercache connector: WARNING: {problem}. LMCache is using its torch fallback for GPU \
             copies, which can store wrong KV data and change model output after a cache hit \
             (LMCache issue #6673). Install an LMCache build that matches this torch version. KV \
             data stored while this warning showed may be wrong: after the fix, use a new namespace."
        ),
        Err(error) => eprintln!("hypercache connector: could not check lmcache.cuda_ops: {error}"),
    }
}

/// The installed LMCache version, from its package metadata, or None when it isn't installed.
fn installed_lmcache_version(py: Python<'_>) -> PyResult<Option<String>> {
    let metadata = py.import("importlib.metadata")?;
    match metadata.getattr("version")?.call1(("lmcache",)) {
        Ok(version) => Ok(Some(version.extract()?)),
        Err(error) if error.is_instance(py, &metadata.getattr("PackageNotFoundError")?) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Checks the installed LMCache against the versions this connector was tested with. Returns None
/// when it is one of them (or LMCache isn't installed), or a description of the problem. The
/// connector runs this at startup and prints a warning on a problem; it never refuses to start.
#[pyfunction]
fn check_lmcache_version(py: Python<'_>) -> PyResult<Option<String>> {
    Ok(installed_lmcache_version(py)?.and_then(|v| untested_version_problem(&v)))
}

fn warn_on_lmcache_version(py: Python<'_>) {
    match check_lmcache_version(py) {
        Ok(None) => {}
        Ok(Some(problem)) => eprintln!("hypercache connector: WARNING: {problem}."),
        Err(error) => eprintln!("hypercache connector: could not read the LMCache version: {error}"),
    }
}

/// Stores LMCache objects in HyperCache. Load it through LMCache's `native_plugin` adapter type.
#[pyclass(module = "hypercache_lmcache_native")]
struct HyperCacheNativeConnector {
    inner: Connector,
}

#[pymethods]
impl HyperCacheNativeConnector {
    #[new]
    #[pyo3(signature = (namespace, socket_path=None, tier="nvme", num_workers=32, timeout_s=30.0))]
    fn new(
        py: Python<'_>,
        namespace: &str,
        socket_path: Option<&str>,
        tier: &str,
        num_workers: usize,
        timeout_s: f64,
    ) -> PyResult<Self> {
        if !(timeout_s.is_finite() && timeout_s > 0.0) {
            return Err(PyValueError::new_err("timeout_s must be above zero"));
        }
        warn_on_lmcache_version(py);
        warn_on_lmcache_gpu_ops(py);
        let config = Config {
            namespace: namespace.to_string(),
            socket_path: socket_path.map(PathBuf::from).unwrap_or_else(default_socket_path),
            tier: tier.to_lowercase(),
            num_workers,
            timeout: Duration::from_secs_f64(timeout_s),
        };
        config.validate().map_err(PyValueError::new_err)?;
        Ok(HyperCacheNativeConnector {
            inner: Connector::new(config).map_err(runtime_error)?,
        })
    }

    fn event_fd(&self) -> i32 {
        self.inner.event_fd()
    }

    fn submit_batch_set(&self, keys: Vec<String>, memviews: &Bound<'_, PyAny>) -> PyResult<u64> {
        let (held, raws) = acquire_all(memviews, false)?;
        // The held buffers travel with the batch and come back through drain_completions
        unsafe { self.inner.submit_set(keys, raws, Box::new(held)) }.map_err(runtime_error)
    }

    fn submit_batch_get(&self, keys: Vec<String>, memviews: &Bound<'_, PyAny>) -> PyResult<u64> {
        let (held, raws) = acquire_all(memviews, true)?;
        unsafe { self.inner.submit_get(keys, raws, Box::new(held)) }.map_err(runtime_error)
    }

    fn submit_batch_exists(&self, keys: Vec<String>) -> PyResult<u64> {
        self.inner.submit_exists(keys).map_err(runtime_error)
    }

    fn submit_batch_delete(&self, keys: Vec<String>) -> PyResult<u64> {
        self.inner.submit_delete(keys).map_err(runtime_error)
    }

    /// Returns a list of `(future_id, ok, error, per-key results)` tuples.
    fn drain_completions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(py);
        for completion in py.allow_threads(|| self.inner.drain_completions()) {
            let results = PyList::new(py, &completion.results)?;
            result.append((completion.future_id, completion.ok, &completion.error, results))?;
            // Dropping the completion here releases its buffers, with the interpreter lock held
        }
        Ok(result)
    }

    fn close(&self, py: Python<'_>) {
        py.allow_threads(|| self.inner.close());
    }

    /// The HyperCache URI for one of LMCache's key strings.
    fn uri(&self, key: &str) -> PyResult<String> {
        self.inner.uri(key).map_err(PyValueError::new_err)
    }

    /// Failure and I/O path counters. LMCache's wrapper doesn't surface these in report_status.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counters = self.inner.counters();
        let io = self.inner.io_stats();
        let stats = PyDict::new(py);
        for (name, counter) in [
            ("checksum_failures", &counters.checksum_failures),
            ("load_failures", &counters.load_failures),
            ("store_failures", &counters.store_failures),
            ("direct_read_bytes", &io.direct_read_bytes),
            ("direct_write_bytes", &io.direct_write_bytes),
            ("bounced_read_bytes", &io.bounced_read_bytes),
            ("bounced_write_bytes", &io.bounced_write_bytes),
            ("buffered_read_bytes", &io.buffered_read_bytes),
            ("buffered_write_bytes", &io.buffered_write_bytes),
        ] {
            stats.set_item(name, counter.load(Relaxed).into_py_any(py)?)?;
        }
        Ok(stats)
    }
}

#[pymodule]
fn hypercache_lmcache_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HyperCacheNativeConnector>()?;
    module.add_function(wrap_pyfunction!(check_lmcache_gpu_ops, module)?)?;
    module.add_function(wrap_pyfunction!(check_lmcache_version, module)?)?;
    module.add(
        "TESTED_LMCACHE_VERSIONS",
        PyTuple::new(module.py(), TESTED_LMCACHE_VERSIONS)?,
    )?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
