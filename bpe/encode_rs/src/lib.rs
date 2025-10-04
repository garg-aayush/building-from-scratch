// Declare the bpe_encode module (loads src/bpe_encode.rs)
mod bpe_encode;

// Re-export types so users can import them directly from the crate root
pub use bpe_encode::{AllowedSpecialTokens, BpeEncode};

// Only compile Python bindings when building with --features python
#[cfg(feature = "python")]
use pyo3::prelude::*;

// Python wrapper class - visible to Python as a class
#[cfg(feature = "python")]
#[pyclass]
struct PyBpeEncode {
    inner: BpeEncode,  // Wraps the actual Rust implementation
}

// Methods that Python can call on PyBpeEncode instances
#[cfg(feature = "python")]
#[pymethods]
impl PyBpeEncode {
    // Constructor: called from Python as PyBpeEncode()
    #[new]
    fn new() -> Self {
        PyBpeEncode {
            inner: BpeEncode::new(),  // Create the inner Rust encoder
        }
    }

    // Load a BPE model file - converts Rust errors to Python RuntimeError
    fn load(&mut self, model_file: &str) -> PyResult<()> {
        self.inner
            .load(model_file)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    // Encode text to token IDs - returns Python list[int]
    fn encode(&self, text: &str, allowed_special_tokens: &str) -> PyResult<Vec<u32>> {
        // Convert Python string ("none"/"all"/"none_raise") to Rust enum
        let policy = AllowedSpecialTokens::from_str(allowed_special_tokens)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        // Call inner encoder and convert errors to Python exceptions
        self.inner
            .encode(text, policy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    // Decode token IDs back to text - Python list[int] -> str
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

// Define the Python module - the function name becomes the module name
#[cfg(feature = "python")]
#[pymodule]
fn bpe_encode_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBpeEncode>()?;  // Register PyBpeEncode class in the module
    Ok(())
}

