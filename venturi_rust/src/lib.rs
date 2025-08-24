use pyo3::prelude::*;

#[pyfunction]
fn adjust_batch(current_batch: i32, last_latency: f64, target_latency: f64, min_batch: i32, max_batch: i32, gamma: f64) -> i32 {
    if last_latency < target_latency && current_batch < max_batch {
        return std::cmp::min(max_batch, (current_batch as f64 * gamma) as i32);
    } else if last_latency > target_latency && current_batch > min_batch {
        return std::cmp::max(min_batch, (current_batch as f64 / gamma) as i32);
    } else {
        return current_batch;
    }
}

#[pymodule]
fn venturi_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adjust_batch, m)?)?;
    Ok(())
}
