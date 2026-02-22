use pyo3::prelude::*;

mod measurement;
mod simulator;

pub use simulator::RustLQubitSimulator;

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<simulator::RustLQubitSimulator>()?;
    Ok(())
}
