use pyo3::prelude::*;

use crate::factors::*;
use crate::*;

mod py_factors;
mod py_optimizer;
mod py_problem;

/// For factors submodule
fn register_child_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "factors")?;
    child_module.add_class::<BetweenFactorSE2>()?;
    child_module.add_class::<PriorFactor>()?;
    parent_module.add_submodule(child_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.factors", child_module)?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn tiny_solver<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Problem>()?;
    m.add_class::<LinearSolver>()?;
    m.add_class::<OptimizerOptions>()?;
    m.add_class::<GaussNewtonOptimizer>()?;
    register_child_module(_py, m)?;

    Ok(())
}
