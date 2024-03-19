use pyo3::prelude::*;

use crate::factors::*;
use crate::loss_functions::*;
use crate::*;

mod py_factors;
mod py_loss_functions;
mod py_optimizer;
mod py_problem;
use self::py_factors::*;

fn register_child_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    // For factors submodule
    let factors_module = PyModule::new(py, "factors")?;
    factors_module.add_class::<BetweenFactorSE2>()?;
    factors_module.add_class::<PriorFactor>()?;
    parent_module.add_submodule(factors_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.factors", factors_module)?;

    let loss_functions_module = PyModule::new(py, "loss_functions")?;
    loss_functions_module.add_class::<HuberLoss>()?;
    parent_module.add_submodule(loss_functions_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.loss_functions", loss_functions_module)?;
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
    m.add_class::<PyDualDVec64>()?;
    register_child_module(_py, m)?;

    Ok(())
}
