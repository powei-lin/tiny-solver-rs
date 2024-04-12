use pyo3::prelude::*;

use crate::factors::*;
use crate::loss_functions::*;
use crate::*;

mod py_factors;
mod py_loss_functions;
mod py_optimizer;
mod py_problem;
use self::py_factors::*;

fn register_child_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // For factors submodule
    let factors_module = PyModule::new_bound(parent_module.py(), "factors")?;
    factors_module.add_class::<BetweenFactorSE2>()?;
    factors_module.add_class::<PriorFactor>()?;
    factors_module.add_class::<PyFactor>()?;
    parent_module.add_submodule(&factors_module)?;
    parent_module
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.factors", factors_module)?;

    let loss_functions_module = PyModule::new_bound(parent_module.py(), "loss_functions")?;
    loss_functions_module.add_class::<HuberLoss>()?;
    parent_module.add_submodule(&loss_functions_module)?;
    parent_module
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.loss_functions", loss_functions_module)?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn tiny_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // pyo3_log::init();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Problem>()?;
    m.add_class::<LinearSolver>()?;
    m.add_class::<OptimizerOptions>()?;
    m.add_class::<GaussNewtonOptimizer>()?;
    register_child_module(m)?;

    Ok(())
}
