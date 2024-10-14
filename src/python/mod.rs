use pyo3::prelude::*;

mod py_factors;
mod py_linear;
mod py_loss_functions;
mod py_optimizer;
mod py_problem;
use self::py_factors::*;
use self::py_linear::*;
use self::py_loss_functions::*;
use self::py_optimizer::*;
use self::py_problem::*;

fn register_child_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // For factors submodule
    let factors_module = PyModule::new_bound(parent_module.py(), "factors")?;
    factors_module.add_class::<PyBetweenFactorSE2>()?;
    factors_module.add_class::<PyPriorFactor>()?;
    factors_module.add_class::<PyFactor>()?;
    parent_module.add_submodule(&factors_module)?;
    parent_module
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.factors", factors_module)?;

    let loss_functions_module = PyModule::new_bound(parent_module.py(), "loss_functions")?;
    loss_functions_module.add_class::<PyHuberLoss>()?;
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
    m.add_class::<PyProblem>()?;
    m.add_class::<PyLinearSolver>()?;
    m.add_class::<PyOptimizerOptions>()?;
    m.add_class::<PyGaussNewtonOptimizer>()?;
    register_child_module(m)?;

    Ok(())
}
