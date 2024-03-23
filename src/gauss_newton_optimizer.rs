use log::{info, trace, warn};
use std::time::Instant;

use faer::sparse::linalg::solvers;
use faer_ext::IntoNalgebra;
use pyo3::prelude::*;

use crate::{linear::sparse_cholesky, optimizer, OptimizerOptions};

#[pyclass]
#[derive(Debug, Clone)]
pub struct GaussNewtonOptimizer {}

impl optimizer::Optimizer for GaussNewtonOptimizer {
    fn optimize(
        &self,
        problem: &crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> std::collections::HashMap<String, nalgebra::DVector<f64>> {
        let mut params = initial_values.clone();
        let opt_option = optimizer_option.unwrap_or_default();

        let mut last_err: f64 = 1.0;
        let mut symbolic_pattern: Option<solvers::SymbolicCholesky<usize>> = None;

        for i in 0..opt_option.max_iteration {
            let (residuals, jac) = problem.compute_residual_and_jacobian(&params);
            let current_error = residuals.norm_l2();
            trace!("iter:{} total err:{}", i, current_error);

            if current_error < opt_option.min_error_threshold {
                trace!("error too low");
                break;
            }
            if i > 0 {
                if last_err - current_error < opt_option.min_abs_error_decrease_threshold {
                    trace!("absolute error decreas low");
                    break;
                } else if (last_err - current_error) / last_err
                    < opt_option.min_rel_error_decrease_threshold
                {
                    trace!("reletive error decrease low");
                    break;
                }
            }
            last_err = current_error;

            let start = Instant::now();
            let dx = sparse_cholesky(&residuals, &jac, &mut symbolic_pattern);
            let duration = start.elapsed();
            trace!("Time elapsed in solve Ax=b is: {:?}", duration);

            let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
            self.apply_dx(&dx_na, &mut params, &problem.variable_name_to_col_idx_dict);
        }
        params
    }
}
