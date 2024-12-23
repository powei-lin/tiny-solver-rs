use log::trace;
use nalgebra::zero;
use std::convert::identity;
use std::{collections::HashMap, time::Instant};

use faer_ext::IntoNalgebra;

use crate::common::OptimizerOptions;
use crate::linear;
use crate::optimizer;
use crate::sparse::LinearSolverType;
use crate::sparse::SparseLinearSolver;

#[derive(Debug)]
pub struct LevenbergMarquardtOptimizer {}
impl LevenbergMarquardtOptimizer {
    pub fn new() -> Self {
        Self {}
    }
}
impl Default for LevenbergMarquardtOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl optimizer::Optimizer for LevenbergMarquardtOptimizer {
    fn optimize(
        &self,
        problem: &crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> Option<HashMap<String, nalgebra::DVector<f64>>> {
        let mut params = initial_values.clone();
        let opt_option = optimizer_option.unwrap_or_default();
        let mut linear_solver: Box<dyn SparseLinearSolver> = match opt_option.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(linear::SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(linear::SparseQRSolver::new()),
        };

        let mut jacobi_scaling : Option<faer::sparse::SparseColMat> = None;
        let g_new = nalgebra::DMatrix::<f64>::identity(problem.total_residual_dimension, problem.total_variable_dimension);


        let mut last_err: f64 = 1.0;

        for i in 0..opt_option.max_iteration {
            let (residuals, jac) = problem.compute_residual_and_jacobian(&params);

            if i == 0 {
                let (rows, cols) = jac.shape();
                let jacobi_scaling_vec: Vec<(usize, usize, f64)> = (0..cols).map(|c| {
                    let v = 
                    jac.values_of_col(c)
                        .iter()
                        .map(|&i| i * i)
                        .sum::<f64>()
                        .sqrt();
                    (c, c, 1.0 / (1.0 + v))
                }).collect();
    
                jacobi_scaling = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                    rows,
                    cols,
                    &jacobi_scaling_vec,
                );
            }

            let current_error = residuals.norm_l2();
            trace!("iter:{} total err:{}", i, current_error);

            jac = jac * jacobi_scaling.unwrap();
            let jtj = jac.transpose().to_col_major().unwrap() * jac;
            let g = jac.transpose() * residuals;

            let u = 1.0 / 1e4;
            let v =  2;

            let jtj_regularized = jtj;



            if current_error < opt_option.min_error_threshold {
                trace!("error too low");
                break;
            } else if current_error.is_nan() {
                log::debug!("solve ax=b failed, current error is nan");
                return None;
            }
            if i > 0 {
                if (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold {
                    trace!("absolute error decreas low");
                    break;
                } else if (last_err - current_error).abs() / last_err
                    < opt_option.min_rel_error_decrease_threshold
                {
                    trace!("reletive error decrease low");
                    break;
                }
            }
            last_err = current_error;

            let start = Instant::now();
            if let Some(dx) = linear_solver.solve(&residuals, &jac) {
                let duration = start.elapsed();
                trace!("Time elapsed in solve Ax=b is: {:?}", duration);

                let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
                self.apply_dx(
                    &dx_na,
                    &mut params,
                    &problem.variable_name_to_col_idx_dict,
                    &problem.fixed_variable_indexes,
                    &problem.variable_bounds,
                );
            } else {
                log::debug!("solve ax=b failed");
                return None;
            }
        }
        Some(params)
    }
}
