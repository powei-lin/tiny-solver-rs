use log::trace;
use std::ops::Mul;
use std::{collections::HashMap, time::Instant};

use faer_ext::IntoNalgebra;

use crate::common::OptimizerOptions;
use crate::linear;
use crate::optimizer;
use crate::parameter_block::ParameterBlock;
use crate::sparse::LinearSolverType;
use crate::sparse::SparseLinearSolver;

const DEFAULT_MIN_DIAGONAL: f64 = 1e-6;
const DEFAULT_MAX_DIAGONAL: f64 = 1e32;
const DEFAULT_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;

#[derive(Debug)]
pub struct LevenbergMarquardtOptimizer {
    min_diagonal: f64,
    max_diagonal: f64,
    initial_trust_region_radius: f64,
}

impl LevenbergMarquardtOptimizer {
    pub fn new(min_diagonal: f64, max_diagonal: f64, initial_trust_region_radius: f64) -> Self {
        Self {
            min_diagonal,
            max_diagonal,
            initial_trust_region_radius,
        }
    }
}

impl Default for LevenbergMarquardtOptimizer {
    fn default() -> Self {
        Self {
            min_diagonal: DEFAULT_MIN_DIAGONAL,
            max_diagonal: DEFAULT_MAX_DIAGONAL,
            initial_trust_region_radius: DEFAULT_INITIAL_TRUST_REGION_RADIUS,
        }
    }
}

impl optimizer::Optimizer for LevenbergMarquardtOptimizer {
    fn optimize(
        &self,
        problem: &crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> Option<HashMap<String, nalgebra::DVector<f64>>> {
        let mut parameter_blocks: HashMap<String, ParameterBlock> =
            problem.initialize_parameter_blocks(initial_values);

        let variable_name_to_col_idx_dict =
            problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
        let total_variable_dimension = parameter_blocks.values().map(|p| p.tangent_size()).sum();

        let opt_option = optimizer_option.unwrap_or_default();
        let mut linear_solver: Box<dyn SparseLinearSolver> = match opt_option.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(linear::SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(linear::SparseQRSolver::new()),
        };

        // On the first iteration, we'll generate a diagonal matrix of the jacobian.
        // Its shape will be (total_variable_dimension, total_variable_dimension).
        // With LM, rather than solving A * dx = b for dx, we solve for (A + lambda * diag(A)) dx = b.
        let mut jacobi_scaling_diagonal: Option<faer::sparse::SparseColMat<usize, f64>> = None;

        let s = problem.build_symbolic_structure(
            &parameter_blocks,
            total_variable_dimension,
            &variable_name_to_col_idx_dict,
        );

        // Damping parameter (a.k.a lambda / Marquardt parameter)
        let mut u = 1.0 / self.initial_trust_region_radius;

        let mut last_err: f64 = 1.0;
        for i in 0..opt_option.max_iteration {
            let (residuals, mut jac) = problem.compute_residual_and_jacobian(
                &parameter_blocks,
                &variable_name_to_col_idx_dict,
                total_variable_dimension,
                &s,
            );

            if i == 0 {
                // On the first iteration, generate the diagonal of the jacobian.
                let cols = jac.shape().1;
                let jacobi_scaling_vec: Vec<(usize, usize, f64)> = (0..cols)
                    .map(|c| {
                        let v = jac
                            .values_of_col(c)
                            .iter()
                            .map(|&i| i * i)
                            .sum::<f64>()
                            .sqrt();
                        (c, c, 1.0 / (1.0 + v))
                    })
                    .collect();

                jacobi_scaling_diagonal = Some(
                    faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                        cols,
                        cols,
                        &jacobi_scaling_vec,
                    )
                    .unwrap(),
                );
            }

            let current_error = residuals.norm_l2();
            trace!("iter:{} total err:{}", i, current_error);

            if current_error < opt_option.min_error_threshold {
                trace!("error too low");
                break;
            } else if current_error.is_nan() {
                log::debug!("solve ax=b failed, current error is nan");
                return None;
            }
            if i > 0 {
                if (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold {
                    trace!("absolute error decrease low");
                    break;
                } else if (last_err - current_error).abs() / last_err
                    < opt_option.min_rel_error_decrease_threshold
                {
                    trace!("relative error decrease low");
                    break;
                }
            }
            last_err = current_error;

            // Scale the current jacobian by the diagonal matrix
            jac = jac * jacobi_scaling_diagonal.as_ref().unwrap();

            // J^T * J = Matrix of shape (total_variable_dimension, total_variable_dimension)
            let jtj = jac
                .as_ref()
                .transpose()
                .to_col_major()
                .unwrap()
                .mul(jac.as_ref());

            // J^T * -r = Matrix of shape (total_variable_dimension, 1)
            let jtr = jac.as_ref().transpose().mul(-&residuals);

            // Regularize the diagonal of jtj between the min and max diagonal values.
            let mut jtj_regularized = jtj.clone();
            for i in 0..total_variable_dimension {
                jtj_regularized[(i, i)] +=
                    u * (jtj[(i, i)].max(self.min_diagonal)).min(self.max_diagonal);
            }

            let start = Instant::now();
            if let Some(lm_step) = linear_solver.solve_jtj(&jtr, &jtj_regularized) {
                let duration = start.elapsed();
                let dx = jacobi_scaling_diagonal.as_ref().unwrap() * &lm_step;

                trace!("Time elapsed in solve Ax=b is: {:?}", duration);

                let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();

                let mut new_param_blocks = parameter_blocks.clone();

                self.apply_dx2(
                    &dx_na,
                    &mut new_param_blocks,
                    &variable_name_to_col_idx_dict,
                );

                // Compute residuals of (x + dx)
                let new_residuals = problem.compute_residuals(&new_param_blocks, true);

                // rho is the ratio between the actual reduction in error and the reduction
                // in error if the problem were linear.
                let actual_residual_change =
                    residuals.squared_norm_l2() - new_residuals.squared_norm_l2();
                trace!("actual_residual_change {}", actual_residual_change);
                let linear_residual_change: faer::Mat<f64> =
                    lm_step.transpose().mul(2.0 * &jtr - &jtj * &lm_step);
                let rho = actual_residual_change / linear_residual_change[(0, 0)];

                if rho > 0.0 {
                    // The linear model appears to be fitting, so accept (x + dx) as the new x.
                    parameter_blocks = new_param_blocks;

                    // Increase the trust region by reducing u
                    let tmp = 2.0 * rho - 1.0;
                    u *= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
                } else {
                    // If there's too much divergence, reduce the trust region and try again with the same parameters.
                    u *= 2.0;
                    trace!("u {}", u);
                }
            } else {
                log::debug!("solve ax=b failed");
                return None;
            }
        }
        let params = parameter_blocks
            .iter()
            .map(|(k, v)| (k.to_owned(), v.params.clone()))
            .collect();
        Some(params)
    }
}
