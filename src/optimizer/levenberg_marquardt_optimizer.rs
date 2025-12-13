// use log::trace;
// use std::ops::Mul;
// use std::{collections::HashMap, time::Instant};

// use faer::sparse::Triplet;
// use faer_ext::IntoNalgebra;

// use crate::common::OptimizerOptions;
// use crate::linear;
// use crate::optimizer;
// use crate::parameter_block::ParameterBlock;
// use crate::sparse::LinearSolverType;
// use crate::sparse::SparseLinearSolver;

// const DEFAULT_MIN_DIAGONAL: f64 = 1e-6;
// const DEFAULT_MAX_DIAGONAL: f64 = 1e32;
// const DEFAULT_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;

// #[derive(Debug)]
// pub struct LevenbergMarquardtOptimizer {
//     min_diagonal: f64,
//     max_diagonal: f64,
//     initial_trust_region_radius: f64,
// }

// impl LevenbergMarquardtOptimizer {
//     pub fn new(min_diagonal: f64, max_diagonal: f64, initial_trust_region_radius: f64) -> Self {
//         Self {
//             min_diagonal,
//             max_diagonal,
//             initial_trust_region_radius,
//         }
//     }
// }

// impl Default for LevenbergMarquardtOptimizer {
//     fn default() -> Self {
//         Self {
//             min_diagonal: DEFAULT_MIN_DIAGONAL,
//             max_diagonal: DEFAULT_MAX_DIAGONAL,
//             initial_trust_region_radius: DEFAULT_INITIAL_TRUST_REGION_RADIUS,
//         }
//     }
// }

// impl optimizer::Optimizer for LevenbergMarquardtOptimizer {
//     fn optimize(
//         &self,
//         problem: &crate::problem::Problem,
//         initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
//         optimizer_option: Option<OptimizerOptions>,
//     ) -> Option<HashMap<String, nalgebra::DVector<f64>>> {
//         let mut parameter_blocks: HashMap<String, ParameterBlock> =
//             problem.initialize_parameter_blocks(initial_values);

//         let variable_name_to_col_idx_dict =
//             problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
//         let total_variable_dimension = parameter_blocks
//             .values()
//             .map(|p| {
//                 if p.manifold.is_some() {
//                     p.tangent_size()
//                 } else {
//                     p.tangent_size() - p.fixed_variables.len()
//                 }
//             })
//             .sum();

//         let opt_option = optimizer_option.unwrap_or_default();
//         let mut linear_solver: Box<dyn SparseLinearSolver> = match opt_option.linear_solver_type {
//             LinearSolverType::SparseCholesky => Box::new(linear::SparseCholeskySolver::new()),
//             LinearSolverType::SparseQR => Box::new(linear::SparseQRSolver::new()),
//         };

//         // On the first iteration, we'll generate a diagonal matrix of the jacobian.
//         // Its shape will be (total_variable_dimension, total_variable_dimension).
//         // With LM, rather than solving A * dx = b for dx, we solve for (A + lambda * diag(A)) dx = b.
//         let mut jacobi_scaling_diagonal: Option<faer::sparse::SparseColMat<usize, f64>> = None;

//         let symbolic_structure = problem.build_symbolic_structure(
//             &parameter_blocks,
//             total_variable_dimension,
//             &variable_name_to_col_idx_dict,
//         );

//         // Damping parameter (a.k.a lambda / Marquardt parameter)
//         let mut u = 1.0 / self.initial_trust_region_radius;

//         let mut last_err;
//         let mut current_error = self.compute_error(problem, &parameter_blocks);
//         for i in 0..opt_option.max_iteration {
//             last_err = current_error;

//             let (residuals, mut jac) = problem.compute_residual_and_jacobian(
//                 &parameter_blocks,
//                 &variable_name_to_col_idx_dict,
//                 &symbolic_structure,
//             );

//             if i == 0 {
//                 // On the first iteration, generate the diagonal of the jacobian.
//                 let cols = jac.shape().1;
//                 let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
//                     .map(|c| {
//                         let v = jac.val_of_col(c).iter().map(|&i| i * i).sum::<f64>().sqrt();
//                         Triplet::new(c, c, 1.0 / (1.0 + v))
//                     })
//                     .collect();

//                 jacobi_scaling_diagonal = Some(
//                     faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
//                         cols,
//                         cols,
//                         &jacobi_scaling_vec,
//                     )
//                     .unwrap(),
//                 );
//             }

//             // Scale the current jacobian by the diagonal matrix
//             jac = jac * jacobi_scaling_diagonal.as_ref().unwrap();

//             // J^T * J = Matrix of shape (total_variable_dimension, total_variable_dimension)
//             let jtj = jac
//                 .as_ref()
//                 .transpose()
//                 .to_col_major()
//                 .unwrap()
//                 .mul(jac.as_ref());

//             // J^T * -r = Matrix of shape (total_variable_dimension, 1)
//             let jtr = jac.as_ref().transpose().mul(-&residuals);

//             // Regularize the diagonal of jtj between the min and max diagonal values.
//             let mut jtj_regularized = jtj.clone();
//             for i in 0..total_variable_dimension {
//                 jtj_regularized[(i, i)] +=
//                     u * (jtj[(i, i)].max(self.min_diagonal)).min(self.max_diagonal);
//             }

//             let start = Instant::now();
//             if let Some(lm_step) = linear_solver.solve_jtj(&jtr, &jtj_regularized) {
//                 let duration = start.elapsed();
//                 let dx = jacobi_scaling_diagonal.as_ref().unwrap() * &lm_step;

//                 trace!("Time elapsed in solve Ax=b is: {:?}", duration);

//                 let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();

//                 let mut new_param_blocks = parameter_blocks.clone();

//                 self.apply_dx2(
//                     &dx_na,
//                     &mut new_param_blocks,
//                     &variable_name_to_col_idx_dict,
//                 );

//                 // Compute residuals of (x + dx)
//                 let new_residuals = problem.compute_residuals(&new_param_blocks, true);

//                 // rho is the ratio between the actual reduction in error and the reduction
//                 // in error if the problem were linear.
//                 let actual_residual_change =
//                     residuals.as_ref().squared_norm_l2() - new_residuals.as_ref().squared_norm_l2();
//                 trace!("actual_residual_change {}", actual_residual_change);
//                 let linear_residual_change: faer::Mat<f64> =
//                     lm_step.transpose().mul(2.0 * &jtr - &jtj * &lm_step);
//                 let rho = actual_residual_change / linear_residual_change[(0, 0)];

//                 if rho > 0.0 {
//                     // The linear model appears to be fitting, so accept (x + dx) as the new x.
//                     parameter_blocks = new_param_blocks;

//                     // Increase the trust region by reducing u
//                     let tmp = 2.0 * rho - 1.0;
//                     u *= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
//                 } else {
//                     // If there's too much divergence, reduce the trust region and try again with the same parameters.
//                     u *= 2.0;
//                     trace!("u {}", u);
//                 }
//             } else {
//                 log::debug!("solve ax=b failed");
//                 return None;
//             }

//             current_error = self.compute_error(problem, &parameter_blocks);
//             trace!("iter:{} total err:{}", i, current_error);

//             if current_error < opt_option.min_error_threshold {
//                 trace!("error too low");
//                 break;
//             } else if current_error.is_nan() {
//                 log::debug!("solve ax=b failed, current error is nan");
//                 return None;
//             }

//             if (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold {
//                 trace!("absolute error decrease low");
//                 break;
//             } else if (last_err - current_error).abs() / last_err
//                 < opt_option.min_rel_error_decrease_threshold
//             {
//                 trace!("relative error decrease low");
//                 break;
//             }
//         }
//         let params = parameter_blocks
//             .iter()
//             .map(|(k, v)| (k.to_owned(), v.params.clone()))
//             .collect();
//         Some(params)
//     }
// }

use log::trace;
use std::ops::Mul;
use std::{collections::HashMap, time::Instant};

use faer::sparse::Triplet;
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
        let total_variable_dimension = parameter_blocks
            .values()
            .map(|p| {
                if p.manifold.is_some() {
                    p.tangent_size()
                } else {
                    p.tangent_size() - p.fixed_variables.len()
                }
            })
            .sum();

        let opt_option = optimizer_option.unwrap_or_default();
        let mut linear_solver: Box<dyn SparseLinearSolver> = match opt_option.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(linear::SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(linear::SparseQRSolver::new()),
        };

        let symbolic_structure = problem.build_symbolic_structure(
            &parameter_blocks,
            total_variable_dimension,
            &variable_name_to_col_idx_dict,
        );

        // Damping parameter (lambda)
        let mut u = 1.0 / self.initial_trust_region_radius;
        // Factor to adjust u
        let mut v = 2.0;

        let mut current_error = self.compute_error(problem, &parameter_blocks);
        let mut last_err;

        // Flags to control computation
        let mut recompute_jacobian = true;
        let mut iter = 0;

        // Variables to cache between rejected steps (using Option because faer types don't impl Default)
        // let mut jac: Option<faer::sparse::SparseColMat<usize, f64>> = None;
        let mut residuals: Option<faer::Mat<f64>> = None;
        let mut jtj: Option<faer::sparse::SparseColMat<usize, f64>> = None;
        let mut jtr: Option<faer::Mat<f64>> = None;
        
        // Diagonal scaling matrix
        let mut jacobi_scaling_diagonal: Option<faer::sparse::SparseColMat<usize, f64>> = None;

        while iter < opt_option.max_iteration {
            last_err = current_error;

            // 1. Compute Jacobian and J^T*J ONLY if the previous step was accepted (or first run)
            if recompute_jacobian {
                let (res, raw_jac) = problem.compute_residual_and_jacobian(
                    &parameter_blocks,
                    &variable_name_to_col_idx_dict,
                    &symbolic_structure,
                );
                
                // Initialize scaling only once
                if jacobi_scaling_diagonal.is_none() {
                    let cols = raw_jac.shape().1;
                    let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
                        .map(|c| {
                            let val = raw_jac.val_of_col(c).iter().map(|&i| i * i).sum::<f64>().sqrt();
                            Triplet::new(c, c, 1.0 / (1.0 + val))
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

                // Apply scaling: J_scaled = J * D^-1
                let scaled_jac = raw_jac * jacobi_scaling_diagonal.as_ref().unwrap();

                // Compute J^T * J (Expensive Operation)
                let calculated_jtj = scaled_jac
                    .as_ref()
                    .transpose()
                    .to_col_major()
                    .unwrap()
                    .mul(scaled_jac.as_ref());

                // Compute J^T * -r
                let calculated_jtr = scaled_jac.as_ref().transpose().mul(-&res);
                
                // Update Cache
                // jac = Some(scaled_jac);
                residuals = Some(res);
                jtj = Some(calculated_jtj);
                jtr = Some(calculated_jtr);
            }

            // 2. Prepare the augmented equation: (J^T*J + u * D^T*D) dx = J^T*r
            // We clone jtj because we need to modify the diagonal, but we want to keep the original jtj for the next retry if this one fails.
            // Safe to unwrap because recompute_jacobian=true in the first iteration guarantees assignment.
            let mut jtj_regularized = jtj.as_ref().unwrap().clone();
            
            // Cache the damping values to speed up rho calculation later
            let mut damping_terms = vec![0.0; total_variable_dimension];

            for i in 0..total_variable_dimension {
                let diag_val = jtj_regularized[(i, i)];
                let damp = u * (diag_val.max(self.min_diagonal)).min(self.max_diagonal);
                jtj_regularized[(i, i)] += damp;
                damping_terms[i] = damp;
            }

            let start = Instant::now();
            let jtr_ref = jtr.as_ref().unwrap();
            
            if let Some(lm_step) = linear_solver.solve_jtj(jtr_ref, &jtj_regularized) {
                let duration = start.elapsed();
                trace!("Time elapsed in solve Ax=b is: {:?}", duration);

                // Unscale step: dx = D^-1 * lm_step
                let dx = jacobi_scaling_diagonal.as_ref().unwrap() * &lm_step;
                let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();

                // Create candidate parameters
                let mut new_param_blocks = parameter_blocks.clone();
                self.apply_dx2(
                    &dx_na,
                    &mut new_param_blocks,
                    &variable_name_to_col_idx_dict,
                );

                // Compute new residuals
                let new_residuals = problem.compute_residuals(&new_param_blocks, true);
                
                let actual_residual_change =
                    residuals.as_ref().unwrap().as_ref().squared_norm_l2() - new_residuals.as_ref().squared_norm_l2();

                // OPTIMIZED Rho Calculation:
                // Denom = h^T * (g + D_damp * h)
                let mut linear_residual_change = 0.0;
                for i in 0..total_variable_dimension {
                    let h_i = lm_step[(i, 0)];
                    let g_i = jtr_ref[(i, 0)];
                    let d_i = damping_terms[i]; 
                    linear_residual_change += h_i * (g_i + d_i * h_i);
                }
                
                let rho = actual_residual_change / linear_residual_change;

                if rho > 0.0 {
                    // ACCEPT STEP
                    parameter_blocks = new_param_blocks;
                    current_error = self.compute_error(problem, &parameter_blocks);
                    
                    // Update trust region
                    let tmp = 2.0 * rho - 1.0;
                    u *= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
                    v = 2.0;

                    trace!("iter:{} total err:{} accepted", iter, current_error);
                    
                    // Important: Next loop must recompute Jacobian for new params
                    recompute_jacobian = true; 
                    iter += 1; 

                    // Convergence Check
                    if current_error < opt_option.min_error_threshold {
                        break;
                    }
                    if (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold {
                        break;
                    }
                    if (last_err - current_error).abs() / last_err < opt_option.min_rel_error_decrease_threshold {
                        break;
                    }

                } else {
                    // REJECT STEP
                    u *= v;
                    v *= 2.0;
                    trace!("Step rejected, increasing u to {}", u);
                    
                    // Important: DO NOT recompute Jacobian
                    recompute_jacobian = false;
                    
                    iter += 1; 
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