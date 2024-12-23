use log::trace;
use nalgebra::zero;
use std::convert::identity;
use std::{collections::HashMap, time::Instant};
use std::ops::Mul;

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

        let mut jacobi_scaling_diagonal : Option<faer::sparse::SparseColMat<usize, f64>> = None;
        let mut cost = 0.0;
        let g_new = nalgebra::DMatrix::<f64>::identity(problem.total_residual_dimension, problem.total_variable_dimension);

        const MIN_DIAGONAL :f64 = 1e-6;
        const MAX_DIAGONAL : f64 = 1e32;
        let u = 1.0 / 1e4;
        let v =  2;

        let mut last_err: f64 = 1.0;

        for i in 0..opt_option.max_iteration {
            let (residuals, mut jac) = problem.compute_residual_and_jacobian(&params);

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
    
                jacobi_scaling_diagonal = Some(faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                    cols,
                    cols,
                    &jacobi_scaling_vec,
                ).unwrap());
            }

            let current_error = residuals.norm_l2();
            trace!("iter:{} total err:{}", i, current_error);

            println!("{}, {}", problem.total_residual_dimension, problem.total_variable_dimension);
            println!("Residual: ({:?}), Jacobi scaling diagonal ({:?}) vs jacobian ({:?})", residuals.shape(), jacobi_scaling_diagonal.as_ref().unwrap().shape(), jac.shape());
            jac = jac * jacobi_scaling_diagonal.as_ref().unwrap();
            cost = residuals.squared_norm_l2() / 2.0;

            
            let jtj = jac
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jac.as_ref());
            
            let jtr = jac.as_ref().transpose().mul(-residuals);

   
            let mut jtj_regularized = jtj.clone();
            println!("jac: {:?} jtj: {:?}, jacobi_scaling: {:?}", jac.shape(), jtj.shape(), jacobi_scaling_diagonal.as_ref().unwrap().shape());
            for i in 0..problem.total_variable_dimension {
                jtj_regularized[(i, i)] = (jtj.get(i,i).unwrap().max(MIN_DIAGONAL)).min(MAX_DIAGONAL);
            }

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
            if let Some(lm_step) = linear_solver.solve_jtj(&jtr, &jtj_regularized) {
                println!("YO DAWG: dx: {:?}, jtr: {:?}, jtj_reg: {:?}", lm_step.shape(), jtr.shape(), jtj_regularized.shape());
                let duration = start.elapsed();
                let dx = jacobi_scaling_diagonal.as_ref().unwrap() * lm_step;

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
