use std::ops::Mul;
use std::time::Instant;

use faer::prelude::SpSolver;
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

        // let (_, jac) = problem.compute_residual_and_jacobian(&params);
        // let hessian = jac
        //     .as_ref()
        //     .transpose()
        //     .to_col_major()
        //     .unwrap()
        //     .mul(jac.as_ref());
        // let symbolic = solvers::SymbolicCholesky::try_new(hessian.symbolic(), faer::Side::Lower).unwrap();
        let mut symbolic: Option<solvers::SymbolicCholesky<usize>> = None;

        for i in 0..opt_option.max_iteration {
            let (residuals, jac) = problem.compute_residual_and_jacobian(&params);
            let current_error = residuals.norm_l2();
            println!("iter:{} total err:{}", i, current_error);
            if current_error < opt_option.min_error_threshold {
                println!("error too low");
                break;
            }
            if i > 0 {
                if last_err - current_error < opt_option.min_abs_error_decrease_threshold {
                    println!("abs low");
                    break;
                } else if (last_err - current_error) / last_err
                    < opt_option.min_rel_error_decrease_threshold
                {
                    println!("rel low");
                    break;
                }
            }
            last_err = current_error;

            let start = Instant::now();
            let hessian = jac
                .as_ref()
                .transpose()
                .to_col_major()
                .unwrap()
                .mul(jac.as_ref());
            let b = jac.as_ref().transpose().mul(-residuals);
            let sym = if symbolic.is_some() {
                symbolic.as_ref().unwrap()
            } else {
                symbolic = Some(
                    solvers::SymbolicCholesky::try_new(hessian.symbolic(), faer::Side::Lower)
                        .unwrap(),
                );
                symbolic.as_ref().unwrap()
            };
            let dx = solvers::Cholesky::try_new_with_symbolic(
                sym.clone(),
                hessian.as_ref(),
                faer::Side::Lower,
            )
            .unwrap()
            .solve(b);
            // let dx = hessian.sp_cholesky(faer::Side::Lower).unwrap().solve(b);
            // let dx = sparse_cholesky(&residuals, &jac);
            let duration = start.elapsed();
            println!("Time elapsed in solve() is: {:?}", duration);

            // if dx.norm_l1() < opt_option.gradient_threshold {
            //     println!("grad too low");
            //     break;
            // }
            let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
            self.apply_dx(&dx_na, &mut params, &problem.variable_name_to_col_idx_dict);
        }
        params
    }
}
