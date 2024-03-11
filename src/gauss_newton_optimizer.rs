use std::ops::Mul;
use std::time::{Duration, Instant};

use faer::solvers::{SpSolver, SpSolverLstsq};
use faer_ext::IntoNalgebra;
use nalgebra as na;

use crate::{linear::sparse_cholesky, optimizer};
pub struct GaussNewtonOptimizer {}
impl optimizer::Optimizer for GaussNewtonOptimizer {
    fn optimize(
        &self,
        problem: crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
    ) -> std::collections::HashMap<String, nalgebra::DVector<f64>> {
        let mut params = initial_values.clone();

        for i in 0..10 {
            println!("{}", i);

            let (residuals, jac) = problem.compute_residual_and_jacobian(&params);
            let start = Instant::now();
            let dx = sparse_cholesky(&residuals, &jac);
            let duration = start.elapsed();
            println!("Time elapsed in solve() is: {:?}", duration);

            if dx.norm_l1() < 1e-16 {
                println!("grad too low");
                break;
            }
            let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
            self.apply_dx(&dx_na, &mut params, &problem.variable_name_to_col_idx_dict);
        }
        return params;
    }
}
