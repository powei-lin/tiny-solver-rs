use std::time::{Duration, Instant};

use crate::optimizer;
use std::ops::Mul;
extern crate nalgebra as na;
use faer::solvers::{SpSolver, SpSolverLstsq};
use faer_ext::IntoNalgebra;
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
            let hessian = jac
                .as_ref()
                .transpose()
                .to_col_major()
                .unwrap()
                .mul(jac.as_ref());
            let b = jac.into_transpose().mul(-residuals);
            let dx = hessian.sp_cholesky(faer::Side::Lower).unwrap().solve(b);
            // let qr = jac.sp_qr().unwrap();
            // let dx = qr.solve_lstsq(-residuals);
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
