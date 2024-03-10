use std::time::{Duration, Instant};

use crate::optimizer;
use std::ops::Mul;
extern crate nalgebra as na;
use nalgebra_sparse::factorization::CscCholesky;
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
            let b = jac.transpose().mul(-residuals);
            let hessian = jac.transpose().mul(jac);
            println!("matrix size {}x{}", hessian.nrows(), hessian.ncols());
            let start = Instant::now();
            let cholesky = CscCholesky::factor(&hessian).unwrap();
            let dx = na::DVector::from(cholesky.solve(&b).fixed_columns(0));
            let duration = start.elapsed();
            println!("Time elapsed in solve() is: {:?}", duration);

            if dx.norm() < 1e-16 {
                println!("grad too low");
                break;
            }
            self.apply_dx(&dx, &mut params, &problem.variable_name_to_col_idx_dict);
            // params += dx
            // problem.write_back_variables(params)
        }
        return params;
    }
}
