use pyo3::prelude::*;
use std::time::Instant;

use faer_ext::IntoNalgebra;

use crate::{linear::sparse_cholesky, optimizer};

#[pyclass]
#[derive(Debug, Clone)]
pub struct GaussNewtonOptimizer {}

impl optimizer::Optimizer for GaussNewtonOptimizer {
    fn optimize(
        &self,
        problem: crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
    ) -> std::collections::HashMap<String, nalgebra::DVector<f64>> {
        let mut params = initial_values.clone();

        for i in 0..50 {
            println!("{}", i);

            let (residuals, jac) = problem.compute_residual_and_jacobian(&params);
            let start = Instant::now();
            let dx = sparse_cholesky(&residuals, &jac);
            let duration = start.elapsed();
            println!("Time elapsed in solve() is: {:?}", duration);

            if dx.norm_l1() < 1e-8 {
                println!("grad too low");
                break;
            }
            let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
            self.apply_dx(&dx_na, &mut params, &problem.variable_name_to_col_idx_dict);
        }
        params
    }
}
