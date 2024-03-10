use crate::optimizer;
use std::ops::Mul;
extern crate nalgebra as na;
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
            println!("residual{}, jac{}", residuals, jac);
            let b = jac.transpose().mul(-residuals);
            let hessian = jac.transpose().mul(jac);
            // let dx = hessian.qr().solve(&b).expect("msg");
            let dx = hessian.lu().solve(&b).expect("msg");
            // let dx = na::linalg::SVD::new(hessian).solve(&b).unwrap();
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
