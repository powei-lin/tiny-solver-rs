use crate::optimizer;
pub struct GaussNewtonOptimizer {}
impl optimizer::Optimizer for GaussNewtonOptimizer {
    fn optimize(
        self,
        problem: crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
    ) -> std::collections::HashMap<String, nalgebra::DVector<f64>> {
        // let mut gcombined_variables = problem.combine_variables(&initial_values);
        for i in 0..1 {
            println!("{}", i);

            let (residuals, jac) = problem.compute_residual_and_jacobian(&initial_values);
            println!("residual{}, jac{}", residuals, jac);
            // hessian = jac.T @ jac
            // b = -jac.T @ residuals
            // dx = spsolve(hessian, b)
            // if np.linalg.norm(dx) < 1e-16:
            //     break
            // params += dx
            // problem.write_back_variables(params)
        }
        return initial_values.clone();
    }
}
