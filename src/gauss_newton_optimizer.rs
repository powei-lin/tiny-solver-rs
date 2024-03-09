use crate::optimizer;
pub struct GaussNewtonOptimizer {}
impl optimizer::Optimizer for GaussNewtonOptimizer {
    fn optimize(
        self,
        problem: crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
    ) -> std::collections::HashMap<String, nalgebra::DVector<f64>> {
        problem.combine_variables(&initial_values);
        return initial_values.clone();
    }
}
