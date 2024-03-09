use crate::problem;
use std::collections::HashMap;
extern crate nalgebra as na;
pub trait Optimizer {
    fn optimize(
        self,
        problem: problem::Problem,
        initial_values: &HashMap<String, na::DVector<f64>>,
    ) -> HashMap<String, na::DVector<f64>>;
}
