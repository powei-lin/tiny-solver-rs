#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;
    use tiny_solver::Optimizer;

    /// r(x) = atan(x). From x0 = 2 the undamped (Gauss-Newton) step overshoots
    /// to x ~ -3.5, where |r| is larger, so LM has to reject it and retry with
    /// more damping.
    struct AtanFactor;
    impl<T: na::RealField> tiny_solver::factors::Factor<T> for AtanFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            na::dvector![params[0][0].clone().atan()]
        }
    }

    #[test]
    fn levenberg_marquardt_retries_after_rejected_step() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(1, &["x"], Box::new(AtanFactor), None);
        let initial_values = HashMap::from([("x".to_string(), na::dvector![2.0])]);

        let result = tiny_solver::LevenbergMarquardtOptimizer::default()
            .optimize(&problem, &initial_values, None)
            .unwrap();

        assert!(result["x"][0].abs() < 1e-2, "x = {}", result["x"][0]);
    }
}
