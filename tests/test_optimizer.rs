#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// r(x) = x^2 + 1, counting its evaluations. At x = 0 the gradient is zero
    /// but the cost is not, so every LM step is zero.
    struct CountingBowlFactor {
        evaluations: Arc<AtomicUsize>,
    }
    impl<T: na::RealField> tiny_solver::factors::Factor<T> for CountingBowlFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            let x = params[0][0].clone();
            na::dvector![x.clone() * x + T::one()]
        }
    }

    #[test]
    fn levenberg_marquardt_stops_at_a_stationary_point() {
        let evaluations = Arc::new(AtomicUsize::new(0));
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(CountingBowlFactor {
                evaluations: evaluations.clone(),
            }),
            None,
        );
        let initial_values = HashMap::from([("x".to_string(), na::dvector![0.0])]);

        let result = tiny_solver::LevenbergMarquardtOptimizer::default()
            .optimize(&problem, &initial_values, None)
            .unwrap();

        assert_eq!(result["x"][0], 0.0);
        // Running to max_iteration (100) would take hundreds of evaluations.
        let evaluations = evaluations.load(Ordering::Relaxed);
        assert!(evaluations < 10, "{evaluations} residual evaluations");
    }
}
