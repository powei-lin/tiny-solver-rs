#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use nalgebra as na;
    use tiny_solver::Optimizer;
    use tiny_solver::loss_functions::{CauchyLoss, HuberLoss};

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

    /// A residual block with a loss function adds rho(||r||^2) to the cost,
    /// not the squared norm of its loss-corrected residuals.
    #[test]
    fn cost_is_the_sum_of_the_loss_functions() {
        let mut problem = tiny_solver::Problem::new();
        let prior = |v: f64| Box::new(tiny_solver::factors::PriorFactor { v: na::dvector![v] });
        // s = 0.25: inside the quadratic region of Huber(1), rho = s.
        problem.add_residual_block(1, &["x"], prior(0.5), Some(Box::new(HuberLoss::new(1.0))));
        // s = 9: rho = 2 * 1 * 3 - 1 = 5. The corrected residual gives 3.
        problem.add_residual_block(1, &["x"], prior(-2.0), Some(Box::new(HuberLoss::new(1.0))));
        // s = 4: rho = ln(1 + 4) = 1.609. The corrected residual gives 0.8.
        problem.add_residual_block(1, &["x"], prior(3.0), Some(Box::new(CauchyLoss::new(1.0))));
        // s = 1, no loss function: rho = s.
        problem.add_residual_block(1, &["x"], prior(2.0), None);
        let initial_values = HashMap::from([("x".to_string(), na::dvector![1.0])]);

        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let cost = problem.compute_cost(&parameter_blocks);

        let expected = 0.25 + 5.0 + 5.0_f64.ln() + 1.0;
        assert!(
            (cost - expected).abs() < 1e-12,
            "cost {cost}, expected {expected}"
        );
    }

    /// r_i = a * x_i + b - y_i
    struct LineFactor {
        x: f64,
        y: f64,
    }
    impl<T: na::RealField> tiny_solver::factors::Factor<T> for LineFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            let x = T::from_f64(self.x).unwrap();
            let y = T::from_f64(self.y).unwrap();
            na::dvector![params[0][0].clone() * x + params[0][1].clone() - y]
        }
    }

    /// A line fit with Huber loss is convex, so LM must end where the gradient
    /// of sum_i rho(r_i^2) vanishes. It used to accept or reject steps by the
    /// squared norm of the loss-corrected residuals, a different function, and
    /// stopped short of the minimum.
    #[test]
    fn levenberg_marquardt_minimizes_the_robust_cost() {
        let scale = 0.25;
        // y = 2x + 1 with deterministic noise, every fourth point an outlier.
        let points: Vec<(f64, f64)> = (0..12)
            .map(|i| {
                let x = i as f64 * 0.5;
                let noise = 0.5 * (1.7 * i as f64 + 0.3).sin();
                let outlier = if i % 4 == 0 { 10.0 } else { 0.0 };
                (x, 2.0 * x + 1.0 + noise + outlier)
            })
            .collect();
        let mut problem = tiny_solver::Problem::new();
        for &(x, y) in &points {
            problem.add_residual_block(
                1,
                &["ab"],
                Box::new(LineFactor { x, y }),
                Some(Box::new(HuberLoss::new(scale))),
            );
        }
        let initial_values = HashMap::from([("ab".to_string(), na::dvector![0.0, 0.0])]);
        // Only stop once the step is negligible.
        let options = tiny_solver::optimizer::OptimizerOptions {
            min_abs_error_decrease_threshold: 0.0,
            min_rel_error_decrease_threshold: 0.0,
            min_error_threshold: 0.0,
            ..Default::default()
        };

        let result = tiny_solver::LevenbergMarquardtOptimizer::default()
            .optimize(&problem, &initial_values, Some(options))
            .unwrap();

        // Gradient of sum_i rho(r_i^2), with rho'(s) = 1 for s <= scale^2 and
        // scale / sqrt(s) beyond.
        let gradient = |a: f64, b: f64| {
            points.iter().fold([0.0, 0.0], |g, &(x, y)| {
                let r = a * x + b - y;
                let rho1 = if r * r <= scale * scale {
                    1.0
                } else {
                    scale / r.abs()
                };
                [g[0] + 2.0 * rho1 * r * x, g[1] + 2.0 * rho1 * r]
            })
        };
        let g0 = gradient(0.0, 0.0);
        let g = gradient(result["ab"][0], result["ab"][1]);
        assert!(
            g[0].abs().max(g[1].abs()) < 1e-6 * g0[0].abs().max(g0[1].abs()),
            "gradient {g:?} at a = {}, b = {}",
            result["ab"][0],
            result["ab"][1]
        );
    }
}
