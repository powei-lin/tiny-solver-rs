#[cfg(test)]
mod tests {
    use core::f64;
    use tiny_solver::loss_functions::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_derivatives(loss: &dyn Loss, squared_norm: f64) {
        let step = 1e-4;
        let rho = loss.evaluate(squared_norm);
        let forward = loss.evaluate(squared_norm + step);
        let backward = loss.evaluate(squared_norm - step);
        let first_derivative = (forward[0] - backward[0]) / (2.0 * step);
        let second_derivative = (forward[0] - 2.0 * rho[0] + backward[0]) / (step * step);

        assert_close(rho[1], first_derivative, 1e-6);
        assert_close(rho[2], second_derivative, 1e-6);
    }

    #[test]
    fn trivial_loss() {
        let loss = TrivialLoss;
        assert_eq!(loss.evaluate(0.0), [0.0, 1.0, 0.0]);
        assert_derivatives(&loss, 0.357);
        assert_derivatives(&loss, 1.792);
    }

    #[test]
    fn huber_loss() {
        for scale in [0.7, 1.3] {
            let loss = HuberLoss::new(scale);
            assert_derivatives(&loss, 0.357);
            assert_derivatives(&loss, 1.792);
        }
        assert_eq!(HuberLoss::new(0.7).evaluate(0.0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn soft_l_one_loss() {
        for scale in [0.7, 1.3] {
            let loss = SoftLOneLoss::new(scale);
            assert_derivatives(&loss, 0.357);
            assert_derivatives(&loss, 1.792);
        }

        let rho = SoftLOneLoss::new(0.7).evaluate(0.0);
        assert_close(rho[0], 0.0, 1e-12);
        assert_close(rho[1], 1.0, 1e-12);
        assert_close(rho[2], -0.5 / (0.7 * 0.7), 1e-12);
    }

    #[test]
    fn cauchy_loss_uses_natural_logarithm() {
        for scale in [0.7, 1.3] {
            let loss = CauchyLoss::new(scale);
            assert_derivatives(&loss, 0.357);
            assert_derivatives(&loss, 1.792);
        }

        assert_close(CauchyLoss::new(1.0).evaluate(3.0)[0], 4.0_f64.ln(), 1e-12);
        let rho = CauchyLoss::new(0.7).evaluate(0.0);
        assert_eq!(rho[0], 0.0);
        assert_eq!(rho[1], 1.0);
        assert_close(rho[2], -1.0 / (0.7 * 0.7), 1e-12);
    }

    #[test]
    fn arctan_loss() {
        let tolerance = 100.0;
        let asymptote = tolerance * f64::consts::PI / 2.0;

        let arctan_loss = ArctanLoss::new(tolerance);

        let rho1 = arctan_loss.evaluate(1.0);
        let rho2 = arctan_loss.evaluate(30.0);

        // Test that rho[0] grows linearly with the scale
        assert!(rho1[0] < rho2[0]);

        // Test that rho[1] decreases linearly with the scale
        assert!(rho1[1] > rho2[1]);

        // Test that scales largely above the tolerance are asymptotically bounded
        assert!(arctan_loss.evaluate(tolerance * tolerance * tolerance)[0] < asymptote);

        for scale in [0.7, 1.3] {
            let loss = ArctanLoss::new(scale);
            assert_derivatives(&loss, 0.357);
            assert_derivatives(&loss, 1.792);
        }
        assert_eq!(ArctanLoss::new(0.7).evaluate(0.0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn tolerant_loss() {
        for (tolerance, transition) in [(0.7, 0.4), (1.3, 0.1)] {
            let loss = TolerantLoss::new(tolerance, transition);
            for squared_norm in [0.357, 1.792, 55.5] {
                assert_derivatives(&loss, squared_norm);
            }
        }

        assert_close(TolerantLoss::new(0.7, 0.4).evaluate(0.0)[0], 0.0, 1e-12);
        let loss = TolerantLoss::new(20.0, 1.0);
        for squared_norm in [56.6, 56.7, 56.8, 1020.0] {
            assert_derivatives(&loss, squared_norm);
        }
    }

    #[test]
    fn tukey_loss() {
        for scale in [0.7, 1.3] {
            let loss = TukeyLoss::new(scale);
            assert_derivatives(&loss, 0.357);
            assert_derivatives(&loss, 1.792);
        }

        let rho = TukeyLoss::new(0.7).evaluate(0.0);
        assert_eq!(rho[0], 0.0);
        assert_eq!(rho[1], 1.0);
        assert_close(rho[2], -2.0 / (0.7 * 0.7), 1e-12);

        let saturated = TukeyLoss::new(0.7).evaluate(1.0);
        assert_close(saturated[0], 0.7 * 0.7 / 3.0, 1e-12);
        assert_eq!(saturated[1], 0.0);
        assert_eq!(saturated[2], 0.0);
    }

    #[test]
    fn composed_loss_applies_the_chain_rule() {
        let huber_cauchy = ComposedLoss::new(
            Box::new(HuberLoss::new(0.7)),
            Box::new(CauchyLoss::new(1.3)),
        );
        let cauchy_huber = ComposedLoss::new(
            Box::new(CauchyLoss::new(0.7)),
            Box::new(HuberLoss::new(1.3)),
        );

        for squared_norm in [0.357, 1.792] {
            assert_derivatives(&huber_cauchy, squared_norm);
            assert_derivatives(&cauchy_huber, squared_norm);
        }
    }

    #[test]
    fn scaled_loss_scales_value_and_derivatives() {
        let identity = ScaledLoss::new(None, 6.0);
        let rho = identity.evaluate(0.323);
        assert_close(rho[0], 1.938, 1e-12);
        assert_eq!(rho[1], 6.0);
        assert_eq!(rho[2], 0.0);
        assert_derivatives(&identity, 0.323);

        let loss = ScaledLoss::new(Some(Box::new(SoftLOneLoss::new(1.3))), 0.1);
        assert_derivatives(&loss, 1.792);
    }
}
