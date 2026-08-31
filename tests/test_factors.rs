#[cfg(test)]
mod tests {
    use nalgebra as na;
    use tiny_solver::ResidualBlock;
    use tiny_solver::factors::*;
    use tiny_solver::manifold::EigenQuaternionManifold;
    use tiny_solver::parameter_block::ParameterBlock;

    struct F64OnlyFactor;

    impl Factor<f64> for F64OnlyFactor {
        fn residual_func(&self, params: &[na::DVector<f64>]) -> na::DVector<f64> {
            let x = &params[0];
            let y = params[1][0];
            na::dvector![x[0] * x[0] + y.sin(), x[1] * y]
        }
    }

    struct AnalyticTestFactor;

    impl AnalyticFactor for AnalyticTestFactor {
        fn residual_and_jacobians(
            &self,
            params: &[na::DVector<f64>],
        ) -> (na::DVector<f64>, Vec<na::DMatrix<f64>>) {
            let x = &params[0];
            let y = params[1][0];
            (
                na::dvector![x[0] * x[0] + y.sin(), x[1] * y],
                vec![
                    na::DMatrix::from_row_slice(2, 2, &[2.0 * x[0], 0.0, 0.0, y]),
                    na::DMatrix::from_column_slice(2, 1, &[y.cos(), x[1]]),
                ],
            )
        }
    }

    struct AnalyticIdentityFactor;

    impl AnalyticFactor for AnalyticIdentityFactor {
        fn residual_and_jacobians(
            &self,
            params: &[na::DVector<f64>],
        ) -> (na::DVector<f64>, Vec<na::DMatrix<f64>>) {
            (
                params[0].clone(),
                vec![na::DMatrix::identity(params[0].len(), params[0].len())],
            )
        }
    }

    fn evaluate_factor(factor: Box<dyn FactorImpl + Send>) -> na::DMatrix<f64> {
        let x = ParameterBlock::from_vec(na::dvector![1.2, -0.7]);
        let y = ParameterBlock::from_vec(na::dvector![0.4]);
        let residual_block = ResidualBlock::new(0, 2, 0, &["x", "y"], factor, None);
        residual_block.residual_and_jacobian(&[&x, &y]).1
    }

    fn expected_test_jacobian() -> na::DMatrix<f64> {
        na::DMatrix::from_row_slice(2, 3, &[2.4, 0.0, 0.4_f64.cos(), 0.0, 0.4, -0.7])
    }

    fn assert_matrix_close(actual: &na::DMatrix<f64>, expected: &na::DMatrix<f64>, tolerance: f64) {
        assert_eq!(actual.shape(), expected.shape());
        assert!(
            (actual - expected).abs().max() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn prior_factor() {
        let prior_factor = PriorFactor {
            v: na::dvector![3.0, 1.0],
        };

        let params = na::dvector![1.0, 2.0];

        let residual = prior_factor.residual_func(&[params]);
        assert_eq!(residual[0], -2.0);
        assert_eq!(residual[1], 1.0);
        // the parameters contains two values [a, b]
        let dual_params = na::dvector![
            // dual vec has [1.0, 0.0] means it's for tracking the derivative of the first parameter a
            num_dual::DualDVec64::new(1.0, num_dual::Derivative::some(na::dvector![1.0, 0.0])),
            // dual vec has [0.0, 1.0] is for tracking the derivative of the second parameter b
            num_dual::DualDVec64::new(2.0, num_dual::Derivative::some(na::dvector![0.0, 1.0]))
        ];
        // there are two residuals r0, r1
        let residual_with_jacobian = prior_factor.residual_func_dual(&[dual_params]);
        assert!(residual_with_jacobian[0].re == -2.0);
        assert!(residual_with_jacobian[1].re == 1.0);

        let jacobian =
            residual_with_jacobian.map(|x| x.eps.unwrap_generic(na::Dyn(2), na::Const::<1>));

        // partial derivative of r0 with respect to a
        let d_r0_d_a: f64 = jacobian[0][0];
        // partial derivative of r0 with respect to b
        let d_r0_d_b: f64 = jacobian[0][1];

        // partial derivative of r1 with respect to a
        let d_r1_d_a: f64 = jacobian[1][0];
        // partial derivative of r1 with respect to b
        let d_r1_d_b: f64 = jacobian[1][1];

        // a only contribute to r0 and b only contribute to r1
        assert!(d_r0_d_a == 1.0);
        assert!(d_r0_d_b == 0.0);
        assert!(d_r1_d_a == 0.0);
        assert!(d_r1_d_b == 1.0);
    }

    #[test]
    fn between_factor_se2() {
        let factor = BetweenFactorSE2 {
            dtheta: 1.0,
            dx: 2.0,
            dy: 3.0,
        };

        let params = [na::dvector![1.0, 2.0, 3.0], na::dvector![1.0, 2.0, 3.0]];

        let residual = factor.residual_func(&params);
        assert_eq!(residual, na::dvector![2.0, 3.0, 1.0]);
    }

    #[test]
    fn numeric_diff_methods_match_analytic_jacobian() {
        let expected = expected_test_jacobian();
        for (method, tolerance) in [
            (NumericDiffMethod::Forward, 2e-6),
            (NumericDiffMethod::Central, 1e-9),
            (NumericDiffMethod::Ridders, 1e-11),
        ] {
            let jacobian = evaluate_factor(Box::new(NumericDiffFactor::new(F64OnlyFactor, method)));
            assert_matrix_close(&jacobian, &expected, tolerance);
        }
    }

    #[test]
    fn analytic_factor_adapter_uses_supplied_jacobians() {
        let jacobian = evaluate_factor(Box::new(AnalyticFactorAdapter::new(AnalyticTestFactor)));
        assert_matrix_close(&jacobian, &expected_test_jacobian(), 1e-12);
    }

    #[test]
    fn analytic_ambient_jacobian_is_projected_to_manifold_tangent_space() {
        let mut quaternion = ParameterBlock::from_vec(na::dvector![0.0, 0.0, 0.0, 1.0]);
        quaternion.set_manifold(std::sync::Arc::new(EigenQuaternionManifold));
        let residual_block = ResidualBlock::new(
            0,
            4,
            0,
            &["q"],
            Box::new(AnalyticFactorAdapter::new(AnalyticIdentityFactor)),
            None,
        );

        let (_, jacobian) = residual_block.residual_and_jacobian(&[&quaternion]);

        assert_matrix_close(
            &jacobian,
            &na::DMatrix::from_row_slice(
                4,
                3,
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
            1e-12,
        );
    }
}
