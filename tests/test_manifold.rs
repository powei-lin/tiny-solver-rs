#[cfg(test)]
mod tests {
    use nalgebra as na;
    use std::f64::consts::PI;
    use std::sync::Arc;

    use tiny_solver::manifold::so3::SO3;
    use tiny_solver::manifold::{
        EigenQuaternionManifold, EuclideanManifold, LineManifold, Manifold, ProductManifold,
        QuaternionManifold, SphereManifold, SubsetManifold,
    };
    use tiny_solver::parameter_block::ParameterBlock;
    use tiny_solver::{ResidualBlock, factors::PriorFactor};

    fn equal_to_na(so3: &SO3<f64>) -> bool {
        let q = so3.to_vec();
        let na_so3 =
            na::UnitQuaternion::from_quaternion(na::Quaternion::new(q[3], q[0], q[1], q[2]));
        let diff = so3.log() - na_so3.scaled_axis();
        diff.norm() < 1e-6
    }

    #[test]
    fn test_so3() {
        for _ in 0..10000 {
            let mut rvec = na::DVector::new_random(3);
            rvec /= rvec.norm();
            rvec *= rand::random::<f64>() * PI;
            let r = SO3::exp(rvec.as_view());
            assert!(equal_to_na(&r));
        }
    }

    #[test]
    fn euclidean_plus_and_minus_are_inverse() {
        let manifold = EuclideanManifold::new(3);
        let x = na::dvector![1.0, 2.0, 3.0];
        let delta = na::dvector![0.5, -1.0, 2.0];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());

        assert_eq!(manifold.ambient_size(), 3);
        assert_eq!(manifold.tangent_size(), 3);
        assert_eq!(y, na::dvector![1.5, 1.0, 5.0]);
        assert_eq!(manifold.minus_f64(y.as_view(), x.as_view()), delta);
    }

    #[test]
    fn subset_manifold_only_updates_variable_components() {
        let manifold = SubsetManifold::new(5, &[1, 3]);
        let x = na::dvector![1.0, 2.0, 3.0, 4.0, 5.0];
        let delta = na::dvector![0.5, -1.0, 2.0];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());

        assert_eq!(manifold.ambient_size(), 5);
        assert_eq!(manifold.tangent_size(), 3);
        assert_eq!(y, na::dvector![1.5, 2.0, 2.0, 4.0, 7.0]);
        assert_eq!(manifold.minus_f64(y.as_view(), x.as_view()), delta);
    }

    #[test]
    fn subset_manifold_can_hold_every_component_constant() {
        let manifold = SubsetManifold::new(3, &[0, 1, 2]);
        let x = na::dvector![1.0, 2.0, 3.0];
        let delta = na::DVector::zeros(0);

        assert_eq!(manifold.tangent_size(), 0);
        assert_eq!(manifold.plus_f64(x.as_view(), delta.as_view()), x);
        assert_eq!(
            manifold.minus_f64(x.as_view(), x.as_view()),
            na::DVector::zeros(0)
        );
    }

    #[test]
    fn quaternion_manifold_matches_ceres_wxyz_layout() {
        let manifold = QuaternionManifold;
        let x: na::DVector<f64> = na::dvector![0.9, 0.1, -0.2, 0.3];
        let x = x.clone() / x.norm();
        let delta: na::DVector<f64> = na::dvector![0.1, -0.05, 0.2];
        let norm = delta.norm();
        let delta_quaternion = na::Quaternion::new(
            norm.cos(),
            norm.sin() * delta[0] / norm,
            norm.sin() * delta[1] / norm,
            norm.sin() * delta[2] / norm,
        );
        let x_quaternion = na::Quaternion::new(x[0], x[1], x[2], x[3]);
        let expected = delta_quaternion * x_quaternion;
        let expected = na::dvector![expected.w, expected.i, expected.j, expected.k];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());

        assert!((y.norm() - 1.0).abs() < 1e-12);
        assert!((&y - expected).norm() < 1e-12);
        assert!((manifold.minus_f64(y.as_view(), x.as_view()) - delta).norm() < 1e-12);
    }

    #[test]
    fn eigen_quaternion_manifold_uses_xyzw_layout() {
        let manifold = EigenQuaternionManifold;
        let x: na::DVector<f64> = na::dvector![0.1, -0.2, 0.3, 0.9];
        let x = x.clone() / x.norm();
        let delta: na::DVector<f64> = na::dvector![0.1, -0.05, 0.2];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());
        let recovered = manifold.minus_f64(y.as_view(), x.as_view());

        assert!((y.norm() - 1.0).abs() < 1e-12);
        assert!((recovered - delta).norm() < 1e-12);
        assert_eq!(manifold.ambient_size(), 4);
        assert_eq!(manifold.tangent_size(), 3);
    }

    #[test]
    fn sphere_manifold_preserves_norm_and_inverts_retraction() {
        let manifold = SphereManifold::new(3);
        let x: na::DVector<f64> = na::dvector![1.0, -2.0, 3.0];
        let delta: na::DVector<f64> = na::dvector![0.2, -0.1];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());
        let recovered = manifold.minus_f64(y.as_view(), x.as_view());

        assert!((y.norm() - x.norm()).abs() < 1e-12);
        assert!((recovered - delta).norm() < 1e-12);
        assert_eq!(manifold.ambient_size(), 3);
        assert_eq!(manifold.tangent_size(), 2);
    }

    #[test]
    fn sphere_manifold_handles_zero_and_antipodal_updates() {
        let manifold = SphereManifold::new(3);
        let x: na::DVector<f64> = na::dvector![1.0, -2.0, 3.0];

        let unchanged = manifold.plus_f64(x.as_view(), na::DVector::zeros(2).as_view());
        assert!((unchanged - &x).norm() < 1e-12);

        let antipodal_delta = manifold.minus_f64((-&x).as_view(), x.as_view());
        assert!((antipodal_delta.norm() - PI).abs() < 1e-12);
        let antipodal = manifold.plus_f64(x.as_view(), antipodal_delta.as_view());
        assert!((antipodal + x).norm() < 1e-12);
    }

    #[test]
    fn product_manifold_dispatches_ambient_and_tangent_blocks() {
        let components: Vec<Arc<dyn Manifold + Send + Sync>> = vec![
            Arc::new(EigenQuaternionManifold),
            Arc::new(EuclideanManifold::new(3)),
        ];
        let manifold = ProductManifold::new(components);
        let x: na::DVector<f64> = na::dvector![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0];
        let delta: na::DVector<f64> = na::dvector![0.1, -0.2, 0.05, 2.0, -1.0, 0.5];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());
        let recovered = manifold.minus_f64(y.as_view(), x.as_view());

        assert_eq!(manifold.ambient_size(), 7);
        assert_eq!(manifold.tangent_size(), 6);
        assert!((y.rows(0, 4).norm() - 1.0).abs() < 1e-12);
        assert_eq!(y.rows(4, 3), na::dvector![3.0, 1.0, 3.5]);
        assert!((recovered - delta).norm() < 1e-12);
    }

    #[test]
    fn line_manifold_updates_origin_orthogonally_to_direction() {
        let manifold = LineManifold::new(3);
        let direction = na::Vector3::new(1.0, -2.0, 3.0).normalize();
        let x: na::DVector<f64> =
            na::dvector![2.0, -1.0, 0.5, direction[0], direction[1], direction[2]];
        let delta: na::DVector<f64> = na::dvector![0.3, -0.1, 0.2, 0.05];

        let y = manifold.plus_f64(x.as_view(), delta.as_view());
        let recovered = manifold.minus_f64(y.as_view(), x.as_view());
        let origin_change = y.rows(0, 3) - x.rows(0, 3);

        assert_eq!(manifold.ambient_size(), 6);
        assert_eq!(manifold.tangent_size(), 4);
        assert!(origin_change.dot(&direction).abs() < 1e-12);
        assert!((y.rows(3, 3).norm() - 1.0).abs() < 1e-12);
        assert!((recovered - delta).norm() < 1e-12);
    }

    #[test]
    fn new_manifolds_produce_finite_autodiff_jacobians_at_zero_delta() {
        let product_components: Vec<Arc<dyn Manifold + Send + Sync>> = vec![
            Arc::new(EigenQuaternionManifold),
            Arc::new(EuclideanManifold::new(3)),
        ];
        let cases: Vec<(Arc<dyn Manifold + Send + Sync>, na::DVector<f64>)> = vec![
            (
                Arc::new(QuaternionManifold),
                na::dvector![1.0, 0.0, 0.0, 0.0],
            ),
            (
                Arc::new(EigenQuaternionManifold),
                na::dvector![0.0, 0.0, 0.0, 1.0],
            ),
            (
                Arc::new(SphereManifold::new(3)),
                na::dvector![1.0, 2.0, 3.0],
            ),
            (
                Arc::new(LineManifold::new(3)),
                na::dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            (
                Arc::new(ProductManifold::new(product_components)),
                na::dvector![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
            ),
        ];

        for (manifold, parameters) in cases {
            let mut parameter_block = ParameterBlock::from_vec(parameters.clone());
            parameter_block.set_manifold(manifold.clone());
            let residual_block = ResidualBlock::new(
                0,
                parameters.len(),
                0,
                &["x"],
                Box::new(PriorFactor {
                    v: parameters.clone(),
                }),
                None,
            );

            let (_, jacobian) = residual_block.residual_and_jacobian(&[&parameter_block]);

            assert_eq!(jacobian.ncols(), manifold.tangent_size());
            assert!(jacobian.iter().all(|value| value.is_finite()));
        }
    }
}
