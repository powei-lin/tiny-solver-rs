#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra as na;
    use num_dual::DualDVec64;
    use tiny_solver::factors::*;
    use tiny_solver::manifold::AutoDiffManifold;
    use tiny_solver::manifold::se3::SE3Manifold;
    use tiny_solver::parameter_block::ParameterBlock;
    use tiny_solver::residual_block::ResidualBlock;

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
        let residual_with_jacobian = prior_factor.residual_func(&[dual_params]);
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

    /// Every residual depends on every coordinate, each in its own way.
    struct Mix;
    impl<T: na::RealField> Factor<T> for Mix {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            let mut r = na::DVector::from_element(3, T::zero());
            for (i, x) in params.iter().flat_map(|p| p.iter()).enumerate() {
                let w = T::from_f64((i + 1) as f64).unwrap();
                r[0] += x.clone() * w.clone();
                r[1] += x.clone() * x.clone() * w.clone();
                r[2] += (x.clone() * w).sin();
            }
            r
        }
    }

    /// The Jacobian in the tangent space, from num-dual in one pass: x ⊞ δ at δ = 0.
    fn reference_jacobian<F: Factor<DualDVec64>>(
        factor: &F,
        params: &[(na::DVector<f64>, bool)],
    ) -> na::DMatrix<f64> {
        let sizes: Vec<usize> = params
            .iter()
            .map(|(x, se3)| if *se3 { 6 } else { x.len() })
            .collect();
        let delta = na::DVector::zeros(sizes.iter().sum());
        let (_, jacobian) = num_dual::jacobian(
            |delta: na::DVector<DualDVec64>| {
                let mut offset = 0;
                let inputs: Vec<na::DVector<DualDVec64>> = params
                    .iter()
                    .zip(&sizes)
                    .map(|((x, se3), &n)| {
                        let x = x.map(DualDVec64::from_re);
                        let d = delta.rows(offset, n);
                        offset += n;
                        if *se3 {
                            SE3Manifold.plus(x.as_view(), d)
                        } else {
                            x + d
                        }
                    })
                    .collect();
                factor.residual_func(&inputs)
            },
            &delta,
        );
        jacobian
    }

    /// The Jacobian is computed in passes of a few directions at a time. It must
    /// match a single pass, whether the number of directions is a multiple of
    /// the pass size or not.
    #[test]
    fn jacobian_matches_a_single_pass() {
        let pose = |q: [f64; 4], t: [f64; 3]| {
            let q = na::Vector4::from(q).normalize();
            (na::dvector![q[0], q[1], q[2], q[3], t[0], t[1], t[2]], true)
        };
        let p0 = pose([0.1, -0.2, 0.3, 0.9], [1.0, 2.0, 3.0]);
        let p1 = pose([-0.3, 0.1, 0.2, 0.8], [0.5, -1.0, 2.0]);
        let euclidean = |n: usize| (na::DVector::from_fn(n, |i, _| 0.3 * i as f64 - 0.4), false);
        let between = BetweenFactorSE3 {
            dtx: 0.1,
            dty: 0.2,
            dtz: -0.3,
            dqx: 0.0,
            dqy: 0.0,
            dqz: 0.0,
            dqw: 1.0,
        };
        let prior = PriorFactor { v: p1.0.clone() };

        let check = |factor: Box<dyn FactorImpl + Send>,
                     expected: na::DMatrix<f64>,
                     params: &[(na::DVector<f64>, bool)]| {
            let blocks: Vec<ParameterBlock> = params
                .iter()
                .map(|(x, se3)| {
                    let mut block = ParameterBlock::from_vec(x.clone());
                    if *se3 {
                        block.manifold = Some(Arc::new(SE3Manifold));
                    }
                    block
                })
                .collect();
            let keys: Vec<String> = (0..params.len()).map(|i| format!("x{i}")).collect();
            let keys: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
            let block = ResidualBlock::new(0, expected.nrows(), 0, &keys, factor, None);
            let (_, jacobian) = block.residual_and_jacobian(&blocks.iter().collect::<Vec<_>>());
            assert_eq!(jacobian.shape(), expected.shape());
            assert!(
                (&jacobian - &expected).abs().max() < 1e-12,
                "{jacobian} != {expected}"
            );
        };

        // 12 and 7 Euclidean directions, 12 and 6 on SE3.
        for params in [[euclidean(5), euclidean(7)], [euclidean(3), euclidean(4)]] {
            check(Box::new(Mix), reference_jacobian(&Mix, &params), &params);
        }
        let params = [p0.clone(), p1.clone()];
        check(
            Box::new(between.clone()),
            reference_jacobian(&between, &params),
            &params,
        );
        let params = [p0];
        check(
            Box::new(prior.clone()),
            reference_jacobian(&prior, &params),
            &params,
        );
    }

    struct Constant;
    impl<T: na::RealField> Factor<T> for Constant {
        fn residual_func(&self, _params: &[na::DVector<T>]) -> na::DVector<T> {
            na::dvector![T::from_f64(2.0).unwrap()]
        }
    }

    /// A residual block with no variables still has its residual.
    #[test]
    fn residual_without_variables() {
        let block = ResidualBlock::new(0, 1, 0, &[], Box::new(Constant), None);
        let (residual, jacobian) = block.residual_and_jacobian(&[]);
        assert_eq!(residual, na::dvector![2.0]);
        assert_eq!(jacobian.shape(), (1, 0));
    }
}
