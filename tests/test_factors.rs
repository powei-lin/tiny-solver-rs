#[cfg(test)]
mod tests {
    use nalgebra as na;
    use tiny_solver::factors::*;

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
            dy: 3.0
        };

        let params = [na::dvector![1.0, 2.0, 3.0], na::dvector![1.0, 2.0, 3.0]];
        
        let residual = factor.residual_func(&params);
        assert_eq!(residual, na::dvector![2.0, 3.0, 1.0]);

    }
}
