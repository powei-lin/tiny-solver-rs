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
        let dual_params = na::dvector![
            num_dual::DualDVec64::new(1.0, num_dual::Derivative::some(na::DVector::identity(1))),
            num_dual::DualDVec64::new(2.0, num_dual::Derivative::some(na::DVector::identity(1)))
        ];

        let residual_dual = prior_factor.residual_func_dual(&[dual_params]);
        assert_eq!(residual_dual[0].re, -2.0);
        assert_eq!(residual_dual[1].re, 1.0);
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
