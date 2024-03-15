use nalgebra as na;
use pyo3::prelude::*;

pub trait Factor: Send + Sync {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64>;
}

#[pyclass]
#[derive(FromPyObject)]
// #[derive(Default, Debug, Clone, Copy)]
pub struct CostFactorSE2 {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
}
impl Factor for CostFactorSE2 {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64> {
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];
        let se2_origin_k0 = na::Isometry2::new(
            na::Vector2::new(t_origin_k0[1].clone(), t_origin_k0[2].clone()),
            t_origin_k0[0].clone(),
        );
        let se2_origin_k1 = na::Isometry2::new(
            na::Vector2::new(t_origin_k1[1].clone(), t_origin_k1[2].clone()),
            t_origin_k1[0].clone(),
        );
        let se2_k0_k1 = na::Isometry2::new(
            na::Vector2::<num_dual::DualDVec64>::new(
                num_dual::DualDVec64::from_re(self.dx),
                num_dual::DualDVec64::from_re(self.dy),
            ),
            num_dual::DualDVec64::from_re(self.dtheta),
        );

        let se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;
        return na::dvector![
            se2_diff.translation.x.clone(),
            se2_diff.translation.y.clone(),
            se2_diff.rotation.angle()
        ];
    }
}

#[pyclass]
#[derive(Default, Debug, Clone, Copy)]
pub struct BetweenFactor {}
impl Factor for BetweenFactor {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64> {
        return params[0].clone();
    }
}
