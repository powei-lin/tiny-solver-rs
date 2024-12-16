use nalgebra as na;

pub trait Factor<T: na::RealField>: Send + Sync {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T>;
}
pub trait FactorImpl: Factor<num_dual::DualDVec64> + Factor<f64> {
    fn residual_func_dual(
        &self,
        params: &[na::DVector<num_dual::DualDVec64>],
    ) -> na::DVector<num_dual::DualDVec64> {
        self.residual_func(params)
    }
    fn residual_func_f64(&self, params: &[na::DVector<f64>]) -> na::DVector<f64> {
        self.residual_func(params)
    }
}

impl<T> FactorImpl for T
where
    T: Factor<num_dual::DualDVec64> + Factor<f64>,
{
    fn residual_func_dual(
        &self,
        params: &[na::DVector<num_dual::DualDVec64>],
    ) -> na::DVector<num_dual::DualDVec64> {
        self.residual_func(params)
    }

    fn residual_func_f64(&self, params: &[na::DVector<f64>]) -> na::DVector<f64> {
        self.residual_func(params)
    }
}

#[derive(Debug, Clone)]
pub struct BetweenFactorSE2 {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
}
impl<T: na::RealField> Factor<T> for BetweenFactorSE2 {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
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
            na::Vector2::<T>::new(T::from_f64(self.dx).unwrap(), T::from_f64(self.dy).unwrap()),
            T::from_f64(self.dtheta).unwrap(),
        );

        let se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;
        na::dvector![
            se2_diff.translation.x.clone(),
            se2_diff.translation.y.clone(),
            se2_diff.rotation.angle()
        ]
    }
}

#[derive(Debug, Clone)]
pub struct PriorFactor {
    pub v: na::DVector<f64>,
}
impl<T: na::RealField> Factor<T> for PriorFactor {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        params[0].clone() - self.v.clone().cast()
    }
}
