pub use nalgebra as na;

use std::sync::Arc;

use crate::manifold::se3::SE3;

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
pub struct BetweenFactorSE3 {
    pub dtx: f64,
    pub dty: f64,
    pub dtz: f64,
    pub dqx: f64,
    pub dqy: f64,
    pub dqz: f64,
    pub dqw: f64,
}
impl<T: na::RealField> Factor<T> for BetweenFactorSE3 {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];
        let se3_origin_k0 = SE3::from_vec(t_origin_k0.as_view());
        let se3_origin_k1 = SE3::from_vec(t_origin_k1.as_view());

        let se3_k0_k1 = SE3::from_vec(
            na::dvector![
                self.dqx, self.dqy, self.dqz, self.dqw, self.dtx, self.dty, self.dtz,
            ]
            .as_view(),
        )
        .cast::<T>();

        let se3_diff = se3_origin_k1.inverse() * se3_origin_k0 * se3_k0_k1.cast();

        se3_diff.log()
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

/// A linearized prior factor produced by Schur complement marginalization.
///
/// Encodes the quadratic cost `0.5 * ||sqrt_info * Δx + residual_at_lin||²`
/// where `Δx` is the tangent-space difference between the current value and
/// the linearization point.  Add this factor back into a new [`crate::Problem`]
/// to retain the information from the marginalized variables.
///
/// # Usage
/// ```ignore
/// let marg = problem.marginalize(&initial_values, &["point_0"])?;
/// let names: Vec<&str> = marg.variable_names.iter().map(|s| s.as_str()).collect();
/// let dim = marg.sqrt_info.nrows();
/// new_problem.add_residual_block(dim, &names, Box::new(marg), None);
/// ```
pub struct MarginalizationFactor {
    /// Remaining (kept) variable names, in the order used by the factor.
    pub variable_names: Vec<String>,
    /// Ambient-space linearization points for each kept variable.
    pub linearization_points: Vec<na::DVector<f64>>,
    /// Square-root information matrix `L^T` where `H_sc = L * L^T`.
    /// Shape: `(dim_keep, dim_keep)`.
    pub sqrt_info: na::DMatrix<f64>,
    /// Residual at the linearization point: `r₀ = L⁻¹ * b_sc`.
    /// Shape: `(dim_keep,)`.
    pub residual_at_lin: na::DVector<f64>,
    /// Optional manifold for each kept variable (same order as `variable_names`).
    pub manifolds: Vec<Option<Arc<dyn crate::manifold::Manifold + Sync + Send>>>,
}

impl Factor<f64> for MarginalizationFactor {
    fn residual_func(&self, params: &[na::DVector<f64>]) -> na::DVector<f64> {
        let total_tangent_dim = self.sqrt_info.ncols();
        let mut dx = na::DVector::<f64>::zeros(total_tangent_dim);
        let mut offset = 0;
        for (i, (x, x_lin)) in params.iter().zip(self.linearization_points.iter()).enumerate() {
            let delta = if let Some(m) = &self.manifolds[i] {
                m.minus_f64(x.as_view(), x_lin.as_view())
            } else {
                x - x_lin
            };
            let tdim = delta.len();
            dx.rows_mut(offset, tdim).copy_from(&delta);
            offset += tdim;
        }
        &self.sqrt_info * dx + &self.residual_at_lin
    }
}

impl Factor<num_dual::DualDVec64> for MarginalizationFactor {
    fn residual_func(
        &self,
        params: &[na::DVector<num_dual::DualDVec64>],
    ) -> na::DVector<num_dual::DualDVec64> {
        let total_tangent_dim = self.sqrt_info.ncols();
        let mut dx = na::DVector::<num_dual::DualDVec64>::zeros(total_tangent_dim);
        let mut offset = 0;
        for (i, (x, x_lin)) in params.iter().zip(self.linearization_points.iter()).enumerate() {
            let x_lin_dual = x_lin.clone().cast::<num_dual::DualDVec64>();
            let delta = if let Some(m) = &self.manifolds[i] {
                m.minus_dual(x.as_view(), x_lin_dual.as_view())
            } else {
                x - &x_lin_dual
            };
            let tdim = delta.len();
            dx.rows_mut(offset, tdim).copy_from(&delta);
            offset += tdim;
        }
        self.sqrt_info.clone().cast::<num_dual::DualDVec64>() * dx
            + self.residual_at_lin.clone().cast::<num_dual::DualDVec64>()
    }
}
