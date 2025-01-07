use std::{num::NonZero, ops::Mul};

use nalgebra as na;

use super::{lie_group::LieGroup, so3::SO3, AutoDiffManifold, Manifold};

pub struct SE3<T: na::RealField> {
    xyz: na::Vector3<T>,
    rot: SO3<T>,
}

impl<T: na::RealField> LieGroup<T> for SE3<T> {
    const TANGENT_SIZE: usize = 6;

    fn exp(xi: nalgebra::DVectorView<T>) -> Self {
        let rot = SO3::<T>::exp(xi.rows(0, 3).as_view());
        let xyz = na::Vector3::new(xi[3].clone(), xi[4].clone(), xi[5].clone());
        SE3 { xyz, rot }
    }
}

impl<T: na::RealField> SE3<T> {
    /// [qx, qy, qz, qw, tx, ty, tz]
    pub fn from_vec(qxyzw_txyz: na::DVectorView<T>) -> Self {
        let rot = SO3::from_xyzw(
            qxyzw_txyz[0].clone(),
            qxyzw_txyz[1].clone(),
            qxyzw_txyz[2].clone(),
            qxyzw_txyz[3].clone(),
        );
        let xyz = na::Vector3::new(
            qxyzw_txyz[4].clone(),
            qxyzw_txyz[5].clone(),
            qxyzw_txyz[6].clone(),
        );
        SE3 { xyz, rot }
    }
    pub fn from_qvec_tvec(qxyzw: na::DVectorView<T>, tvec: na::DVectorView<T>) -> Self {
        let rot = SO3::from_vec(qxyzw);
        let xyz = na::Vector3::from_row_slice(tvec.as_slice());
        SE3 { xyz, rot }
    }
    pub fn identity() -> Self {
        let rot = SO3::identity();
        let xyz = na::Vector3::zeros();
        SE3 { xyz, rot }
    }

    pub fn exp(xi: na::DVectorView<T>) -> Self {
        // fake exp
        let rot = SO3::<T>::exp(xi.rows(0, 3).as_view());
        let xyz = na::Vector3::new(xi[3].clone(), xi[4].clone(), xi[5].clone());
        SE3 { xyz, rot }
    }

    pub fn log(&self) -> na::DVector<T> {
        let mut xi = na::DVector::zeros(6);
        let xi_theta = self.rot.log();
        // let xyz = self.xyz;
        xi.as_mut_slice()[0..3].clone_from_slice(xi_theta.as_slice());
        xi.as_mut_slice()[3..6].clone_from_slice(self.xyz.as_slice());
        xi
    }

    pub fn to_dvec(&self) -> na::DVector<T> {
        let quat = self.rot.to_vec();
        na::dvector![
            quat[0].clone(),
            quat[1].clone(),
            quat[2].clone(),
            quat[3].clone(),
            self.xyz[0].clone(),
            self.xyz[1].clone(),
            self.xyz[2].clone(),
        ]
    }

    // pub fn hat(xi: na::VectorView3<T>) -> na::Matrix3<T> {
    //     let mut xi_hat = na::Matrix3::zeros();
    //     xi_hat[(0, 1)] = -xi[2].clone();
    //     xi_hat[(0, 2)] = xi[1].clone();
    //     xi_hat[(1, 0)] = xi[2].clone();
    //     xi_hat[(1, 2)] = -xi[0].clone();
    //     xi_hat[(2, 0)] = -xi[1].clone();
    //     xi_hat[(2, 1)] = xi[0].clone();

    //     xi_hat
    // }

    pub fn cast<U: na::RealField + simba::scalar::SupersetOf<T>>(&self) -> SE3<U> {
        SE3 {
            rot: self.rot.cast(),
            xyz: self.xyz.clone().cast(),
        }
    }
    pub fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        let xyz = -(&inv * self.xyz.as_view());
        SE3 { xyz, rot: inv }
    }
    pub fn compose(&self, rhs: &Self) -> Self {
        SE3 {
            rot: &self.rot * &rhs.rot,
            xyz: (&self.rot * rhs.xyz.as_view()) + self.xyz.clone(),
        }
    }
}

impl<T: na::RealField> Mul for SE3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}
impl<T: na::RealField> Mul for &SE3<T> {
    type Output = SE3<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(rhs)
    }
}

impl<T: na::RealField> Mul<na::VectorView3<'_, T>> for SE3<T> {
    type Output = na::Vector3<T>;

    fn mul(self, rhs: na::VectorView3<'_, T>) -> Self::Output {
        let qv = SO3::from_xyzw(rhs[0].clone(), rhs[1].clone(), rhs[2].clone(), T::zero());
        let rinv = self.rot.inverse();
        let v_rot = ((self.rot * qv) * rinv).to_vec();
        let v = na::Vector3::new(v_rot[0].clone(), v_rot[1].clone(), v_rot[2].clone());
        v + self.xyz
    }
}

pub struct SE3Manifold {}
impl<T: na::RealField> AutoDiffManifold<T> for SE3Manifold {
    fn plus(
        &self,
        x: nalgebra::DVectorView<T>,
        delta: nalgebra::DVectorView<T>,
    ) -> nalgebra::DVector<T> {
        let d = SE3::exp(delta);
        let x_se3 = SE3::from_vec(x);
        let x_plus = x_se3 * d;
        x_plus.to_dvec()
    }

    fn minus(
        &self,
        y: nalgebra::DVectorView<T>,
        x: nalgebra::DVectorView<T>,
    ) -> nalgebra::DVector<T> {
        let y_se3 = SE3::from_vec(y);
        let x_se3_inv = SE3::from_vec(x).inverse();
        let x_inv_y_log = (x_se3_inv * y_se3).log();
        na::dvector![
            x_inv_y_log[0].clone(),
            x_inv_y_log[1].clone(),
            x_inv_y_log[2].clone(),
            x_inv_y_log[3].clone(),
            x_inv_y_log[4].clone(),
            x_inv_y_log[5].clone()
        ]
    }
}
impl Manifold for SE3Manifold {
    fn tangent_size(&self) -> NonZero<usize> {
        NonZero::new(6).unwrap()
    }
}
