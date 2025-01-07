use std::{num::NonZero, ops::Mul};

use nalgebra as na;

use super::{AutoDiffManifold, Manifold};

pub struct SO3<T: na::RealField> {
    qx: T,
    qy: T,
    qz: T,
    qw: T,
}

impl<T: na::RealField> SO3<T> {
    /// [x, y, z, w]
    pub fn from_vec(xyzw: na::DVectorView<T>) -> Self {
        SO3 {
            qx: xyzw[0].clone(),
            qy: xyzw[1].clone(),
            qz: xyzw[2].clone(),
            qw: xyzw[3].clone(),
        }
    }
    pub fn from_xyzw(x: T, y: T, z: T, w: T) -> Self {
        SO3 {
            qx: x,
            qy: y,
            qz: z,
            qw: w,
        }
    }
    pub fn identity() -> Self {
        SO3 {
            qx: T::zero(),
            qy: T::zero(),
            qz: T::zero(),
            qw: T::one(),
        }
    }

    pub fn exp(xi: na::DVectorView<T>) -> Self {
        let mut xyzw = na::Vector4::zeros();

        let theta2 = xi.norm_squared();

        if theta2 < T::from_f64(1e-6).unwrap() {
            // cos(theta / 2) \approx 1 - theta^2 / 8
            xyzw.w = T::one() - theta2 / T::from_f64(8.0).unwrap();
            // Complete the square so that norm is one
            let tmp = T::from_f64(0.5).unwrap();
            xyzw.x = xi[0].clone() * tmp.clone();
            xyzw.y = xi[1].clone() * tmp.clone();
            xyzw.z = xi[2].clone() * tmp;
        } else {
            let theta = theta2.sqrt();
            xyzw.w = (theta.clone() * T::from_f64(0.5).unwrap()).cos();

            let omega = xi / theta;
            let sin_theta_half = (T::one() - xyzw.w.clone() * xyzw.w.clone()).sqrt();
            xyzw.x = omega[0].clone() * sin_theta_half.clone();
            xyzw.y = omega[1].clone() * sin_theta_half.clone();
            xyzw.z = omega[2].clone() * sin_theta_half;
        }

        SO3::from_vec(xyzw.as_view())
    }

    pub fn log(&self) -> na::DVector<T> {
        let xi = na::dvector![self.qx.clone(), self.qy.clone(), self.qz.clone()];
        // Abs value in case we had a negative quaternion
        let w = self.qw.clone().abs();

        let norm_v = xi.norm();
        if norm_v < T::from_f64(1e-3).unwrap() {
            xi * T::from_f64(2.0).unwrap()
        } else {
            xi * norm_v.clone().atan2(w) * T::from_f64(2.0).unwrap() / norm_v
        }
    }

    pub fn hat(xi: na::VectorView3<T>) -> na::Matrix3<T> {
        let mut xi_hat = na::Matrix3::zeros();
        xi_hat[(0, 1)] = -xi[2].clone();
        xi_hat[(0, 2)] = xi[1].clone();
        xi_hat[(1, 0)] = xi[2].clone();
        xi_hat[(1, 2)] = -xi[0].clone();
        xi_hat[(2, 0)] = -xi[1].clone();
        xi_hat[(2, 1)] = xi[0].clone();

        xi_hat
    }

    pub fn to_vec(&self) -> na::Vector4<T> {
        na::Vector4::new(
            self.qx.clone(),
            self.qy.clone(),
            self.qz.clone(),
            self.qw.clone(),
        )
    }
    pub fn to_dvec(&self) -> na::DVector<T> {
        na::dvector![
            self.qx.clone(),
            self.qy.clone(),
            self.qz.clone(),
            self.qw.clone(),
        ]
    }
    pub fn cast<U: na::RealField + simba::scalar::SupersetOf<T>>(&self) -> SO3<U> {
        SO3::from_vec(self.to_vec().cast().as_view())
    }
    pub fn inverse(&self) -> Self {
        SO3 {
            qx: -self.qx.clone(),
            qy: -self.qy.clone(),
            qz: -self.qz.clone(),
            qw: self.qw.clone(),
        }
    }
    pub fn compose(&self, rhs: &Self) -> Self {
        let x0 = self.qx.clone();
        let y0 = self.qy.clone();
        let z0 = self.qz.clone();
        let w0 = self.qw.clone();

        let x1 = rhs.qx.clone();
        let y1 = rhs.qy.clone();
        let z1 = rhs.qz.clone();
        let w1 = rhs.qw.clone();

        // Compute the product of the two quaternions, term by term
        let qx = w0.clone() * x1.clone() + x0.clone() * w1.clone() + y0.clone() * z1.clone()
            - z0.clone() * y1.clone();
        let qy = w0.clone() * y1.clone() - x0.clone() * z1.clone()
            + y0.clone() * w1.clone()
            + z0.clone() * x1.clone();
        let qz = w0.clone() * z1.clone() + x0.clone() * y1.clone() - y0.clone() * x1.clone()
            + z0.clone() * w1.clone();
        let qw = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;

        SO3 { qx, qy, qz, qw }
    }
}

impl<T: na::RealField> Mul for SO3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}
impl<T: na::RealField> Mul for &SO3<T> {
    type Output = SO3<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(rhs)
    }
}

impl<T: na::RealField> Mul<na::VectorView3<'_, T>> for &SO3<T> {
    type Output = na::Vector3<T>;

    fn mul(self, rhs: na::VectorView3<'_, T>) -> Self::Output {
        let qv = SO3::from_xyzw(rhs[0].clone(), rhs[1].clone(), rhs[2].clone(), T::zero());
        let inv = self.inverse();
        let v_rot: SO3<T> = (self * &qv) * inv;
        na::Vector3::new(v_rot.qx, v_rot.qy, v_rot.qz)
    }
}

pub struct QuaternionManifold {}
impl<T: na::RealField> AutoDiffManifold<T> for QuaternionManifold {
    fn plus(
        &self,
        x: nalgebra::DVectorView<T>,
        delta: nalgebra::DVectorView<T>,
    ) -> nalgebra::DVector<T> {
        let d: SO3<T> = SO3::exp(delta);
        let x_s03: SO3<T> = SO3::from_xyzw(x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone());
        let x_plus = x_s03 * d;
        na::dvector![
            x_plus.qx.clone(),
            x_plus.qy.clone(),
            x_plus.qz.clone(),
            x_plus.qw.clone(),
        ]
    }

    fn minus(
        &self,
        y: nalgebra::DVectorView<T>,
        x: nalgebra::DVectorView<T>,
    ) -> nalgebra::DVector<T> {
        let y_so3 = SO3::from_vec(y);
        let x_so3_inv = SO3::from_vec(x).inverse();
        let x_inv_y_log = (x_so3_inv * y_so3).log();
        na::dvector![
            x_inv_y_log[0].clone(),
            x_inv_y_log[1].clone(),
            x_inv_y_log[2].clone()
        ]
    }
}
impl Manifold for QuaternionManifold {
    fn tangent_size(&self) -> NonZero<usize> {
        NonZero::new(3).unwrap()
    }
}
