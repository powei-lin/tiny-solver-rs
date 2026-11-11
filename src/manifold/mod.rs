use std::num::NonZero;

use nalgebra as na;

use crate::factors::DualStride;

pub mod se3;
pub mod so3;

pub trait AutoDiffManifold<T: na::RealField> {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T>;
    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T>;
}

pub trait Manifold: AutoDiffManifold<f64> + AutoDiffManifold<DualStride> {
    fn tangent_size(&self) -> NonZero<usize>;
    fn plus_f64(&self, x: na::DVectorView<f64>, delta: na::DVectorView<f64>) -> na::DVector<f64> {
        self.plus(x, delta)
    }
    fn plus_dual(
        &self,
        x: na::DVectorView<DualStride>,
        delta: na::DVectorView<DualStride>,
    ) -> na::DVector<DualStride> {
        self.plus(x, delta)
    }
    fn minus_f64(&self, y: na::DVectorView<f64>, x: na::DVectorView<f64>) -> na::DVector<f64> {
        self.minus(y, x)
    }
    fn minus_dual(
        &self,
        y: na::DVectorView<DualStride>,
        x: na::DVectorView<DualStride>,
    ) -> na::DVector<DualStride> {
        self.minus(y, x)
    }
}
