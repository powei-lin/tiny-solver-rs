use std::sync::Arc;

use nalgebra as na;
use num_dual::DualDVec64;

use super::{AutoDiffManifold, Manifold};

pub type ManifoldArc = Arc<dyn Manifold + Send + Sync>;

#[derive(Clone)]
pub struct ProductManifold {
    manifolds: Vec<ManifoldArc>,
    ambient_offsets: Vec<usize>,
    tangent_offsets: Vec<usize>,
    ambient_size: usize,
    tangent_size: usize,
}

impl ProductManifold {
    pub fn new(manifolds: Vec<ManifoldArc>) -> Self {
        assert!(
            manifolds.len() >= 2,
            "a product manifold needs at least two components"
        );
        let mut ambient_offsets = Vec::with_capacity(manifolds.len());
        let mut tangent_offsets = Vec::with_capacity(manifolds.len());
        let mut ambient_size = 0;
        let mut tangent_size = 0;
        for manifold in &manifolds {
            ambient_offsets.push(ambient_size);
            tangent_offsets.push(tangent_size);
            ambient_size += manifold.ambient_size();
            tangent_size += manifold.tangent_size();
        }
        Self {
            manifolds,
            ambient_offsets,
            tangent_offsets,
            ambient_size,
            tangent_size,
        }
    }
}

impl AutoDiffManifold<f64> for ProductManifold {
    fn plus(&self, x: na::DVectorView<f64>, delta: na::DVectorView<f64>) -> na::DVector<f64> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(delta.len(), self.tangent_size);
        let mut result = na::DVector::zeros(self.ambient_size);
        for (index, manifold) in self.manifolds.iter().enumerate() {
            let ambient_offset = self.ambient_offsets[index];
            let tangent_offset = self.tangent_offsets[index];
            let component = manifold.plus_f64(
                x.rows(ambient_offset, manifold.ambient_size()),
                delta.rows(tangent_offset, manifold.tangent_size()),
            );
            result
                .rows_mut(ambient_offset, manifold.ambient_size())
                .copy_from(&component);
        }
        result
    }

    fn minus(&self, y: na::DVectorView<f64>, x: na::DVectorView<f64>) -> na::DVector<f64> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(y.len(), self.ambient_size);
        let mut result = na::DVector::zeros(self.tangent_size);
        for (index, manifold) in self.manifolds.iter().enumerate() {
            let ambient_offset = self.ambient_offsets[index];
            let tangent_offset = self.tangent_offsets[index];
            let component = manifold.minus_f64(
                y.rows(ambient_offset, manifold.ambient_size()),
                x.rows(ambient_offset, manifold.ambient_size()),
            );
            result
                .rows_mut(tangent_offset, manifold.tangent_size())
                .copy_from(&component);
        }
        result
    }
}

impl AutoDiffManifold<DualDVec64> for ProductManifold {
    fn plus(
        &self,
        x: na::DVectorView<DualDVec64>,
        delta: na::DVectorView<DualDVec64>,
    ) -> na::DVector<DualDVec64> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(delta.len(), self.tangent_size);
        let mut result = na::DVector::zeros(self.ambient_size);
        for (index, manifold) in self.manifolds.iter().enumerate() {
            let ambient_offset = self.ambient_offsets[index];
            let tangent_offset = self.tangent_offsets[index];
            let component = manifold.plus_dual(
                x.rows(ambient_offset, manifold.ambient_size()),
                delta.rows(tangent_offset, manifold.tangent_size()),
            );
            result
                .rows_mut(ambient_offset, manifold.ambient_size())
                .copy_from(&component);
        }
        result
    }

    fn minus(
        &self,
        y: na::DVectorView<DualDVec64>,
        x: na::DVectorView<DualDVec64>,
    ) -> na::DVector<DualDVec64> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(y.len(), self.ambient_size);
        let mut result = na::DVector::zeros(self.tangent_size);
        for (index, manifold) in self.manifolds.iter().enumerate() {
            let ambient_offset = self.ambient_offsets[index];
            let tangent_offset = self.tangent_offsets[index];
            let component = manifold.minus_dual(
                y.rows(ambient_offset, manifold.ambient_size()),
                x.rows(ambient_offset, manifold.ambient_size()),
            );
            result
                .rows_mut(tangent_offset, manifold.tangent_size())
                .copy_from(&component);
        }
        result
    }
}

impl Manifold for ProductManifold {
    fn ambient_size(&self) -> usize {
        self.ambient_size
    }

    fn tangent_size(&self) -> usize {
        self.tangent_size
    }
}
