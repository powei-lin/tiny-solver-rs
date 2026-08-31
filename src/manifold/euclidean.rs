use nalgebra as na;

use super::{AutoDiffManifold, Manifold};

#[derive(Debug, Clone)]
pub struct EuclideanManifold {
    size: usize,
}

impl EuclideanManifold {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl<T: na::RealField> AutoDiffManifold<T> for EuclideanManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.size);
        assert_eq!(delta.len(), self.size);
        x + delta
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.size);
        assert_eq!(y.len(), self.size);
        y - x
    }
}

impl Manifold for EuclideanManifold {
    fn ambient_size(&self) -> usize {
        self.size
    }

    fn tangent_size(&self) -> usize {
        self.size
    }
}
