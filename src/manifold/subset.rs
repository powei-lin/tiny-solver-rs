use nalgebra as na;

use super::{AutoDiffManifold, Manifold};

#[derive(Debug, Clone)]
pub struct SubsetManifold {
    constant: Vec<bool>,
    tangent_size: usize,
}

impl SubsetManifold {
    pub fn new(ambient_size: usize, constant_parameters: &[usize]) -> Self {
        let mut constant = vec![false; ambient_size];
        for &index in constant_parameters {
            assert!(
                index < ambient_size,
                "constant parameter index is out of bounds"
            );
            assert!(
                !constant[index],
                "constant parameter indices must be unique"
            );
            constant[index] = true;
        }
        Self {
            constant,
            tangent_size: ambient_size - constant_parameters.len(),
        }
    }
}

impl<T: na::RealField> AutoDiffManifold<T> for SubsetManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.constant.len());
        assert_eq!(delta.len(), self.tangent_size);
        let mut delta_index = 0;
        na::DVector::from_fn(self.constant.len(), |index, _| {
            if self.constant[index] {
                x[index].clone()
            } else {
                let value = x[index].clone() + delta[delta_index].clone();
                delta_index += 1;
                value
            }
        })
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.constant.len());
        assert_eq!(y.len(), self.constant.len());
        let values = (0..self.constant.len())
            .filter(|&index| !self.constant[index])
            .map(|index| y[index].clone() - x[index].clone());
        na::DVector::from_iterator(self.tangent_size, values)
    }
}

impl Manifold for SubsetManifold {
    fn ambient_size(&self) -> usize {
        self.constant.len()
    }

    fn tangent_size(&self) -> usize {
        self.tangent_size
    }
}
