use nalgebra as na;

use super::{AutoDiffManifold, Manifold};

#[derive(Debug, Clone)]
pub struct SphereManifold {
    ambient_size: usize,
}

impl SphereManifold {
    pub fn new(ambient_size: usize) -> Self {
        assert!(
            ambient_size > 1,
            "sphere ambient size must be greater than one"
        );
        Self { ambient_size }
    }

    pub(crate) fn householder<T: na::RealField>(x: na::DVectorView<T>) -> (na::DVector<T>, T) {
        let tangent_size = x.len() - 1;
        let sigma = x.rows(0, tangent_size).norm_squared();
        let pivot = x[tangent_size].clone();
        let mut vector = x.into_owned();
        vector[tangent_size] = T::one();
        if sigma <= T::from_f64(f64::EPSILON).unwrap() {
            let beta = if pivot < T::zero() {
                T::from_f64(2.0).unwrap()
            } else {
                T::zero()
            };
            return (vector, beta);
        }

        let mu = (pivot.clone() * pivot.clone() + sigma.clone()).sqrt();
        let vector_pivot = if pivot <= T::zero() {
            pivot - mu
        } else {
            -sigma.clone() / (pivot + mu)
        };
        let vector_pivot2 = vector_pivot.clone() * vector_pivot.clone();
        let beta = T::from_f64(2.0).unwrap() * vector_pivot2.clone() / (sigma + vector_pivot2);
        for value in vector.rows_mut(0, tangent_size).iter_mut() {
            *value /= vector_pivot.clone();
        }
        (vector, beta)
    }

    pub(crate) fn apply_householder<T: na::RealField>(
        vector: &na::DVector<T>,
        beta: T,
        value: na::DVector<T>,
    ) -> na::DVector<T> {
        let projection = beta * vector.dot(&value);
        value - vector * projection
    }
}

impl<T: na::RealField> AutoDiffManifold<T> for SphereManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(delta.len(), self.ambient_size - 1);
        let (householder, beta) = Self::householder(x.clone());
        let norm2 = delta.norm_squared();
        let (last, scale) = if norm2 < T::from_f64(1e-12).unwrap() {
            (
                T::one() - norm2.clone() / T::from_f64(2.0).unwrap(),
                T::one() - norm2 / T::from_f64(6.0).unwrap(),
            )
        } else {
            let norm = norm2.sqrt();
            (norm.clone().cos(), norm.clone().sin() / norm)
        };
        let mut update = na::DVector::zeros(self.ambient_size);
        update
            .rows_mut(0, self.ambient_size - 1)
            .copy_from(&(delta * scale));
        update[self.ambient_size - 1] = last;
        Self::apply_householder(&householder, beta, update) * x.norm()
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(y.len(), self.ambient_size);
        let (householder, beta) = Self::householder(x.clone());
        let transformed = Self::apply_householder(&householder, beta, y.into_owned()) / x.norm();
        let tangent_size = self.ambient_size - 1;
        let head = transformed.rows(0, tangent_size).into_owned();
        let head_norm2 = head.norm_squared();
        if head_norm2 <= T::from_f64(f64::EPSILON).unwrap() {
            if transformed[tangent_size] < T::zero() {
                let mut result = na::DVector::zeros(tangent_size);
                result[tangent_size - 1] = T::from_f64(std::f64::consts::PI).unwrap();
                return result;
            }
            let last = transformed[tangent_size].clone();
            let last2 = last.clone() * last.clone();
            let scale =
                T::one() / last.clone() - head_norm2 / (T::from_f64(3.0).unwrap() * last2 * last);
            return head * scale;
        }
        let head_norm = head_norm2.sqrt();
        let scale = head_norm.clone().atan2(transformed[tangent_size].clone()) / head_norm;
        head * scale
    }
}

impl Manifold for SphereManifold {
    fn ambient_size(&self) -> usize {
        self.ambient_size
    }

    fn tangent_size(&self) -> usize {
        self.ambient_size - 1
    }
}
