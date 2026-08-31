use nalgebra as na;

use super::{AutoDiffManifold, Manifold, SphereManifold};

#[derive(Debug, Clone)]
pub struct LineManifold {
    space_dimension: usize,
}

impl LineManifold {
    pub fn new(space_dimension: usize) -> Self {
        assert!(
            space_dimension > 1,
            "line space dimension must be greater than one"
        );
        Self { space_dimension }
    }
}

impl<T: na::RealField> AutoDiffManifold<T> for LineManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        let tangent_dimension = self.space_dimension - 1;
        assert_eq!(x.len(), 2 * self.space_dimension);
        assert_eq!(delta.len(), 2 * tangent_dimension);

        let origin = x.rows(0, self.space_dimension);
        let direction = x.rows(self.space_dimension, self.space_dimension);
        let delta_origin = delta.rows(0, tangent_dimension);
        let delta_direction = delta.rows(tangent_dimension, tangent_dimension);
        let updated_direction =
            SphereManifold::new(self.space_dimension).plus(direction.clone(), delta_direction);

        let (householder, beta) = SphereManifold::householder(direction);
        let mut ambient_origin_delta = na::DVector::zeros(self.space_dimension);
        ambient_origin_delta
            .rows_mut(0, tangent_dimension)
            .copy_from(&delta_origin);
        let updated_origin =
            origin + SphereManifold::apply_householder(&householder, beta, ambient_origin_delta);

        let mut result = na::DVector::zeros(2 * self.space_dimension);
        result
            .rows_mut(0, self.space_dimension)
            .copy_from(&updated_origin);
        result
            .rows_mut(self.space_dimension, self.space_dimension)
            .copy_from(&updated_direction);
        result
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        let tangent_dimension = self.space_dimension - 1;
        assert_eq!(x.len(), 2 * self.space_dimension);
        assert_eq!(y.len(), 2 * self.space_dimension);

        let x_origin = x.rows(0, self.space_dimension);
        let x_direction = x.rows(self.space_dimension, self.space_dimension);
        let y_origin = y.rows(0, self.space_dimension);
        let y_direction = y.rows(self.space_dimension, self.space_dimension);
        let direction_delta =
            SphereManifold::new(self.space_dimension).minus(y_direction, x_direction.clone());

        let (householder, beta) = SphereManifold::householder(x_direction);
        let ambient_origin_delta = y_origin - x_origin;
        let transformed_origin_delta =
            SphereManifold::apply_householder(&householder, beta, ambient_origin_delta);

        let mut result = na::DVector::zeros(2 * tangent_dimension);
        result
            .rows_mut(0, tangent_dimension)
            .copy_from(&transformed_origin_delta.rows(0, tangent_dimension));
        result
            .rows_mut(tangent_dimension, tangent_dimension)
            .copy_from(&direction_delta);
        result
    }
}

impl Manifold for LineManifold {
    fn ambient_size(&self) -> usize {
        2 * self.space_dimension
    }

    fn tangent_size(&self) -> usize {
        2 * (self.space_dimension - 1)
    }
}
