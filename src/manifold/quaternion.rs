use nalgebra as na;

use super::{AutoDiffManifold, Manifold};

fn product<T: na::RealField>(left: [T; 4], right: [T; 4]) -> [T; 4] {
    let [left_w, left_x, left_y, left_z] = left;
    let [right_w, right_x, right_y, right_z] = right;
    [
        left_w.clone() * right_w.clone()
            - left_x.clone() * right_x.clone()
            - left_y.clone() * right_y.clone()
            - left_z.clone() * right_z.clone(),
        left_w.clone() * right_x.clone()
            + left_x.clone() * right_w.clone()
            + left_y.clone() * right_z.clone()
            - left_z.clone() * right_y.clone(),
        left_w.clone() * right_y.clone() - left_x.clone() * right_z.clone()
            + left_y.clone() * right_w.clone()
            + left_z.clone() * right_x.clone(),
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    ]
}

fn plus<T: na::RealField>(quaternion: [T; 4], delta: na::DVectorView<T>) -> [T; 4] {
    assert_eq!(delta.len(), 3);
    let norm2 = delta.norm_squared();
    let (delta_w, scale) = if norm2 < T::from_f64(1e-12).unwrap() {
        (
            T::one() - norm2.clone() / T::from_f64(2.0).unwrap(),
            T::one() - norm2 / T::from_f64(6.0).unwrap(),
        )
    } else {
        let norm = norm2.sqrt();
        (norm.clone().cos(), norm.clone().sin() / norm)
    };
    let delta_quaternion = [
        delta_w,
        scale.clone() * delta[0].clone(),
        scale.clone() * delta[1].clone(),
        scale * delta[2].clone(),
    ];
    product(delta_quaternion, quaternion)
}

fn minus<T: na::RealField>(y: [T; 4], x: [T; 4]) -> na::DVector<T> {
    let conjugate_x = [x[0].clone(), -x[1].clone(), -x[2].clone(), -x[3].clone()];
    let difference = product(y, conjugate_x);
    let vector = na::dvector![
        difference[1].clone(),
        difference[2].clone(),
        difference[3].clone()
    ];
    let norm2 = vector.norm_squared();
    let scale = if norm2 < T::from_f64(1e-12).unwrap() {
        let w2 = difference[0].clone() * difference[0].clone();
        T::one() / difference[0].clone()
            - norm2 / (T::from_f64(3.0).unwrap() * w2 * difference[0].clone())
    } else {
        let norm = norm2.sqrt();
        norm.clone().atan2(difference[0].clone()) / norm
    };
    vector * scale
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuaternionManifold;

impl<T: na::RealField> AutoDiffManifold<T> for QuaternionManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), 4);
        na::DVector::from_vec(Vec::from(plus(
            [x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()],
            delta,
        )))
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), 4);
        assert_eq!(y.len(), 4);
        minus(
            [y[0].clone(), y[1].clone(), y[2].clone(), y[3].clone()],
            [x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()],
        )
    }
}

impl Manifold for QuaternionManifold {
    fn ambient_size(&self) -> usize {
        4
    }

    fn tangent_size(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EigenQuaternionManifold;

impl<T: na::RealField> AutoDiffManifold<T> for EigenQuaternionManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), 4);
        let result = plus(
            [x[3].clone(), x[0].clone(), x[1].clone(), x[2].clone()],
            delta,
        );
        na::dvector![
            result[1].clone(),
            result[2].clone(),
            result[3].clone(),
            result[0].clone()
        ]
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), 4);
        assert_eq!(y.len(), 4);
        minus(
            [y[3].clone(), y[0].clone(), y[1].clone(), y[2].clone()],
            [x[3].clone(), x[0].clone(), x[1].clone(), x[2].clone()],
        )
    }
}

impl Manifold for EigenQuaternionManifold {
    fn ambient_size(&self) -> usize {
        4
    }

    fn tangent_size(&self) -> usize {
        3
    }
}
