use nalgebra as na;

pub trait LieGroup<T: na::RealField> {
    const TANGENT_SIZE: usize;

    fn exp(xi: na::DVectorView<T>) -> Self;
}
