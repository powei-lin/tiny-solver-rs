extern crate nalgebra as na;

use num_traits;
use na::{SMatrix, Scalar};
pub trait CostFunc<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    type T: Scalar + num_traits::identities::Zero;
    const NUM_PARAMETERS: usize = NUM_PARAMETERS;
    const NUM_RESIDUALS: usize = NUM_RESIDUALS;
    fn apply(
        _params: &mut SMatrix<Self::T, NUM_PARAMETERS, 1>,
        _residual: &mut SMatrix<Self::T, NUM_RESIDUALS, 1>,
        _jacobian: Option<&mut SMatrix<Self::T, NUM_RESIDUALS, NUM_PARAMETERS>>,
    ) {
        unimplemented!()
    }
}
