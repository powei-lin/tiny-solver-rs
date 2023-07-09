extern crate nalgebra as na;

use na::SMatrix;

pub trait CostFuncNA<T, const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    fn apply(
        _params: &SMatrix<T, NUM_PARAMETERS, 1>,
        _residual: &mut SMatrix<T, NUM_RESIDUALS, 1>,
        _jacobian: Option<&mut SMatrix<T, NUM_RESIDUALS, NUM_PARAMETERS>>,
    ) {
    }
    fn num_params() -> usize {
        NUM_PARAMETERS
    }
    fn num_residuals() -> usize {
        NUM_RESIDUALS
    }
}
