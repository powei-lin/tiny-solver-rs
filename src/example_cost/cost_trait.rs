use faer_core::{Entity, Mat};
pub trait CostFunc<T: Entity, const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    fn apply(
        _params: &[T; NUM_PARAMETERS],
        _residual: &mut Mat<T>,
        _jacobian: Option<&mut Mat<T>>,
    ) {
    }
    fn num_params() -> usize {
        NUM_PARAMETERS
    }
    fn num_residuals() -> usize {
        NUM_RESIDUALS
    }
}
