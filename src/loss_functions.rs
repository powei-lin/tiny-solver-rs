use faer::prelude::*;
use pyo3::prelude::*;

pub trait Loss: Send + Sync {
    fn weight_in_place(&self);
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct HuberLoss {}

impl Loss for HuberLoss {
    fn weight_in_place(&self) {
        // pub fn huberloss_in_place(mat: &mut Mat<f64>){
        //     let sqrtw = std::sqrt(weight(b.norm()));
        //     b *= sqrtw;
    }
}
