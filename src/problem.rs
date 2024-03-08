use num_dual;

extern crate nalgebra as na;
// use std::collections::HashMap;
pub struct Problem {
    _dim_variable: usize,
    _dim_residual: usize,
    // residual_blocks: Vec<ResidualBlock>,
    // variable_addr_to_col_idx_dict: HashMap<usize, usize>,
    // col_idx_to_variable_dict: HashMap<usize, usize>,
}

pub struct ResidualBlock {
    pub dim_residual: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub residual_func:
        Box<dyn Fn(&Vec<na::DVector<num_dual::DualDVec64>>) -> na::DVector<num_dual::DualDVec64>>,
}
impl ResidualBlock {
    pub fn jacobian(self, params: &Vec<na::DVector<f64>>) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let dim_variable = params.iter().fold(0, |acc, x| acc + x.shape().0);
        println!("dim_variable {}", dim_variable);
        (
            na::DVector::<f64>::zeros(self.dim_residual),
            na::DMatrix::<f64>::zeros(self.dim_residual, self.dim_residual),
        )
    }
}

// impl ResidualBlock {}

// trait Factor {
// fn cost_function(
//     _params: na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_PARAMETERS>,
// ) -> na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_RESIDUALS>
// }
