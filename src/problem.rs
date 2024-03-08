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
    // residual_func: dyn FnOnce(Vec<na::DVector<num_dual::DualDVec64>>) -> num_dual::DualDVec64,
}
// impl ResidualBlock {}

// trait Factor {
// fn cost_function(
//     _params: na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_PARAMETERS>,
// ) -> na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_RESIDUALS>
// }
