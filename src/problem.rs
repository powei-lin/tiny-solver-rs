extern crate nalgebra as na;
// use std::collections::HashMap;
pub struct Problem{
    _dim_variable: usize,
    _dim_residual: usize,
    // residual_blocks: Vec<ResidualBlock>,
    // variable_addr_to_col_idx_dict: HashMap<usize, usize>,
    // col_idx_to_variable_dict: HashMap<usize, usize>,
}

// struct ResidualBlock{
    // dim_residual: usize,
    // residual_row_start_idx: usize,
    // variable_col_start_index_list: Vec<usize>,
    // residual_func: callable
    // jac_func = None
// }

// trait Factor {
    // fn cost_function(
    //     _params: na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_PARAMETERS>,
    // ) -> na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_RESIDUALS>
// }