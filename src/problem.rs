use na::Dyn;
use num_dual;

extern crate nalgebra as na;
use crate::residual_block;
use std::collections::HashMap;
pub struct Problem {
    pub total_variable_dimension: usize,
    pub total_residual_dimension: usize,
    residual_blocks: Vec<residual_block::ResidualBlock>,
    variable_name_to_col_idx_dict: HashMap<String, usize>,
    // col_idx_to_variable_dict: HashMap<usize, usize>,
}
impl Problem {
    pub fn new() -> Problem {
        Problem {
            total_variable_dimension: 0,
            total_residual_dimension: 0,
            residual_blocks: Vec::<residual_block::ResidualBlock>::new(),
            variable_name_to_col_idx_dict: HashMap::<String, usize>::new(),
        }
    }
    pub fn add_residual_block(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: Vec<(String, usize)>,
        residual_func: Box<
            dyn Fn(&Vec<na::DVector<num_dual::DualDVec64>>) -> na::DVector<num_dual::DualDVec64>,
        >,
    ) {
        self.residual_blocks.push(residual_block::ResidualBlock {
            dim_residual: dim_residual,
            residual_row_start_idx: self.total_residual_dimension,
            variable_key_list: variable_key_size_list
                .iter()
                .map(|(x, _)| x.to_string())
                .collect(),
            residual_func: residual_func,
        });
        for (key, variable_dimesion) in variable_key_size_list {
            if !self.variable_name_to_col_idx_dict.contains_key(&key) {
                self.variable_name_to_col_idx_dict
                    .insert(key, self.total_variable_dimension);
                self.total_variable_dimension += variable_dimesion;
            }
        }
        self.total_residual_dimension += dim_residual;
    }
    pub fn combine_variables(
        self,
        variable_key_value_map: &HashMap<String, na::DVector<f64>>,
    ) -> na::DVector<f64> {
        let mut combined_variables = na::DVector::<f64>::zeros(self.total_variable_dimension);
        for (k, v) in variable_key_value_map {
            if let Some(col_idx) = self.variable_name_to_col_idx_dict.get(k) {
                combined_variables
                    .rows_mut(*col_idx, v.shape().0)
                    .copy_from(v);
            };
        }
        return combined_variables;
    }
}
