use na::Dyn;
use num_dual;

extern crate nalgebra as na;
use crate::residual_block;
use std::collections::HashMap;
pub struct Problem {
    pub total_variable_dimension: usize,
    pub total_residual_dimension: usize,
    residual_blocks: Vec<residual_block::ResidualBlock>,
    pub variable_name_to_col_idx_dict: HashMap<String, usize>,
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
        &self,
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
    pub fn compute_residual_and_jacobian(
        &self,
        variable_key_value_map: &HashMap<String, na::DVector<f64>>,
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let mut total_residual = na::DVector::<f64>::zeros(self.total_residual_dimension);
        // TODO sparse
        let mut total_jacobian =
            na::DMatrix::<f64>::zeros(self.total_residual_dimension, self.total_variable_dimension);
        for residual_block in &self.residual_blocks {
            let mut params = Vec::<na::DVector<f64>>::new();
            let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
            let mut count_variable_local_idx: usize = 0;
            for vk in &residual_block.variable_key_list {
                if let Some(param) = variable_key_value_map.get(vk) {
                    params.push(param.clone());
                    variable_local_idx_size_list.push((count_variable_local_idx, param.shape().0));
                    count_variable_local_idx += param.shape().0;
                };
            }
            let (res, jac) = residual_block.jacobian(&params);
            total_residual
                .rows_mut(
                    residual_block.residual_row_start_idx,
                    residual_block.dim_residual,
                )
                .copy_from(&res);
            for (i, vk) in residual_block.variable_key_list.iter().enumerate() {
                if let Some(variable_global_idx) = self.variable_name_to_col_idx_dict.get(vk) {
                    let (variable_local_idx, var_size) = variable_local_idx_size_list[i];
                    let variable_jac = jac.view((0, variable_local_idx), (jac.shape().0, var_size));
                    total_jacobian
                        .view_mut(
                            (residual_block.residual_row_start_idx, *variable_global_idx),
                            variable_jac.shape(),
                        )
                        .copy_from(&variable_jac);
                }
            }
        }
        (total_residual, total_jacobian)
    }
}
