use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use faer::sparse::SparseColMat;
use faer_ext::IntoFaer;
use nalgebra as na;
use rayon::prelude::*;

use crate::{factors, loss_functions, residual_block};

pub struct Problem {
    pub total_variable_dimension: usize,
    pub total_residual_dimension: usize,
    residual_blocks: Vec<residual_block::ResidualBlock>,
    pub variable_name_to_col_idx_dict: HashMap<String, usize>,
    pub fixed_variable_set: HashSet<String>,
    _has_py_factor: bool,
}
impl Default for Problem {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem {
    pub fn new() -> Problem {
        Problem {
            total_variable_dimension: 0,
            total_residual_dimension: 0,
            residual_blocks: Vec::<residual_block::ResidualBlock>::new(),
            variable_name_to_col_idx_dict: HashMap::<String, usize>::new(),
            _has_py_factor: false,
            fixed_variable_set: HashSet::new(),
        }
    }
    pub fn add_residual_block(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: Vec<(String, usize)>,
        factor: Box<dyn factors::Factor + Send>,
        loss_func: Option<Box<dyn loss_functions::Loss + Send>>,
    ) {
        self.residual_blocks.push(residual_block::ResidualBlock {
            dim_residual,
            residual_row_start_idx: self.total_residual_dimension,
            variable_key_list: variable_key_size_list
                .iter()
                .map(|(x, _)| x.to_string())
                .collect(),
            factor,
            loss_func,
        });
        for (key, variable_dimesion) in variable_key_size_list {
            if let std::collections::hash_map::Entry::Vacant(e) =
                self.variable_name_to_col_idx_dict.entry(key)
            {
                e.insert(self.total_variable_dimension);
                self.total_variable_dimension += variable_dimesion;
            }
        }
        self.total_residual_dimension += dim_residual;
    }
    pub fn fixed_variable(&mut self, var_to_fix: &str) {
        self.fixed_variable_set.insert(var_to_fix.to_owned());
    }
    pub fn unfixed_variable(&mut self, var_to_fix: &str) {
        self.fixed_variable_set.remove(var_to_fix);
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
        combined_variables
    }
    pub fn compute_residual_and_jacobian(
        &self,
        variable_key_value_map: &HashMap<String, na::DVector<f64>>,
    ) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
        // multi
        let total_residual: Arc<
            Mutex<
                na::Matrix<f64, na::Dyn, na::Const<1>, na::VecStorage<f64, na::Dyn, na::Const<1>>>,
            >,
        > = Arc::new(Mutex::new(na::DVector::<f64>::zeros(
            self.total_residual_dimension,
        )));
        let jacobian_list: Arc<Mutex<Vec<(usize, usize, f64)>>> =
            Arc::new(Mutex::new(Vec::<(usize, usize, f64)>::new()));

        if self._has_py_factor {
            self.residual_blocks.iter().for_each(|residual_block| {
                self.compute_residual_and_jacobian_impl(
                    residual_block,
                    variable_key_value_map,
                    &total_residual,
                    &jacobian_list,
                )
            });
        } else {
            self.residual_blocks.par_iter().for_each(|residual_block| {
                self.compute_residual_and_jacobian_impl(
                    residual_block,
                    variable_key_value_map,
                    &total_residual,
                    &jacobian_list,
                )
            });
        }

        let total_residual = Arc::try_unwrap(total_residual)
            .unwrap()
            .into_inner()
            .unwrap();
        let jacobian_list = Arc::try_unwrap(jacobian_list)
            .unwrap()
            .into_inner()
            .unwrap();
        // end

        let residual_faer = total_residual.view_range(.., ..).into_faer().to_owned();
        let jacobian_faer = SparseColMat::try_new_from_triplets(
            self.total_residual_dimension,
            self.total_variable_dimension,
            &jacobian_list,
        )
        .unwrap();
        (residual_faer, jacobian_faer)
    }
    fn compute_residual_and_jacobian_impl(
        &self,
        residual_block: &crate::ResidualBlock,
        variable_key_value_map: &HashMap<String, na::DVector<f64>>,
        total_residual: &Arc<
            Mutex<
                na::Matrix<f64, na::Dyn, na::Const<1>, na::VecStorage<f64, na::Dyn, na::Const<1>>>,
            >,
        >,
        jacobian_list: &Arc<Mutex<Vec<(usize, usize, f64)>>>,
    ) {
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

        {
            let mut total_residual = total_residual.lock().unwrap();
            total_residual
                .rows_mut(
                    residual_block.residual_row_start_idx,
                    residual_block.dim_residual,
                )
                .copy_from(&res);
        }

        for (i, vk) in residual_block.variable_key_list.iter().enumerate() {
            if let Some(variable_global_idx) = self.variable_name_to_col_idx_dict.get(vk) {
                let (variable_local_idx, var_size) = variable_local_idx_size_list[i];
                let variable_jac = jac.view((0, variable_local_idx), (jac.shape().0, var_size));
                let mut local_jacobian_list = Vec::new();
                for row_idx in 0..jac.shape().0 {
                    for col_idx in 0..var_size {
                        let global_row_idx = residual_block.residual_row_start_idx + row_idx;
                        let global_col_idx = variable_global_idx + col_idx;
                        let value = variable_jac[(row_idx, col_idx)];
                        local_jacobian_list.push((global_row_idx, global_col_idx, value));
                    }
                }
                let mut jacobian_list = jacobian_list.lock().unwrap();
                jacobian_list.extend(local_jacobian_list);
            }
        }
    }
    pub fn has_py_factor(&mut self) {
        self._has_py_factor = true
    }
}
