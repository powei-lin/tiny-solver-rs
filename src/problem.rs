use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use faer::sparse::{SparseColMat, SymbolicSparseColMat, ValuesOrder};
use faer_ext::IntoFaer;
use nalgebra as na;
use rayon::prelude::*;

use crate::manifold::Manifold;
use crate::parameter_block::ParameterBlock;
use crate::{factors, loss_functions, residual_block};

type ResidualBlockId = usize;

pub struct Problem {
    pub total_residual_dimension: usize,
    residual_id_count: usize,
    residual_blocks: HashMap<ResidualBlockId, residual_block::ResidualBlock>,
    pub fixed_variable_indexes: HashMap<String, HashSet<usize>>,
    pub variable_bounds: HashMap<String, HashMap<usize, (f64, f64)>>,
    pub variable_manifold: HashMap<String, Arc<dyn Manifold + Sync + Send>>,
}
impl Default for Problem {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SymbolicStructure {
    pattern: SymbolicSparseColMat<usize>,
    order: ValuesOrder<usize>,
}

type JacobianValue = f64;

impl Problem {
    pub fn new() -> Problem {
        Problem {
            total_residual_dimension: 0,
            residual_id_count: 0,
            residual_blocks: HashMap::new(),
            fixed_variable_indexes: HashMap::new(),
            variable_bounds: HashMap::new(),
            variable_manifold: HashMap::new(),
        }
    }

    pub fn build_symbolic_structure(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        total_variable_dimension: usize,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
    ) -> SymbolicStructure {
        let mut indices = Vec::<(usize, usize)>::new();

        self.residual_blocks.iter().for_each(|(_, residual_block)| {
            let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
            let mut count_variable_local_idx: usize = 0;
            for var_key in &residual_block.variable_key_list {
                if let Some(param) = parameter_blocks.get(var_key) {
                    variable_local_idx_size_list
                        .push((count_variable_local_idx, param.tangent_size()));
                    count_variable_local_idx += param.tangent_size();
                };
            }
            for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
                if let Some(variable_global_idx) = variable_name_to_col_idx_dict.get(var_key) {
                    let (_, var_size) = variable_local_idx_size_list[i];
                    for row_idx in 0..residual_block.dim_residual {
                        for col_idx in 0..var_size {
                            let global_row_idx = residual_block.residual_row_start_idx + row_idx;
                            let global_col_idx = variable_global_idx + col_idx;
                            indices.push((global_row_idx, global_col_idx));
                        }
                    }
                }
            }
        });
        let start = std::time::Instant::now();
        let (s, o) = SymbolicSparseColMat::try_new_from_indices(
            self.total_residual_dimension,
            total_variable_dimension,
            &indices,
        )
        .unwrap();
        log::trace!("Built symbolic matrix: {:?}", start.elapsed());
        SymbolicStructure {
            pattern: s,
            order: o,
        }
    }

    pub fn get_variable_name_to_col_idx_dict(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
    ) -> HashMap<String, usize> {
        let mut count_col_idx = 0;
        let mut variable_name_to_col_idx_dict = HashMap::new();
        parameter_blocks
            .iter()
            .for_each(|(param_name, param_block)| {
                variable_name_to_col_idx_dict.insert(param_name.to_owned(), count_col_idx);
                count_col_idx += param_block.tangent_size();
            });
        variable_name_to_col_idx_dict
    }
    pub fn add_residual_block(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn factors::FactorImpl + Send>,
        loss_func: Option<Box<dyn loss_functions::Loss + Send>>,
    ) -> ResidualBlockId {
        self.residual_blocks.insert(
            self.residual_id_count,
            residual_block::ResidualBlock::new(
                self.residual_id_count,
                dim_residual,
                self.total_residual_dimension,
                variable_key_size_list,
                factor,
                loss_func,
            ),
        );
        let block_id = self.residual_id_count;
        self.residual_id_count += 1;

        self.total_residual_dimension += dim_residual;
        
        block_id
    }
    pub fn remove_residual_block(
        &mut self,
        block_id: ResidualBlockId
    ) -> Option<residual_block::ResidualBlock> {
            if let Some(residual_block) = self.residual_blocks.remove(&block_id) {
                self.total_residual_dimension -= residual_block.dim_residual;
                Some(residual_block)
            } else {
                None
            }
    }
    pub fn fix_variable(&mut self, var_to_fix: &str, idx: usize) {
        if let Some(var_mut) = self.fixed_variable_indexes.get_mut(var_to_fix) {
            var_mut.insert(idx);
        } else {
            self.fixed_variable_indexes
                .insert(var_to_fix.to_owned(), HashSet::from([idx]));
        }
    }
    pub fn unfix_variable(&mut self, var_to_unfix: &str) {
        self.fixed_variable_indexes.remove(var_to_unfix);
    }
    pub fn set_variable_bounds(
        &mut self,
        var_to_bound: &str,
        idx: usize,
        lower_bound: f64,
        upper_bound: f64,
    ) {
        if lower_bound > upper_bound {
            log::error!("lower bound is larger than upper bound");
        } else if let Some(var_mut) = self.variable_bounds.get_mut(var_to_bound) {
            var_mut.insert(idx, (lower_bound, upper_bound));
        } else {
            self.variable_bounds.insert(
                var_to_bound.to_owned(),
                HashMap::from([(idx, (lower_bound, upper_bound))]),
            );
        }
    }
    pub fn set_variable_manifold(
        &mut self,
        var_name: &str,
        manifold: Arc<dyn Manifold + Sync + Send>,
    ) {
        self.variable_manifold
            .insert(var_name.to_string(), manifold);
    }
    pub fn remove_variable_bounds(&mut self, var_to_unbound: &str) {
        self.variable_bounds.remove(var_to_unbound);
    }
    pub fn initialize_parameter_blocks(
        &self,
        initial_values: &HashMap<String, na::DVector<f64>>,
    ) -> HashMap<String, ParameterBlock> {
        let parameter_blocks: HashMap<String, ParameterBlock> = initial_values
            .iter()
            .map(|(k, v)| {
                let mut p_block = ParameterBlock::from_vec(v.clone());
                if let Some(indexes) = self.fixed_variable_indexes.get(k) {
                    p_block.fixed_variables = indexes.clone();
                }
                if let Some(bounds) = self.variable_bounds.get(k) {
                    p_block.variable_bounds = bounds.clone();
                }
                if let Some(manifold) = self.variable_manifold.get(k) {
                    p_block.manifold = Some(manifold.clone())
                }

                (k.to_owned(), p_block)
            })
            .collect();
        parameter_blocks
    }

    pub fn compute_residuals(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        with_loss_fn: bool,
    ) -> faer::Mat<f64> {
        let total_residual = Arc::new(Mutex::new(na::DVector::<f64>::zeros(
            self.total_residual_dimension,
        )));
        self.residual_blocks
            .par_iter()
            .for_each(|(_, residual_block)| {
                self.compute_residual_impl(
                    residual_block,
                    parameter_blocks,
                    &total_residual,
                    with_loss_fn,
                )
            });
        let total_residual = Arc::try_unwrap(total_residual)
            .unwrap()
            .into_inner()
            .unwrap();

        total_residual.view_range(.., ..).into_faer().to_owned()
    }

    pub fn compute_residual_and_jacobian(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
        // multi
        let total_residual = Arc::new(Mutex::new(na::DVector::<f64>::zeros(
            self.total_residual_dimension,
        )));

        let jacobian_lists: Vec<JacobianValue> = self
            .residual_blocks
            .par_iter()
            .map(|(_, residual_block)| {
                self.compute_residual_and_jacobian_impl(
                    residual_block,
                    parameter_blocks,
                    variable_name_to_col_idx_dict,
                    &total_residual,
                )
            })
            .flatten()
            .collect();

        let total_residual = Arc::try_unwrap(total_residual)
            .unwrap()
            .into_inner()
            .unwrap();

        let residual_faer = total_residual.view_range(.., ..).into_faer().to_owned();
        let jacobian_faer = SparseColMat::new_from_order_and_values(
            symbolic_structure.pattern.clone(),
            &symbolic_structure.order,
            jacobian_lists.as_slice(),
        )
        .unwrap();
        (residual_faer, jacobian_faer)
    }

    fn compute_residual_impl(
        &self,
        residual_block: &crate::ResidualBlock,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        total_residual: &Arc<Mutex<na::DVector<f64>>>,
        with_loss_fn: bool,
    ) {
        let mut params = Vec::new();
        for var_key in &residual_block.variable_key_list {
            if let Some(param) = parameter_blocks.get(var_key) {
                params.push(param);
            };
        }
        let res = residual_block.residual(&params, with_loss_fn);

        {
            let mut total_residual = total_residual.lock().unwrap();
            total_residual
                .rows_mut(
                    residual_block.residual_row_start_idx,
                    residual_block.dim_residual,
                )
                .copy_from(&res);
        }
    }

    fn compute_residual_and_jacobian_impl(
        &self,
        residual_block: &crate::ResidualBlock,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
        total_residual: &Arc<Mutex<na::DVector<f64>>>,
    ) -> Vec<JacobianValue> {
        let mut params = Vec::new();
        let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
        let mut count_variable_local_idx: usize = 0;
        for var_key in &residual_block.variable_key_list {
            if let Some(param) = parameter_blocks.get(var_key) {
                params.push(param);
                variable_local_idx_size_list.push((count_variable_local_idx, param.tangent_size()));
                count_variable_local_idx += param.tangent_size();
            };
        }
        let (res, jac) = residual_block.residual_and_jacobian(&params);
        {
            let mut total_residual = total_residual.lock().unwrap();
            total_residual
                .rows_mut(
                    residual_block.residual_row_start_idx,
                    residual_block.dim_residual,
                )
                .copy_from(&res);
        }

        let mut local_jacobian_list = Vec::new();

        for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
            if variable_name_to_col_idx_dict.contains_key(var_key) {
                let (variable_local_idx, var_size) = variable_local_idx_size_list[i];
                let variable_jac = jac.view((0, variable_local_idx), (jac.shape().0, var_size));
                for row_idx in 0..jac.shape().0 {
                    for col_idx in 0..var_size {
                        local_jacobian_list.push(variable_jac[(row_idx, col_idx)]);
                    }
                }
            } else {
                panic!(
                    "Missing key {} in variable-to-column-index mapping",
                    var_key
                );
            }
        }

        local_jacobian_list
    }
}
