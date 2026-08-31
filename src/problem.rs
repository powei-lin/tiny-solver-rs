use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer_ext::IntoFaer;
use nalgebra as na;
use rayon::prelude::*;

use crate::ParameterBlockOrdering;
use crate::manifold::Manifold;
use crate::parameter_block::ParameterBlock;
use crate::{factors, loss_functions, residual_block};

type ResidualBlockId = usize;

pub struct Problem {
    pub total_residual_dimension: usize,
    residual_id_count: usize,
    residual_order: Vec<ResidualBlockId>,
    residual_blocks: Vec<Option<residual_block::ResidualBlock>>,
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
    value_scatter: Vec<(usize, usize)>,
    parameter_names: Vec<String>,
    residual_parameter_indices: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterLayout {
    pub variable_name_to_col_idx: HashMap<String, usize>,
    pub total_dimension: usize,
    pub parameter_block_sizes: Vec<usize>,
    pub schur_elimination_dimension: usize,
    pub schur_elimination_block_sizes: Vec<usize>,
    pub schur_retained_block_sizes: Vec<usize>,
}

type JacobianValue = f64;

impl Problem {
    pub fn new() -> Problem {
        Problem {
            total_residual_dimension: 0,
            residual_id_count: 0,
            residual_order: Vec::new(),
            residual_blocks: Vec::new(),
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
        let mut columns = vec![Vec::<(usize, usize)>::new(); total_variable_dimension];
        let mut source_index = 0;
        let parameter_names: Vec<_> = parameter_blocks.keys().cloned().collect();
        let parameter_indices: HashMap<_, _> = parameter_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let mut residual_parameter_indices = Vec::with_capacity(self.residual_order.len());

        self.residual_order.iter().for_each(|residual_id| {
            let residual_block = self.residual_blocks[*residual_id].as_ref().unwrap();
            residual_parameter_indices.push(
                residual_block
                    .variable_key_list
                    .iter()
                    .map(|name| parameter_indices[name.as_str()])
                    .collect(),
            );
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
                        let mut current_var_col_offset = 0;
                        for col_idx in 0..var_size {
                            if let Some(param) = parameter_blocks.get(var_key)
                                && param.manifold.is_none()
                                && param.fixed_variables.contains(&col_idx)
                            {
                                continue;
                            }
                            let global_row_idx = residual_block.residual_row_start_idx + row_idx;
                            let global_col_idx = variable_global_idx + current_var_col_offset;
                            columns[global_col_idx].push((global_row_idx, source_index));
                            source_index += 1;
                            current_var_col_offset += 1;
                        }
                    }
                }
            }
        });
        let start = std::time::Instant::now();
        let mut col_ptr = Vec::with_capacity(total_variable_dimension + 1);
        let mut row_idx = Vec::with_capacity(source_index);
        let mut value_scatter = Vec::with_capacity(source_index);
        col_ptr.push(0);
        for entries in &mut columns {
            if entries.windows(2).any(|window| window[0].0 > window[1].0) {
                entries.sort_unstable_by_key(|entry| entry.0);
            }
            let mut previous_row = None;
            for &(row, source) in entries.iter() {
                if previous_row != Some(row) {
                    row_idx.push(row);
                    previous_row = Some(row);
                }
                value_scatter.push((row_idx.len() - 1, source));
            }
            col_ptr.push(row_idx.len());
        }
        let pattern = SymbolicSparseColMat::new_checked(
            self.total_residual_dimension,
            total_variable_dimension,
            col_ptr,
            None,
            row_idx,
        );
        log::trace!("Built symbolic matrix: {:?}", start.elapsed());
        SymbolicStructure {
            pattern,
            value_scatter,
            parameter_names,
            residual_parameter_indices,
        }
    }

    pub fn get_variable_name_to_col_idx_dict(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
    ) -> HashMap<String, usize> {
        self.parameter_layout(parameter_blocks, None)
            .expect("the default parameter layout is always valid")
            .variable_name_to_col_idx
    }

    pub fn parameter_layout(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        ordering: Option<&ParameterBlockOrdering>,
    ) -> Result<ParameterLayout, String> {
        let mut ordered_names = Vec::with_capacity(parameter_blocks.len());
        if let Some(ordering) = ordering {
            for name in ordering.ordered_elements() {
                if !parameter_blocks.contains_key(name) {
                    return Err(format!("ordered parameter block '{name}' does not exist"));
                }
                ordered_names.push(name);
            }
        }

        let mut remaining_names: Vec<&str> = parameter_blocks
            .keys()
            .map(String::as_str)
            .filter(|name| ordering.is_none_or(|ordering| !ordering.is_member(name)))
            .collect();
        remaining_names.sort_unstable();
        ordered_names.extend(remaining_names);

        let mut count_col_idx = 0;
        let mut variable_name_to_col_idx_dict = HashMap::new();
        let mut parameter_block_sizes = Vec::new();
        let first_group = ordering.and_then(ParameterBlockOrdering::min_nonempty_group);
        let mut schur_elimination_dimension = 0;
        let mut schur_elimination_block_sizes = Vec::new();
        let mut schur_retained_block_sizes = Vec::new();
        for name in ordered_names {
            let parameter_block = &parameter_blocks[name];
            let effective_size = parameter_block.effective_tangent_size();
            variable_name_to_col_idx_dict.insert(name.to_owned(), count_col_idx);
            count_col_idx += effective_size;
            if effective_size > 0 {
                parameter_block_sizes.push(effective_size);
            }
            let is_eliminated = first_group.is_some()
                && ordering.and_then(|ordering| ordering.group_id(name)) == first_group;
            if is_eliminated {
                schur_elimination_dimension += effective_size;
                if effective_size > 0 {
                    schur_elimination_block_sizes.push(effective_size);
                }
            } else if effective_size > 0 {
                schur_retained_block_sizes.push(effective_size);
            }
        }

        if let (Some(ordering), Some(first_group)) = (ordering, first_group) {
            for residual_block in self.residual_blocks.iter().flatten() {
                let eliminated_blocks = residual_block
                    .variable_key_list
                    .iter()
                    .filter(|name| ordering.group_id(name) == Some(first_group))
                    .filter(|name| {
                        parameter_blocks
                            .get(*name)
                            .is_some_and(|block| block.effective_tangent_size() > 0)
                    })
                    .count();
                if eliminated_blocks > 1 {
                    return Err(format!(
                        "Schur elimination group is not independent in residual block {}",
                        residual_block.residual_block_id
                    ));
                }
            }
        }

        Ok(ParameterLayout {
            variable_name_to_col_idx: variable_name_to_col_idx_dict,
            total_dimension: count_col_idx,
            parameter_block_sizes,
            schur_elimination_dimension,
            schur_elimination_block_sizes,
            schur_retained_block_sizes,
        })
    }
    pub fn add_residual_block(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn factors::FactorImpl + Send>,
        loss_func: Option<Box<dyn loss_functions::Loss + Send>>,
    ) -> ResidualBlockId {
        self.residual_blocks
            .push(Some(residual_block::ResidualBlock::new(
                self.residual_id_count,
                dim_residual,
                self.total_residual_dimension,
                variable_key_size_list,
                factor,
                loss_func,
            )));
        let block_id = self.residual_id_count;
        self.residual_id_count += 1;

        self.total_residual_dimension += dim_residual;
        self.residual_order.push(block_id);

        block_id
    }
    pub fn remove_residual_block(
        &mut self,
        block_id: ResidualBlockId,
    ) -> Option<residual_block::ResidualBlock> {
        if let Some(residual_block) = self
            .residual_blocks
            .get_mut(block_id)
            .and_then(Option::take)
        {
            self.total_residual_dimension -= residual_block.dim_residual;
            self.residual_order
                .retain(|&residual_id| residual_id != block_id);
            let mut row_start = 0;
            for residual_id in &self.residual_order {
                let block = self.residual_blocks[*residual_id].as_mut().unwrap();
                block.residual_row_start_idx = row_start;
                row_start += block.dim_residual;
            }
            Some(residual_block)
        } else {
            None
        }
    }
    pub fn num_residual_blocks(&self) -> usize {
        self.residual_order.len()
    }
    pub fn num_residuals(&self) -> usize {
        self.total_residual_dimension
    }
    pub fn has_residual_block(&self, block_id: ResidualBlockId) -> bool {
        self.residual_blocks
            .get(block_id)
            .is_some_and(Option::is_some)
    }
    pub fn residual_block_variable_keys(&self, block_id: ResidualBlockId) -> Option<&[String]> {
        self.residual_blocks
            .get(block_id)
            .and_then(Option::as_ref)
            .map(|block| block.variable_key_list.as_slice())
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
                    p_block.set_manifold(manifold.clone());
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
        let residual_blocks: Vec<_> = self
            .residual_order
            .iter()
            .map(|residual_id| self.residual_blocks[*residual_id].as_ref().unwrap())
            .collect();
        let residuals: Vec<_> = residual_blocks
            .par_iter()
            .map(|residual_block| {
                (
                    residual_block.residual_row_start_idx,
                    self.compute_residual_impl(residual_block, parameter_blocks, with_loss_fn),
                )
            })
            .collect();
        let mut total_residual = na::DVector::<f64>::zeros(self.total_residual_dimension);
        for (row_start, residual) in residuals {
            total_residual
                .rows_mut(row_start, residual.len())
                .copy_from(&residual);
        }

        total_residual.view_range(.., ..).into_faer().to_owned()
    }

    pub fn compute_residuals_with_structure(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        symbolic_structure: &SymbolicStructure,
        with_loss_fn: bool,
    ) -> faer::Mat<f64> {
        let parameter_refs: Vec<_> = symbolic_structure
            .parameter_names
            .iter()
            .map(|name| &parameter_blocks[name])
            .collect();
        let residual_blocks: Vec<_> = self
            .residual_order
            .iter()
            .map(|residual_id| self.residual_blocks[*residual_id].as_ref().unwrap())
            .collect();
        let residuals: Vec<_> = residual_blocks
            .par_iter()
            .zip(symbolic_structure.residual_parameter_indices.par_iter())
            .map(|(residual_block, parameter_indices)| {
                let params: Vec<_> = parameter_indices
                    .iter()
                    .map(|&index| parameter_refs[index])
                    .collect();
                (
                    residual_block.residual_row_start_idx,
                    residual_block.residual(&params, with_loss_fn),
                )
            })
            .collect();
        let mut total_residual = na::DVector::<f64>::zeros(self.total_residual_dimension);
        for (row_start, residual) in residuals {
            total_residual
                .rows_mut(row_start, residual.len())
                .copy_from(&residual);
        }
        total_residual.view_range(.., ..).into_faer().to_owned()
    }

    pub fn compute_residual_and_jacobian(
        &self,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
        let residual_blocks: Vec<_> = self
            .residual_order
            .iter()
            .map(|residual_id| self.residual_blocks[*residual_id].as_ref().unwrap())
            .collect();
        let parameter_refs: Vec<_> = symbolic_structure
            .parameter_names
            .iter()
            .map(|name| &parameter_blocks[name])
            .collect();
        let evaluations: Vec<_> = residual_blocks
            .par_iter()
            .zip(symbolic_structure.residual_parameter_indices.par_iter())
            .map(|(residual_block, parameter_indices)| {
                let params: Vec<_> = parameter_indices
                    .iter()
                    .map(|&index| parameter_refs[index])
                    .collect();
                self.compute_residual_and_jacobian_impl(
                    residual_block,
                    &params,
                    variable_name_to_col_idx_dict,
                )
            })
            .collect();
        let mut total_residual = na::DVector::<f64>::zeros(self.total_residual_dimension);
        let mut jacobian_lists = Vec::new();
        for (row_start, residual, jacobian_values) in evaluations {
            total_residual
                .rows_mut(row_start, residual.len())
                .copy_from(&residual);
            jacobian_lists.extend(jacobian_values);
        }

        let residual_faer = total_residual.view_range(.., ..).into_faer().to_owned();
        let mut jacobian_values = vec![0.0; symbolic_structure.pattern.row_idx().len()];
        for &(target, source) in &symbolic_structure.value_scatter {
            jacobian_values[target] += jacobian_lists[source];
        }
        let jacobian_faer = SparseColMat::new(symbolic_structure.pattern.clone(), jacobian_values);
        (residual_faer, jacobian_faer)
    }

    fn compute_residual_impl(
        &self,
        residual_block: &crate::ResidualBlock,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        with_loss_fn: bool,
    ) -> na::DVector<f64> {
        let mut params = Vec::new();
        for var_key in &residual_block.variable_key_list {
            if let Some(param) = parameter_blocks.get(var_key) {
                params.push(param);
            };
        }
        residual_block.residual(&params, with_loss_fn)
    }

    fn compute_residual_and_jacobian_impl(
        &self,
        residual_block: &crate::ResidualBlock,
        params: &[&ParameterBlock],
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
    ) -> (usize, na::DVector<f64>, Vec<JacobianValue>) {
        let mut variable_local_idx_size_list = Vec::<(usize, usize)>::new();
        let mut count_variable_local_idx: usize = 0;
        for param in params {
            variable_local_idx_size_list.push((count_variable_local_idx, param.tangent_size()));
            count_variable_local_idx += param.tangent_size();
        }
        let (res, jac) = residual_block.residual_and_jacobian(params);

        let mut local_jacobian_list = Vec::new();

        for (i, var_key) in residual_block.variable_key_list.iter().enumerate() {
            if variable_name_to_col_idx_dict.contains_key(var_key) {
                let (variable_local_idx, var_size) = variable_local_idx_size_list[i];
                let variable_jac = jac.view((0, variable_local_idx), (jac.shape().0, var_size));
                let param = &params[i];
                for row_idx in 0..jac.shape().0 {
                    for col_idx in 0..var_size {
                        if param.manifold.is_none() && param.fixed_variables.contains(&col_idx) {
                            continue;
                        }
                        let j_value = variable_jac[(row_idx, col_idx)];
                        if j_value.is_finite() {
                            local_jacobian_list.push(j_value);
                        } else {
                            log::warn!(
                                "Non-finite Jacobian value detected at residual block {}, variable {}, row {}, col {}. Setting to 0.0",
                                residual_block.residual_block_id,
                                var_key,
                                row_idx,
                                col_idx
                            );
                            local_jacobian_list.push(0.0);
                        }
                    }
                }
            } else {
                panic!(
                    "Missing key {} in variable-to-column-index mapping",
                    var_key
                );
            }
        }

        (
            residual_block.residual_row_start_idx,
            res,
            local_jacobian_list,
        )
    }
}
