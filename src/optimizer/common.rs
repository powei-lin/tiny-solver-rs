use std::collections::{HashMap, HashSet};
use std::ops::Add;

use nalgebra as na;

use crate::parameter_block::ParameterBlock;
use crate::problem;
use crate::sparse::LinearSolverType;

pub trait Optimizer {
    fn optimize(
        &self,
        problem: &problem::Problem,
        initial_values: &HashMap<String, na::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> Option<HashMap<String, na::DVector<f64>>>;
    fn apply_dx(
        &self,
        dx: &na::DVector<f64>,
        params: &mut HashMap<String, na::DVector<f64>>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
        fixed_var_indexes: &HashMap<String, HashSet<usize>>,
        variable_bounds: &HashMap<String, HashMap<usize, (f64, f64)>>,
    ) {
        for (key, param) in params.iter_mut() {
            if let Some(col_idx) = variable_name_to_col_idx_dict.get(key) {
                let var_size = param.shape().0;
                let mut updated_param = param.clone().add(dx.rows(*col_idx, var_size));
                if let Some(indexes_to_fix) = fixed_var_indexes.get(key) {
                    for &idx in indexes_to_fix {
                        log::debug!("Fix {} {}", key, idx);
                        updated_param[idx] = param[idx];
                    }
                }
                if let Some(indexes_to_bound) = variable_bounds.get(key) {
                    for (&idx, &(lower, upper)) in indexes_to_bound {
                        let old = updated_param[idx];
                        updated_param[idx] = updated_param[idx].max(lower).min(upper);
                        log::debug!("bound {} {} {} -> {}", key, idx, old, updated_param[idx]);
                    }
                }
                param.copy_from(&updated_param);
            }
        }
    }
    fn apply_dx2(
        &self,
        dx: &na::DVector<f64>,
        params: &mut HashMap<String, ParameterBlock>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
    ) {
        params.iter_mut().for_each(|(key, param)| {
            if let Some(col_idx) = variable_name_to_col_idx_dict.get(key) {
                let var_size = param.tangent_size();
                param.update_params(param.plus_f64(dx.rows(*col_idx, var_size)));
            }
        });
        // for (key, param) in params.par_iter_mut() {
        // }
    }
    fn compute_error(
        &self,
        problem: &problem::Problem,
        params: &HashMap<String, ParameterBlock>,
    ) -> f64 {
        problem.compute_residuals(params, true).squared_norm_l2()
    }
}

#[derive(PartialEq, Debug)]
pub enum SolverStatus {
    Running,
    // Resulting solution may be OK to use.
    GradientTooSmall,         // eps > max(J'*f(x))
    RelativeStepSizeTooSmall, // eps > ||dx|| / ||x||
    ErrorTooSmall,            // eps > ||f(x)||
    HitMaxIterations,
    // Numerical issues
    // FAILED_TO_EVALUATE_COST_FUNCTION,
    // FAILED_TO_SOLVER_LINEAR_SYSTEM,
}

#[derive(Clone)]
pub struct OptimizerOptions {
    pub max_iteration: usize,
    pub linear_solver_type: LinearSolverType,
    pub verbosity_level: usize,
    pub min_abs_error_decrease_threshold: f64,
    pub min_rel_error_decrease_threshold: f64,
    pub min_error_threshold: f64,
    // pub relative_step_threshold: 1e-16,
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        OptimizerOptions {
            max_iteration: 100,
            linear_solver_type: LinearSolverType::SparseCholesky,
            verbosity_level: 0,
            min_abs_error_decrease_threshold: 1e-5,
            min_rel_error_decrease_threshold: 1e-5,
            min_error_threshold: 1e-10,
        }
    }
}
