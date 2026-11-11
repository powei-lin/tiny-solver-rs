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
    ) -> SolverSummary;
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
                let tangent_size = param.tangent_size();
                let effective_size = if param.manifold.is_some() {
                    tangent_size
                } else {
                    tangent_size - param.fixed_variables.len()
                };

                let dx_reduced = dx.rows(*col_idx, effective_size);

                let mut dx_full = na::DVector::zeros(tangent_size);
                if param.manifold.is_some() {
                    dx_full.copy_from(&dx_reduced);
                } else {
                    let mut reduced_idx = 0;
                    for i in 0..tangent_size {
                        if !param.fixed_variables.contains(&i) {
                            dx_full[i] = dx_reduced[reduced_idx];
                            reduced_idx += 1;
                        }
                    }
                }
                param.update_params(param.plus_f64(dx_full.rows(0, tangent_size)));
            }
        });
        // for (key, param) in params.par_iter_mut() {
        // }
    }
}

/// Why the solver stopped, as Ceres' `TerminationType`.
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminationType {
    /// A convergence criterion was met.
    Convergence,
    /// `max_iteration` was reached first.
    NoConvergence,
    /// The solver could not go on, e.g. the cost became NaN.
    Failure,
}

/// What `optimize` returns, after Ceres' `Solver::Summary`. The costs are those
/// `Problem::compute_cost` measures: the sum of rho(||r_i||^2), where Ceres reports half of it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SolverSummary {
    /// The best parameters found, also after a failure.
    pub parameters: HashMap<String, na::DVector<f64>>,
    pub termination_type: TerminationType,
    /// Which criterion stopped the solver, or why it failed.
    pub message: String,
    pub initial_cost: f64,
    pub final_cost: f64,
    /// Accepted steps.
    pub num_successful_steps: usize,
    /// Rejected steps, and failed linear solves.
    pub num_unsuccessful_steps: usize,
    pub total_time: std::time::Duration,
}

#[non_exhaustive]
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
