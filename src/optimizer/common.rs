use std::collections::{HashMap, HashSet};
use std::ops::Add;
use std::sync::Arc;

use nalgebra as na;

use crate::parameter_block::ParameterBlock;
use crate::problem;
use crate::sparse::LinearSolverType;
use crate::{
    LineSearchDirectionType, LineSearchType, ParameterBlockOrdering, PreconditionerType,
    TrustRegionStrategyType,
};

pub trait Optimizer {
    fn optimize_with_summary(
        &self,
        problem: &problem::Problem,
        initial_values: &HashMap<String, na::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> SolverResult;
    fn optimize(
        &self,
        problem: &problem::Problem,
        initial_values: &HashMap<String, na::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> Option<HashMap<String, na::DVector<f64>>> {
        self.optimize_with_summary(problem, initial_values, optimizer_option)
            .parameters
    }
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
    fn compute_error(
        &self,
        problem: &problem::Problem,
        params: &HashMap<String, ParameterBlock>,
    ) -> f64 {
        problem
            .compute_residuals(params, true)
            .as_ref()
            .squared_norm_l2()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationType {
    Convergence,
    NoConvergence,
    Failure,
    UserSuccess,
    UserFailure,
}

#[derive(Clone, Debug)]
pub struct IterationSummary {
    pub iteration: usize,
    pub cost: f64,
    pub cost_change: f64,
    pub gradient_max_norm: f64,
    pub step_norm: f64,
    pub trust_region_radius: Option<f64>,
    pub step_is_successful: bool,
    pub iteration_time: std::time::Duration,
}

#[derive(Clone, Debug)]
pub struct SolverSummary {
    pub termination_type: TerminationType,
    pub message: String,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<IterationSummary>,
    pub num_inner_iteration_steps: usize,
    pub inner_iteration_time: std::time::Duration,
    pub total_time: std::time::Duration,
}

impl SolverSummary {
    pub fn full_report(&self) -> String {
        format!(
            "Termination: {:?}\nMessage: {}\nInitial cost: {:.12e}\nFinal cost: {:.12e}\nIterations: {}\nInner iteration steps: {}\nTotal time: {:?}",
            self.termination_type,
            self.message,
            self.initial_cost,
            self.final_cost,
            self.iterations.len(),
            self.num_inner_iteration_steps,
            self.total_time
        )
    }
}

#[derive(Debug)]
pub struct SolverResult {
    pub parameters: Option<HashMap<String, na::DVector<f64>>>,
    pub summary: SolverSummary,
}

pub(crate) fn configuration_failure(
    message: impl Into<String>,
    total_time: std::time::Duration,
) -> SolverResult {
    SolverResult {
        parameters: None,
        summary: SolverSummary {
            termination_type: TerminationType::Failure,
            message: message.into(),
            initial_cost: f64::NAN,
            final_cost: f64::NAN,
            iterations: Vec::new(),
            num_inner_iteration_steps: 0,
            inner_iteration_time: std::time::Duration::ZERO,
            total_time,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackReturnType {
    Continue,
    TerminateSuccessfully,
    Abort,
}

pub trait IterationCallback: Send + Sync {
    fn on_iteration(&self, summary: &IterationSummary) -> CallbackReturnType;
}

pub trait EvaluationCallback: Send + Sync {
    fn prepare_for_evaluation(&self, evaluate_jacobians: bool, new_evaluation_point: bool);
}

#[derive(Clone)]
pub struct OptimizerOptions {
    pub max_iteration: usize,
    pub linear_solver_type: LinearSolverType,
    pub parameter_block_ordering: Option<ParameterBlockOrdering>,
    pub preconditioner_type: PreconditionerType,
    pub min_linear_solver_iterations: usize,
    pub max_linear_solver_iterations: usize,
    pub eta: f64,
    pub trust_region_strategy_type: TrustRegionStrategyType,
    pub line_search_direction_type: LineSearchDirectionType,
    pub line_search_type: LineSearchType,
    pub max_lbfgs_rank: usize,
    pub min_line_search_step_size: f64,
    pub line_search_sufficient_function_decrease: f64,
    pub line_search_sufficient_curvature_decrease: f64,
    pub max_line_search_step_contraction: f64,
    pub min_line_search_step_contraction: f64,
    pub max_num_line_search_step_size_iterations: usize,
    pub max_line_search_step_expansion: f64,
    pub inner_iteration_ordering: Option<ParameterBlockOrdering>,
    pub inner_iteration_tolerance: f64,
    pub max_num_inner_iterations: usize,
    pub verbosity_level: usize,
    pub min_abs_error_decrease_threshold: f64,
    pub min_rel_error_decrease_threshold: f64,
    pub min_error_threshold: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub callbacks: Vec<Arc<dyn IterationCallback>>,
    pub evaluation_callback: Option<Arc<dyn EvaluationCallback>>,
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        OptimizerOptions {
            max_iteration: 100,
            linear_solver_type: LinearSolverType::SparseCholesky,
            parameter_block_ordering: None,
            preconditioner_type: PreconditionerType::Jacobi,
            min_linear_solver_iterations: 0,
            max_linear_solver_iterations: 500,
            eta: 1e-1,
            trust_region_strategy_type: TrustRegionStrategyType::LevenbergMarquardt,
            line_search_direction_type: LineSearchDirectionType::Lbfgs,
            line_search_type: LineSearchType::Wolfe,
            max_lbfgs_rank: 20,
            min_line_search_step_size: 1e-9,
            line_search_sufficient_function_decrease: 1e-4,
            line_search_sufficient_curvature_decrease: 0.9,
            max_line_search_step_contraction: 1e-3,
            min_line_search_step_contraction: 0.6,
            max_num_line_search_step_size_iterations: 20,
            max_line_search_step_expansion: 10.0,
            inner_iteration_ordering: None,
            inner_iteration_tolerance: 1e-3,
            max_num_inner_iterations: 10,
            verbosity_level: 0,
            min_abs_error_decrease_threshold: 1e-5,
            min_rel_error_decrease_threshold: 1e-5,
            min_error_threshold: 1e-10,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-8,
            callbacks: Vec::new(),
            evaluation_callback: None,
        }
    }
}
