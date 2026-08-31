use std::collections::HashMap;
use std::ops::Mul;
use std::time::Instant;

use nalgebra as na;

use super::common::{
    CallbackReturnType, IterationSummary, Optimizer, OptimizerOptions, SolverResult, SolverSummary,
    TerminationType, configuration_failure,
};
use super::line_search::{
    LineSearchDirection, LineSearchDirectionType, LineSearchParameters, LineSearchSample,
    LineSearchType, search,
};
use crate::parameter_block::ParameterBlock;
use crate::problem::{Problem, SymbolicStructure};

#[derive(Clone, Copy, Debug, Default)]
pub struct LineSearchOptimizer;

impl LineSearchOptimizer {
    pub fn new() -> Self {
        Self
    }

    fn evaluate(
        problem: &Problem,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        variable_name_to_col_idx: &HashMap<String, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> (f64, na::DVector<f64>) {
        let (residuals, jacobian) = problem.compute_residual_and_jacobian(
            parameter_blocks,
            variable_name_to_col_idx,
            symbolic_structure,
        );
        let cost = residuals.as_ref().squared_norm_l2();
        let gradient = jacobian.as_ref().transpose().mul(residuals.as_ref());
        let gradient = na::DVector::from_fn(gradient.nrows(), |row, _| 2.0 * gradient[(row, 0)]);
        (cost, gradient)
    }

    fn parameter_norm(parameter_blocks: &HashMap<String, ParameterBlock>) -> f64 {
        parameter_blocks
            .values()
            .map(|parameter| parameter.params.norm_squared())
            .sum::<f64>()
            .sqrt()
    }
}

impl Optimizer for LineSearchOptimizer {
    fn optimize_with_summary(
        &self,
        problem: &Problem,
        initial_values: &HashMap<String, na::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> SolverResult {
        let total_start = Instant::now();
        let options = optimizer_option.unwrap_or_default();
        if !problem.variable_bounds.is_empty() {
            return configuration_failure(
                "Line-search minimization does not support parameter bounds",
                total_start.elapsed(),
            );
        }
        if options.inner_iteration_ordering.is_some() {
            return configuration_failure(
                "Inner iterations are only supported by trust-region minimization",
                total_start.elapsed(),
            );
        }
        let line_search_parameters = LineSearchParameters {
            sufficient_decrease: options.line_search_sufficient_function_decrease,
            sufficient_curvature_decrease: options.line_search_sufficient_curvature_decrease,
            max_step_contraction: options.max_line_search_step_contraction,
            min_step_contraction: options.min_line_search_step_contraction,
            min_step_size: options.min_line_search_step_size,
            max_step_expansion: options.max_line_search_step_expansion,
            max_num_iterations: options.max_num_line_search_step_size_iterations,
        };
        if !line_search_parameters.is_valid() {
            return configuration_failure("Invalid line-search parameters", total_start.elapsed());
        }
        if options.line_search_direction_type == LineSearchDirectionType::Lbfgs
            && options.line_search_type != LineSearchType::Wolfe
        {
            return configuration_failure(
                "L-BFGS requires a Wolfe line search",
                total_start.elapsed(),
            );
        }
        let Some(mut direction_generator) = LineSearchDirection::new(
            options.line_search_direction_type,
            options.max_lbfgs_rank,
            1e-14,
        ) else {
            return configuration_failure(
                "Invalid line-search direction parameters",
                total_start.elapsed(),
            );
        };

        let mut parameter_blocks = problem.initialize_parameter_blocks(initial_values);
        let layout = match problem
            .parameter_layout(&parameter_blocks, options.parameter_block_ordering.as_ref())
        {
            Ok(layout) => layout,
            Err(message) => return configuration_failure(message, total_start.elapsed()),
        };
        let variable_name_to_col_idx = layout.variable_name_to_col_idx;
        let symbolic_structure = problem.build_symbolic_structure(
            &parameter_blocks,
            layout.total_dimension,
            &variable_name_to_col_idx,
        );

        if let Some(callback) = &options.evaluation_callback {
            callback.prepare_for_evaluation(true, true);
        }
        let (mut current_cost, mut current_gradient) = Self::evaluate(
            problem,
            &parameter_blocks,
            &variable_name_to_col_idx,
            &symbolic_structure,
        );
        let initial_cost = current_cost;
        let mut previous_cost = None;
        let mut previous_step = None;
        let mut iterations = Vec::new();
        let mut termination_type = TerminationType::NoConvergence;
        let mut message = "Maximum number of iterations reached".to_string();

        for iteration in 0..options.max_iteration {
            let iteration_start = Instant::now();
            let gradient_max_norm = current_gradient
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            if gradient_max_norm < options.gradient_tolerance {
                termination_type = TerminationType::Convergence;
                message = "Gradient tolerance reached".to_string();
                break;
            }

            let Some(direction) =
                direction_generator.next(&current_gradient, previous_step.as_ref())
            else {
                termination_type = TerminationType::Failure;
                message = "Failed to compute a line-search direction".to_string();
                break;
            };
            let directional_derivative = current_gradient.dot(&direction);
            let initial_step_size = previous_cost
                .map(|cost: f64| (2.0 * (current_cost - cost) / directional_derivative).min(1.0))
                .unwrap_or_else(|| (1.0 / gradient_max_norm).min(1.0));
            if !initial_step_size.is_finite() || initial_step_size <= 0.0 {
                termination_type = TerminationType::Failure;
                message = "Failed to compute a positive initial line-search step".to_string();
                break;
            }

            let initial_sample = LineSearchSample {
                step_size: 0.0,
                cost: current_cost,
                directional_derivative,
            };
            let result = search(
                options.line_search_type,
                initial_sample,
                initial_step_size,
                direction.amax(),
                line_search_parameters,
                |step_size| {
                    let mut trial_parameters = parameter_blocks.clone();
                    let step = step_size * &direction;
                    self.apply_dx2(&step, &mut trial_parameters, &variable_name_to_col_idx);
                    if let Some(callback) = &options.evaluation_callback {
                        callback.prepare_for_evaluation(true, true);
                    }
                    let (cost, gradient) = Self::evaluate(
                        problem,
                        &trial_parameters,
                        &variable_name_to_col_idx,
                        &symbolic_structure,
                    );
                    Some(LineSearchSample {
                        step_size,
                        cost,
                        directional_derivative: gradient.dot(&direction),
                    })
                },
            );
            let Some(line_search_result) = result else {
                termination_type = TerminationType::Failure;
                message = "Line search failed to find a sufficient decrease".to_string();
                break;
            };

            let last_cost = current_cost;
            let step = line_search_result.sample.step_size * &direction;
            self.apply_dx2(&step, &mut parameter_blocks, &variable_name_to_col_idx);
            if let Some(callback) = &options.evaluation_callback {
                callback.prepare_for_evaluation(true, true);
            }
            let evaluation = Self::evaluate(
                problem,
                &parameter_blocks,
                &variable_name_to_col_idx,
                &symbolic_structure,
            );
            current_cost = evaluation.0;
            current_gradient = evaluation.1;
            previous_cost = Some(last_cost);
            previous_step = Some(step.clone());

            let current_gradient_max_norm = current_gradient
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            iterations.push(IterationSummary {
                iteration,
                cost: current_cost,
                cost_change: last_cost - current_cost,
                gradient_max_norm: current_gradient_max_norm,
                step_norm: step.norm(),
                trust_region_radius: None,
                step_is_successful: true,
                iteration_time: iteration_start.elapsed(),
            });

            let callback_result = options
                .callbacks
                .iter()
                .map(|callback| callback.on_iteration(iterations.last().unwrap()))
                .find(|result| *result != CallbackReturnType::Continue);
            if let Some(result) = callback_result {
                match result {
                    CallbackReturnType::TerminateSuccessfully => {
                        termination_type = TerminationType::UserSuccess;
                        message = "Terminated successfully by callback".to_string();
                    }
                    CallbackReturnType::Abort => {
                        termination_type = TerminationType::UserFailure;
                        message = "Aborted by callback".to_string();
                    }
                    CallbackReturnType::Continue => unreachable!(),
                }
                break;
            }

            if !current_cost.is_finite() {
                termination_type = TerminationType::Failure;
                message = "Non-finite cost encountered".to_string();
                break;
            } else if current_cost < options.min_error_threshold {
                termination_type = TerminationType::Convergence;
                message = "Cost tolerance reached".to_string();
                break;
            } else if current_gradient_max_norm < options.gradient_tolerance {
                termination_type = TerminationType::Convergence;
                message = "Gradient tolerance reached".to_string();
                break;
            } else if step.norm()
                <= options.parameter_tolerance
                    * (Self::parameter_norm(&parameter_blocks) + options.parameter_tolerance)
            {
                termination_type = TerminationType::Convergence;
                message = "Parameter tolerance reached".to_string();
                break;
            } else if (last_cost - current_cost).abs() < options.min_abs_error_decrease_threshold {
                termination_type = TerminationType::Convergence;
                message = "Absolute cost change tolerance reached".to_string();
                break;
            } else if last_cost > 0.0
                && (last_cost - current_cost).abs() / last_cost
                    < options.min_rel_error_decrease_threshold
            {
                termination_type = TerminationType::Convergence;
                message = "Relative cost change tolerance reached".to_string();
                break;
            }
        }

        let parameters = (!matches!(
            termination_type,
            TerminationType::Failure | TerminationType::UserFailure
        ))
        .then(|| {
            parameter_blocks
                .iter()
                .map(|(key, value)| (key.clone(), value.params.clone()))
                .collect()
        });
        SolverResult {
            parameters,
            summary: SolverSummary {
                termination_type,
                message,
                initial_cost,
                final_cost: current_cost,
                iterations,
                num_inner_iteration_steps: 0,
                inner_iteration_time: std::time::Duration::ZERO,
                total_time: total_start.elapsed(),
            },
        }
    }
}
