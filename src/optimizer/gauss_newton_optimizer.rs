use log::trace;
use std::ops::Mul;
use std::{collections::HashMap, time::Instant};

use faer_ext::IntoNalgebra;

use crate::common::{
    CallbackReturnType, IterationSummary, OptimizerOptions, SolverResult, SolverSummary,
    TerminationType, configuration_failure,
};
use crate::linear;
use crate::optimizer;
use crate::parameter_block::ParameterBlock;
use crate::sparse::LinearSolverType;
use crate::sparse::SparseLinearSolver;

#[derive(Debug)]
pub struct GaussNewtonOptimizer {}
impl GaussNewtonOptimizer {
    pub fn new() -> Self {
        Self {}
    }
}
impl Default for GaussNewtonOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl optimizer::Optimizer for GaussNewtonOptimizer {
    fn optimize_with_summary(
        &self,
        problem: &crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> SolverResult {
        let total_start = Instant::now();
        let mut parameter_blocks: HashMap<String, ParameterBlock> =
            problem.initialize_parameter_blocks(initial_values);

        let opt_option = optimizer_option.unwrap_or_default();
        let layout = match problem.parameter_layout(
            &parameter_blocks,
            opt_option.parameter_block_ordering.as_ref(),
        ) {
            Ok(layout) => layout,
            Err(message) => return configuration_failure(message, total_start.elapsed()),
        };
        let total_variable_dimension = layout.total_dimension;
        let parameter_block_sizes = layout.parameter_block_sizes;
        let schur_elimination_dimension = layout.schur_elimination_dimension;
        let schur_elimination_block_sizes = layout.schur_elimination_block_sizes;
        let schur_retained_block_sizes = layout.schur_retained_block_sizes;
        let variable_name_to_col_idx_dict = layout.variable_name_to_col_idx;

        if matches!(
            opt_option.linear_solver_type,
            LinearSolverType::DenseSchur
                | LinearSolverType::SparseSchur
                | LinearSolverType::IterativeSchur
        ) && (schur_elimination_dimension == 0
            || schur_elimination_dimension >= total_variable_dimension)
        {
            return configuration_failure(
                "Schur solvers require an ordering with nonempty eliminated and retained groups",
                total_start.elapsed(),
            );
        }
        if matches!(
            opt_option.linear_solver_type,
            LinearSolverType::IterativeSchur | LinearSolverType::Cgnr
        ) && (opt_option.max_linear_solver_iterations == 0
            || opt_option.min_linear_solver_iterations > opt_option.max_linear_solver_iterations
            || !opt_option.eta.is_finite()
            || opt_option.eta < 0.0)
        {
            return configuration_failure(
                "Iterative solvers require valid iteration limits and a finite nonnegative eta",
                total_start.elapsed(),
            );
        }
        if opt_option.linear_solver_type == LinearSolverType::Cgnr
            && opt_option.preconditioner_type == linear::PreconditionerType::SchurJacobi
        {
            return configuration_failure(
                "CGNR supports only Identity and Jacobi preconditioners",
                total_start.elapsed(),
            );
        }

        let mut linear_solver: Box<dyn SparseLinearSolver> = match opt_option.linear_solver_type {
            LinearSolverType::DenseQR => Box::new(linear::DenseQRSolver::new()),
            LinearSolverType::DenseNormalCholesky => {
                Box::new(linear::DenseNormalCholeskySolver::new())
            }
            LinearSolverType::DenseSchur => {
                Box::new(linear::SchurComplementSolver::new_with_block_sizes(
                    schur_elimination_block_sizes.clone(),
                    linear::SchurComplementMode::Dense,
                ))
            }
            LinearSolverType::SparseSchur => {
                Box::new(linear::SchurComplementSolver::new_with_block_sizes(
                    schur_elimination_block_sizes,
                    linear::SchurComplementMode::Sparse,
                ))
            }
            LinearSolverType::IterativeSchur => Box::new(linear::IterativeSchurSolver::new(
                schur_elimination_block_sizes,
                schur_retained_block_sizes,
                linear::IterativeSchurOptions {
                    preconditioner_type: opt_option.preconditioner_type,
                    min_num_iterations: opt_option.min_linear_solver_iterations,
                    max_num_iterations: opt_option.max_linear_solver_iterations,
                    q_tolerance: opt_option.eta,
                    ..linear::IterativeSchurOptions::default()
                },
            )),
            LinearSolverType::Cgnr => Box::new(linear::CgnrSolver::new(
                parameter_block_sizes,
                linear::CgnrOptions {
                    preconditioner_type: opt_option.preconditioner_type,
                    min_num_iterations: opt_option.min_linear_solver_iterations,
                    max_num_iterations: opt_option.max_linear_solver_iterations,
                    q_tolerance: opt_option.eta,
                    ..linear::CgnrOptions::default()
                },
            )),
            LinearSolverType::SparseCholesky => Box::new(linear::SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(linear::SparseQRSolver::new()),
        };

        let symbolic_structure = problem.build_symbolic_structure(
            &parameter_blocks,
            total_variable_dimension,
            &variable_name_to_col_idx_dict,
        );

        if let Some(callback) = &opt_option.evaluation_callback {
            callback.prepare_for_evaluation(false, true);
        }
        let mut current_error = problem
            .compute_residuals_with_structure(&parameter_blocks, &symbolic_structure, true)
            .as_ref()
            .squared_norm_l2();
        let initial_cost = current_error;
        let mut iterations = Vec::new();
        let mut termination_type = TerminationType::NoConvergence;
        let mut message = "Maximum number of iterations reached".to_string();

        for i in 0..opt_option.max_iteration {
            let iteration_start = Instant::now();
            let last_err = current_error;
            let mut start = Instant::now();

            if let Some(callback) = &opt_option.evaluation_callback {
                callback.prepare_for_evaluation(true, false);
            }
            let (residuals, jac) = problem.compute_residual_and_jacobian(
                &parameter_blocks,
                &variable_name_to_col_idx_dict,
                &symbolic_structure,
            );
            let residual_and_jacobian_duration = start.elapsed();
            let gradient = jac.as_ref().transpose().mul(&residuals);
            let gradient_max_norm = gradient
                .as_ref()
                .into_nalgebra()
                .iter()
                .fold(0.0_f64, |max_value, value| max_value.max(value.abs()));

            start = Instant::now();
            let step_norm;
            if let Some(dx) = linear_solver.solve(&residuals, &jac) {
                let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
                step_norm = dx_na.norm();
                self.apply_dx2(
                    &dx_na,
                    &mut parameter_blocks,
                    &variable_name_to_col_idx_dict,
                );
            } else {
                log::debug!("solve ax=b failed");
                termination_type = TerminationType::Failure;
                message = "Failed to solve the linear system".to_string();
                break;
            }
            let solving_duration = start.elapsed();

            if let Some(callback) = &opt_option.evaluation_callback {
                callback.prepare_for_evaluation(false, true);
            }
            current_error = problem
                .compute_residuals_with_structure(&parameter_blocks, &symbolic_structure, true)
                .as_ref()
                .squared_norm_l2();
            trace!(
                "iter:{}, total err:{}, residual + jacobian duration: {:?}, solving duration: {:?}",
                i, current_error, residual_and_jacobian_duration, solving_duration
            );

            let iteration_summary = IterationSummary {
                iteration: i,
                cost: current_error,
                cost_change: last_err - current_error,
                gradient_max_norm,
                step_norm,
                trust_region_radius: None,
                step_is_successful: true,
                iteration_time: iteration_start.elapsed(),
            };
            iterations.push(iteration_summary);

            let callback_result = opt_option
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

            if !current_error.is_finite() {
                termination_type = TerminationType::Failure;
                message = "Non-finite cost encountered".to_string();
                break;
            } else if current_error < opt_option.min_error_threshold {
                termination_type = TerminationType::Convergence;
                message = "Cost tolerance reached".to_string();
                break;
            } else if gradient_max_norm < opt_option.gradient_tolerance {
                termination_type = TerminationType::Convergence;
                message = "Gradient tolerance reached".to_string();
                break;
            } else if step_norm
                <= opt_option.parameter_tolerance
                    * (parameter_blocks
                        .values()
                        .map(|parameter| parameter.params.norm_squared())
                        .sum::<f64>()
                        .sqrt()
                        + opt_option.parameter_tolerance)
            {
                termination_type = TerminationType::Convergence;
                message = "Parameter tolerance reached".to_string();
                break;
            }

            if (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold {
                termination_type = TerminationType::Convergence;
                message = "Absolute cost change tolerance reached".to_string();
                break;
            } else if last_err > 0.0
                && (last_err - current_error).abs() / last_err
                    < opt_option.min_rel_error_decrease_threshold
            {
                termination_type = TerminationType::Convergence;
                message = "Relative cost change tolerance reached".to_string();
                break;
            }
        }
        let parameters = (termination_type != TerminationType::Failure
            && termination_type != TerminationType::UserFailure)
            .then(|| {
                parameter_blocks
                    .iter()
                    .map(|(key, value)| (key.to_owned(), value.params.clone()))
                    .collect()
            });
        SolverResult {
            parameters,
            summary: SolverSummary {
                termination_type,
                message,
                initial_cost,
                final_cost: current_error,
                iterations,
                num_inner_iteration_steps: 0,
                inner_iteration_time: std::time::Duration::ZERO,
                total_time: total_start.elapsed(),
            },
        }
    }
}
