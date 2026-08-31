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
use crate::optimizer::trust_region::{
    DoglegStrategy, LevenbergMarquardtStrategy, TrustRegionStrategy, TrustRegionStrategyType,
};
use crate::parameter_block::ParameterBlock;
use crate::sparse::LinearSolverType;
use crate::sparse::SparseLinearSolver;

const DEFAULT_MIN_DIAGONAL: f64 = 1e-6;
const DEFAULT_MAX_DIAGONAL: f64 = 1e32;
const DEFAULT_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;
const DEFAULT_MAX_TRUST_REGION_RADIUS: f64 = 1e16;

#[derive(Debug)]
pub struct LevenbergMarquardtOptimizer {
    min_diagonal: f64,
    max_diagonal: f64,
    initial_trust_region_radius: f64,
}

pub type TrustRegionOptimizer = LevenbergMarquardtOptimizer;

impl LevenbergMarquardtOptimizer {
    pub fn new(min_diagonal: f64, max_diagonal: f64, initial_trust_region_radius: f64) -> Self {
        Self {
            min_diagonal,
            max_diagonal,
            initial_trust_region_radius,
        }
    }
}

impl Default for LevenbergMarquardtOptimizer {
    fn default() -> Self {
        Self {
            min_diagonal: DEFAULT_MIN_DIAGONAL,
            max_diagonal: DEFAULT_MAX_DIAGONAL,
            initial_trust_region_radius: DEFAULT_INITIAL_TRUST_REGION_RADIUS,
        }
    }
}

impl optimizer::Optimizer for LevenbergMarquardtOptimizer {
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
        if matches!(
            opt_option.trust_region_strategy_type,
            TrustRegionStrategyType::Dogleg(_)
        ) && matches!(
            opt_option.linear_solver_type,
            LinearSolverType::IterativeSchur | LinearSolverType::Cgnr
        ) {
            return configuration_failure(
                "Dogleg requires an exact factorization-based linear solver",
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
        let mut trust_region_strategy: Box<dyn TrustRegionStrategy> =
            match opt_option.trust_region_strategy_type {
                TrustRegionStrategyType::LevenbergMarquardt => {
                    Box::new(LevenbergMarquardtStrategy::new(
                        self.initial_trust_region_radius,
                        DEFAULT_MAX_TRUST_REGION_RADIUS,
                        self.min_diagonal,
                        self.max_diagonal,
                    ))
                }
                TrustRegionStrategyType::Dogleg(dogleg_type) => Box::new(DoglegStrategy::new(
                    dogleg_type,
                    self.initial_trust_region_radius,
                    DEFAULT_MAX_TRUST_REGION_RADIUS,
                    self.min_diagonal,
                    self.max_diagonal,
                )),
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

            if let Some(callback) = &opt_option.evaluation_callback {
                callback.prepare_for_evaluation(true, false);
            }
            let (residuals, jac) = problem.compute_residual_and_jacobian(
                &parameter_blocks,
                &variable_name_to_col_idx_dict,
                &symbolic_structure,
            );

            // J^T * -r = Matrix of shape (total_variable_dimension, 1)
            let jtr = jac.as_ref().transpose().mul(-&residuals);
            let gradient_max_norm = jtr
                .as_ref()
                .into_nalgebra()
                .iter()
                .fold(0.0_f64, |max_value, value| max_value.max(value.abs()));

            let step_norm;
            let mut step_is_successful = false;
            if let Some(trust_region_step) =
                trust_region_strategy.compute_step(&residuals, &jac, linear_solver.as_mut())
            {
                let dx_na = trust_region_step.step;
                step_norm = dx_na.norm();

                let mut new_param_blocks = parameter_blocks.clone();

                self.apply_dx2(
                    &dx_na,
                    &mut new_param_blocks,
                    &variable_name_to_col_idx_dict,
                );

                // Compute residuals of (x + dx)
                if let Some(callback) = &opt_option.evaluation_callback {
                    callback.prepare_for_evaluation(false, true);
                }
                let new_residuals = problem.compute_residuals_with_structure(
                    &new_param_blocks,
                    &symbolic_structure,
                    true,
                );
                let new_error = new_residuals.as_ref().squared_norm_l2();

                // rho is the ratio between the actual reduction in error and the reduction
                // in error if the problem were linear.
                let actual_residual_change = residuals.as_ref().squared_norm_l2() - new_error;
                trace!("actual_residual_change {}", actual_residual_change);
                let step_faer = faer::Mat::from_fn(dx_na.len(), 1, |row, _| dx_na[row]);
                let jacobian_step = jac.as_ref().mul(step_faer.as_ref());
                let normal_step = jac.as_ref().transpose().mul(jacobian_step.as_ref());
                let linear_residual_change: faer::Mat<f64> =
                    step_faer.transpose().mul(2.0 * &jtr - normal_step);
                let rho = actual_residual_change / linear_residual_change[(0, 0)];

                if rho > 0.0 && rho.is_finite() {
                    // The linear model appears to be fitting, so accept (x + dx) as the new x.
                    parameter_blocks = new_param_blocks;
                    current_error = new_error;
                    step_is_successful = true;
                    trust_region_strategy.step_accepted(rho);
                } else {
                    trust_region_strategy.step_rejected();
                }
            } else {
                log::debug!("solve ax=b failed");
                termination_type = TerminationType::Failure;
                message = "Failed to solve the linear system".to_string();
                break;
            }

            trace!("iter:{} total err:{}", i, current_error);

            let iteration_summary = IterationSummary {
                iteration: i,
                cost: current_error,
                cost_change: last_err - current_error,
                gradient_max_norm,
                step_norm,
                trust_region_radius: Some(trust_region_strategy.radius()),
                step_is_successful,
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
            } else if step_is_successful
                && step_norm
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

            if step_is_successful
                && (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold
            {
                termination_type = TerminationType::Convergence;
                message = "Absolute cost change tolerance reached".to_string();
                break;
            } else if step_is_successful
                && last_err > 0.0
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
                total_time: total_start.elapsed(),
            },
        }
    }
}
