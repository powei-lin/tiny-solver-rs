use log::trace;
use std::{collections::HashMap, time::Instant};

use faer_ext::IntoNalgebra;

use crate::common::{OptimizerOptions, SolverSummary, TerminationType};
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
    fn optimize(
        &self,
        problem: &crate::problem::Problem,
        initial_values: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
        optimizer_option: Option<OptimizerOptions>,
    ) -> SolverSummary {
        let start_time = Instant::now();
        let mut parameter_blocks: HashMap<String, ParameterBlock> =
            problem.initialize_parameter_blocks(initial_values);

        let variable_name_to_col_idx_dict =
            problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
        let total_variable_dimension = parameter_blocks
            .iter()
            .filter(|(name, _)| variable_name_to_col_idx_dict.contains_key(*name))
            .map(|(_, p)| {
                if p.manifold.is_some() {
                    p.tangent_size()
                } else {
                    p.tangent_size() - p.fixed_variables.len()
                }
            })
            .sum();

        let opt_option = optimizer_option.unwrap_or_default();
        let mut linear_solver: Box<dyn SparseLinearSolver> = match opt_option.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(linear::SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(linear::SparseQRSolver::new()),
        };

        let symbolic_structure = problem.build_symbolic_structure(
            &parameter_blocks,
            total_variable_dimension,
            &variable_name_to_col_idx_dict,
        );

        let mut last_err;
        let mut current_error = problem.compute_cost(&parameter_blocks);
        let initial_cost = current_error;
        let mut num_successful_steps = 0;
        let mut termination = (TerminationType::NoConvergence, "max_iteration reached");

        for i in 0..opt_option.max_iteration {
            last_err = current_error;
            let mut start = Instant::now();

            let (residuals, jac) = problem.compute_residual_and_jacobian(
                &parameter_blocks,
                &variable_name_to_col_idx_dict,
                &symbolic_structure,
            );
            let residual_and_jacobian_duration = start.elapsed();

            start = Instant::now();
            let solving_duration;
            let mut new_param_blocks = parameter_blocks.clone();
            if let Some(dx) = linear_solver.solve(&residuals, &jac) {
                solving_duration = start.elapsed();
                let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
                self.apply_dx2(
                    &dx_na,
                    &mut new_param_blocks,
                    &variable_name_to_col_idx_dict,
                );
            } else {
                log::debug!("solve ax=b failed");
                termination = (TerminationType::Failure, "solve ax=b failed");
                break;
            }

            let new_error = problem.compute_cost(&new_param_blocks);
            if new_error.is_nan() {
                log::debug!("solve ax=b failed, current error is nan");
                termination = (TerminationType::Failure, "current error is nan");
                break;
            }
            parameter_blocks = new_param_blocks;
            current_error = new_error;
            num_successful_steps += 1;
            trace!(
                "iter:{}, total err:{}, residual + jacobian duration: {:?}, solving duration: {:?}",
                i, current_error, residual_and_jacobian_duration, solving_duration
            );

            if current_error < opt_option.min_error_threshold {
                trace!("error too low");
                termination = (TerminationType::Convergence, "error too low");
                break;
            }

            if (last_err - current_error).abs() < opt_option.min_abs_error_decrease_threshold {
                trace!("absolute error decrease low");
                termination = (TerminationType::Convergence, "absolute error decrease low");
                break;
            } else if last_err > 0.0
                && (last_err - current_error).abs() / last_err
                    < opt_option.min_rel_error_decrease_threshold
            {
                trace!("relative error decrease low");
                termination = (TerminationType::Convergence, "relative error decrease low");
                break;
            }
        }
        SolverSummary {
            parameters: parameter_blocks
                .iter()
                .map(|(k, v)| (k.to_owned(), v.params.clone()))
                .collect(),
            termination_type: termination.0,
            message: termination.1.to_string(),
            initial_cost,
            final_cost: current_error,
            num_successful_steps,
            num_unsuccessful_steps: 0,
            total_time: start_time.elapsed(),
        }
    }
}
