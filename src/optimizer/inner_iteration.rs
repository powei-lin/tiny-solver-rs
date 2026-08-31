use std::collections::HashMap;

use nalgebra as na;

use crate::ParameterBlockOrdering;
use crate::parameter_block::ParameterBlock;
use crate::problem::Problem;

#[derive(Clone, Debug)]
pub struct CoordinateDescentMinimizer {
    groups: Vec<Vec<String>>,
    max_num_iterations: usize,
    tolerance: f64,
}

impl CoordinateDescentMinimizer {
    pub fn new(
        problem: &Problem,
        parameter_blocks: &HashMap<String, ParameterBlock>,
        ordering: &ParameterBlockOrdering,
        max_num_iterations: usize,
        tolerance: f64,
    ) -> Result<Self, String> {
        if ordering.num_groups() == 0 {
            return Err("inner-iteration ordering must contain a group".to_string());
        }
        if max_num_iterations == 0 {
            return Err("max_num_inner_iterations must be positive".to_string());
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err("inner_iteration_tolerance must be finite and nonnegative".to_string());
        }
        problem.validate_inner_iteration_ordering(parameter_blocks, ordering)?;
        let groups = ordering
            .groups()
            .map(|(_, elements)| elements.iter().cloned().collect())
            .collect();
        Ok(Self {
            groups,
            max_num_iterations,
            tolerance,
        })
    }

    pub fn minimize(
        &self,
        problem: &Problem,
        parameter_blocks: &mut HashMap<String, ParameterBlock>,
    ) -> Result<bool, String> {
        let mut made_progress = false;
        for group in &self.groups {
            for parameter_name in group {
                made_progress |=
                    self.minimize_parameter(problem, parameter_blocks, parameter_name)?;
            }
        }
        Ok(made_progress)
    }

    fn minimize_parameter(
        &self,
        problem: &Problem,
        parameter_blocks: &mut HashMap<String, ParameterBlock>,
        parameter_name: &str,
    ) -> Result<bool, String> {
        let mut made_progress = false;
        for _ in 0..self.max_num_iterations {
            let (residuals, jacobian) = problem
                .compute_parameter_residual_and_jacobian(parameter_name, parameter_blocks)?;
            if residuals.is_empty() || jacobian.ncols() == 0 {
                break;
            }
            let old_cost = residuals.norm_squared();
            let gradient = jacobian.transpose() * &residuals;
            if gradient.amax() <= self.tolerance {
                break;
            }
            let mut normal = jacobian.transpose() * jacobian;
            for index in 0..normal.nrows() {
                normal[(index, index)] += 1e-8 * normal[(index, index)].max(1e-6);
            }
            let Some(cholesky) = normal.cholesky() else {
                break;
            };
            let step = cholesky.solve(&(-gradient));
            if !step.iter().all(|value| value.is_finite()) || step.norm() <= self.tolerance {
                break;
            }

            let previous = parameter_blocks[parameter_name].params.clone();
            apply_local_step(&step, parameter_blocks.get_mut(parameter_name).unwrap());
            let new_cost = problem
                .compute_parameter_residual_and_jacobian(parameter_name, parameter_blocks)?
                .0
                .norm_squared();
            if !new_cost.is_finite() || new_cost >= old_cost {
                parameter_blocks.get_mut(parameter_name).unwrap().params = previous;
                break;
            }
            made_progress = true;
            let relative_progress = (old_cost - new_cost) / old_cost.max(f64::MIN_POSITIVE);
            if relative_progress <= self.tolerance {
                break;
            }
        }
        Ok(made_progress)
    }
}

fn apply_local_step(step: &na::DVector<f64>, parameter: &mut ParameterBlock) {
    let tangent_size = parameter.tangent_size();
    let mut full_step = na::DVector::zeros(tangent_size);
    if parameter.manifold.is_some() {
        full_step.copy_from(step);
    } else {
        let mut source = 0;
        for target in 0..tangent_size {
            if !parameter.fixed_variables.contains(&target) {
                full_step[target] = step[source];
                source += 1;
            }
        }
    }
    let updated = parameter.plus_f64(full_step.as_view());
    parameter.update_params(updated);
}

#[cfg(test)]
mod tests {
    use crate::factors::PriorFactor;

    use super::*;

    #[test]
    fn minimizes_independent_parameter_blocks() {
        let mut problem = Problem::new();
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(PriorFactor {
                v: na::dvector![2.0],
            }),
            None,
        );
        problem.add_residual_block(
            1,
            &["y"],
            Box::new(PriorFactor {
                v: na::dvector![-1.0],
            }),
            None,
        );
        let initial_values = HashMap::from([
            ("x".to_string(), na::dvector![0.0]),
            ("y".to_string(), na::dvector![0.0]),
        ]);
        let mut parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let mut ordering = ParameterBlockOrdering::new();
        ordering.add_element_to_group("x", 0);
        ordering.add_element_to_group("y", 0);
        let minimizer =
            CoordinateDescentMinimizer::new(&problem, &parameter_blocks, &ordering, 5, 1e-12)
                .unwrap();

        assert!(minimizer.minimize(&problem, &mut parameter_blocks).unwrap());
        assert!((parameter_blocks["x"].params[0] - 2.0).abs() < 1e-8);
        assert!((parameter_blocks["y"].params[0] + 1.0).abs() < 1e-8);
    }
}
