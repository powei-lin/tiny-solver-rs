use nalgebra as na;

use super::{Factor, FactorImpl};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericDiffMethod {
    Forward,
    Central,
    Ridders,
}

#[derive(Clone, Debug)]
pub struct NumericDiffOptions {
    pub relative_step_size: f64,
    pub ridders_relative_initial_step_size: f64,
    pub max_num_ridders_extrapolations: usize,
    pub ridders_epsilon: f64,
    pub ridders_step_shrink_factor: f64,
}

impl Default for NumericDiffOptions {
    fn default() -> Self {
        Self {
            relative_step_size: 1e-6,
            ridders_relative_initial_step_size: 1e-2,
            max_num_ridders_extrapolations: 10,
            ridders_epsilon: 1e-12,
            ridders_step_shrink_factor: 2.0,
        }
    }
}

pub struct NumericDiffFactor<F> {
    factor: F,
    method: NumericDiffMethod,
    options: NumericDiffOptions,
}

impl<F> NumericDiffFactor<F> {
    pub fn new(factor: F, method: NumericDiffMethod) -> Self {
        Self::with_options(factor, method, NumericDiffOptions::default())
    }

    pub fn with_options(factor: F, method: NumericDiffMethod, options: NumericDiffOptions) -> Self {
        assert!(options.relative_step_size > 0.0);
        assert!(options.ridders_relative_initial_step_size > 0.0);
        assert!(options.max_num_ridders_extrapolations > 0);
        assert!(options.ridders_epsilon > 0.0);
        assert!(options.ridders_step_shrink_factor > 1.0);
        Self {
            factor,
            method,
            options,
        }
    }
}

impl<F: Factor<f64>> NumericDiffFactor<F> {
    fn evaluate_offset(
        &self,
        params: &[na::DVector<f64>],
        block: usize,
        parameter: usize,
        offset: f64,
    ) -> na::DVector<f64> {
        let mut offset_params = params.to_vec();
        offset_params[block][parameter] += offset;
        self.factor.residual_func(&offset_params)
    }

    fn central_column(
        &self,
        params: &[na::DVector<f64>],
        block: usize,
        parameter: usize,
        step: f64,
    ) -> na::DVector<f64> {
        (self.evaluate_offset(params, block, parameter, step)
            - self.evaluate_offset(params, block, parameter, -step))
            / (2.0 * step)
    }

    fn ridders_column(
        &self,
        params: &[na::DVector<f64>],
        block: usize,
        parameter: usize,
        step: f64,
    ) -> na::DVector<f64> {
        let shrink = self.options.ridders_step_shrink_factor;
        let mut current_step =
            step * shrink.powi((self.options.max_num_ridders_extrapolations / 2) as i32);
        let mut previous: Vec<na::DVector<f64>> = Vec::new();
        let mut best = self.central_column(params, block, parameter, current_step);
        let mut best_error = f64::MAX;

        for order in 0..self.options.max_num_ridders_extrapolations {
            let mut current = vec![self.central_column(params, block, parameter, current_step)];
            current_step /= shrink;
            let mut richardson_factor = shrink * shrink;
            for level in 1..=order {
                let candidate = (&current[level - 1] * richardson_factor - &previous[level - 1])
                    / (richardson_factor - 1.0);
                let candidate_error = (&candidate - &current[level - 1])
                    .norm()
                    .max((&candidate - &previous[level - 1]).norm());
                if candidate_error <= best_error {
                    best_error = candidate_error;
                    best = candidate.clone();
                }
                current.push(candidate);
                richardson_factor *= shrink * shrink;
            }

            if best_error < self.options.ridders_epsilon {
                break;
            }
            if order > 0 && (&current[order] - &previous[order - 1]).norm() >= 2.0 * best_error {
                break;
            }
            previous = current;
        }
        best
    }

    fn evaluate_with_jacobians(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, Vec<na::DMatrix<f64>>) {
        let residual = self.factor.residual_func(params);
        let minimum_step = f64::EPSILON.sqrt();
        let jacobians = params
            .iter()
            .enumerate()
            .map(|(block, parameter_block)| {
                na::DMatrix::from_columns(
                    &(0..parameter_block.len())
                        .map(|parameter| {
                            let relative_step = match self.method {
                                NumericDiffMethod::Ridders => {
                                    self.options.ridders_relative_initial_step_size
                                }
                                NumericDiffMethod::Forward | NumericDiffMethod::Central => {
                                    self.options.relative_step_size
                                }
                            };
                            let minimum_step = match self.method {
                                NumericDiffMethod::Ridders => minimum_step
                                    .max(self.options.ridders_relative_initial_step_size),
                                NumericDiffMethod::Forward | NumericDiffMethod::Central => {
                                    minimum_step
                                }
                            };
                            let step = (parameter_block[parameter].abs() * relative_step)
                                .max(minimum_step);
                            match self.method {
                                NumericDiffMethod::Forward => {
                                    (self.evaluate_offset(params, block, parameter, step)
                                        - &residual)
                                        / step
                                }
                                NumericDiffMethod::Central => {
                                    self.central_column(params, block, parameter, step)
                                }
                                NumericDiffMethod::Ridders => {
                                    self.ridders_column(params, block, parameter, step)
                                }
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        (residual, jacobians)
    }
}

impl<F: Factor<f64>> FactorImpl for NumericDiffFactor<F> {
    fn residual_func_dual(
        &self,
        _params: &[na::DVector<num_dual::DualDVec64>],
    ) -> na::DVector<num_dual::DualDVec64> {
        panic!("numeric differentiation factors do not evaluate dual numbers")
    }

    fn residual_func_f64(&self, params: &[na::DVector<f64>]) -> na::DVector<f64> {
        self.factor.residual_func(params)
    }

    fn residual_and_jacobians_f64(
        &self,
        params: &[na::DVector<f64>],
    ) -> Option<(na::DVector<f64>, Vec<na::DMatrix<f64>>)> {
        Some(self.evaluate_with_jacobians(params))
    }
}
