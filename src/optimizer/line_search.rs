use std::collections::VecDeque;

use nalgebra as na;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineSearchType {
    Armijo,
    #[default]
    Wolfe,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NonlinearConjugateGradientType {
    #[default]
    FletcherReeves,
    PolakRibiere,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineSearchDirectionType {
    SteepestDescent,
    NonlinearConjugateGradient(NonlinearConjugateGradientType),
    #[default]
    Lbfgs,
}

#[derive(Clone, Debug)]
struct LbfgsUpdate {
    parameter_change: na::DVector<f64>,
    gradient_change: na::DVector<f64>,
    inverse_curvature: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct LineSearchDirection {
    direction_type: LineSearchDirectionType,
    max_lbfgs_rank: usize,
    descent_tolerance: f64,
    previous_gradient: Option<na::DVector<f64>>,
    previous_direction: Option<na::DVector<f64>>,
    lbfgs_history: VecDeque<LbfgsUpdate>,
}

impl LineSearchDirection {
    pub(crate) fn new(
        direction_type: LineSearchDirectionType,
        max_lbfgs_rank: usize,
        descent_tolerance: f64,
    ) -> Option<Self> {
        if !descent_tolerance.is_finite()
            || descent_tolerance < 0.0
            || (direction_type == LineSearchDirectionType::Lbfgs && max_lbfgs_rank == 0)
        {
            return None;
        }
        Some(Self {
            direction_type,
            max_lbfgs_rank,
            descent_tolerance,
            previous_gradient: None,
            previous_direction: None,
            lbfgs_history: VecDeque::with_capacity(max_lbfgs_rank),
        })
    }

    pub(crate) fn next(
        &mut self,
        gradient: &na::DVector<f64>,
        previous_step: Option<&na::DVector<f64>>,
    ) -> Option<na::DVector<f64>> {
        let mut direction = match self.direction_type {
            LineSearchDirectionType::SteepestDescent => -gradient,
            LineSearchDirectionType::NonlinearConjugateGradient(method) => {
                self.nonlinear_conjugate_gradient(gradient, method)
            }
            LineSearchDirectionType::Lbfgs => self.lbfgs(gradient, previous_step),
        };
        if !direction.iter().all(|value| value.is_finite())
            || direction.dot(gradient) > -self.descent_tolerance
        {
            direction = -gradient;
            if self.direction_type == LineSearchDirectionType::Lbfgs {
                self.lbfgs_history.clear();
            }
        }
        if direction.norm_squared() == 0.0 {
            return None;
        }
        self.previous_gradient = Some(gradient.clone());
        self.previous_direction = Some(direction.clone());
        Some(direction)
    }

    fn nonlinear_conjugate_gradient(
        &self,
        gradient: &na::DVector<f64>,
        method: NonlinearConjugateGradientType,
    ) -> na::DVector<f64> {
        let (Some(previous_gradient), Some(previous_direction)) =
            (&self.previous_gradient, &self.previous_direction)
        else {
            return -gradient;
        };
        let denominator = previous_gradient.norm_squared();
        if denominator == 0.0 {
            return -gradient;
        }
        let beta = match method {
            NonlinearConjugateGradientType::FletcherReeves => gradient.norm_squared() / denominator,
            NonlinearConjugateGradientType::PolakRibiere => {
                gradient.dot(&(gradient - previous_gradient)) / denominator
            }
        };
        -gradient + beta * previous_direction
    }

    fn lbfgs(
        &mut self,
        gradient: &na::DVector<f64>,
        previous_step: Option<&na::DVector<f64>>,
    ) -> na::DVector<f64> {
        if let (Some(previous_gradient), Some(parameter_change)) =
            (&self.previous_gradient, previous_step)
        {
            let gradient_change = gradient - previous_gradient;
            let curvature = parameter_change.dot(&gradient_change);
            if curvature > 1e-14 {
                if self.lbfgs_history.len() == self.max_lbfgs_rank {
                    self.lbfgs_history.pop_front();
                }
                self.lbfgs_history.push_back(LbfgsUpdate {
                    parameter_change: parameter_change.clone(),
                    gradient_change,
                    inverse_curvature: 1.0 / curvature,
                });
            }
        }
        if self.lbfgs_history.is_empty() {
            return -gradient;
        }

        let mut value = gradient.clone();
        let mut alphas = Vec::with_capacity(self.lbfgs_history.len());
        for update in self.lbfgs_history.iter().rev() {
            let alpha = update.inverse_curvature * update.parameter_change.dot(&value);
            value -= alpha * &update.gradient_change;
            alphas.push(alpha);
        }
        let latest = self.lbfgs_history.back().unwrap();
        let scale = latest.parameter_change.dot(&latest.gradient_change)
            / latest.gradient_change.norm_squared();
        value *= scale;
        for (update, &alpha) in self.lbfgs_history.iter().zip(alphas.iter().rev()) {
            let beta = update.inverse_curvature * update.gradient_change.dot(&value);
            value += (alpha - beta) * &update.parameter_change;
        }
        -value
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineSearchParameters {
    pub sufficient_decrease: f64,
    pub sufficient_curvature_decrease: f64,
    pub max_step_contraction: f64,
    pub min_step_contraction: f64,
    pub min_step_size: f64,
    pub max_step_expansion: f64,
    pub max_num_iterations: usize,
}

impl Default for LineSearchParameters {
    fn default() -> Self {
        Self {
            sufficient_decrease: 1e-4,
            sufficient_curvature_decrease: 0.9,
            max_step_contraction: 1e-3,
            min_step_contraction: 0.6,
            min_step_size: 1e-9,
            max_step_expansion: 10.0,
            max_num_iterations: 20,
        }
    }
}

impl LineSearchParameters {
    pub(crate) fn is_valid(&self) -> bool {
        self.sufficient_decrease > 0.0
            && self.sufficient_decrease < 1.0
            && self.sufficient_curvature_decrease > self.sufficient_decrease
            && self.sufficient_curvature_decrease < 1.0
            && self.max_step_contraction > 0.0
            && self.max_step_contraction < self.min_step_contraction
            && self.min_step_contraction <= 1.0
            && self.min_step_size > 0.0
            && self.max_step_expansion > 1.0
            && self.max_num_iterations > 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineSearchSample {
    pub step_size: f64,
    pub cost: f64,
    pub directional_derivative: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct LineSearchResult {
    pub sample: LineSearchSample,
    pub iterations: usize,
    pub satisfies_wolfe: bool,
}

pub fn search(
    line_search_type: LineSearchType,
    initial: LineSearchSample,
    initial_step_size: f64,
    direction_infinity_norm: f64,
    parameters: LineSearchParameters,
    mut evaluate: impl FnMut(f64) -> Option<LineSearchSample>,
) -> Option<LineSearchResult> {
    if initial.step_size != 0.0
        || initial.directional_derivative >= 0.0
        || initial_step_size <= 0.0
        || direction_infinity_norm <= 0.0
        || !parameters.is_valid()
    {
        return None;
    }
    match line_search_type {
        LineSearchType::Armijo => armijo(
            initial,
            initial_step_size,
            direction_infinity_norm,
            parameters,
            &mut evaluate,
        ),
        LineSearchType::Wolfe => wolfe(
            initial,
            initial_step_size,
            direction_infinity_norm,
            parameters,
            &mut evaluate,
        ),
    }
}

fn satisfies_armijo(
    initial: LineSearchSample,
    sample: LineSearchSample,
    sufficient_decrease: f64,
) -> bool {
    sample.cost
        <= initial.cost + sufficient_decrease * initial.directional_derivative * sample.step_size
}

fn satisfies_wolfe(
    initial: LineSearchSample,
    sample: LineSearchSample,
    sufficient_curvature_decrease: f64,
) -> bool {
    sample.directional_derivative.abs()
        <= sufficient_curvature_decrease * initial.directional_derivative.abs()
}

fn armijo(
    initial: LineSearchSample,
    mut step_size: f64,
    direction_infinity_norm: f64,
    parameters: LineSearchParameters,
    evaluate: &mut impl FnMut(f64) -> Option<LineSearchSample>,
) -> Option<LineSearchResult> {
    for iteration in 1..=parameters.max_num_iterations {
        let sample = evaluate(step_size)?;
        if sample.cost.is_finite()
            && satisfies_armijo(initial, sample, parameters.sufficient_decrease)
        {
            return Some(LineSearchResult {
                sample,
                iterations: iteration,
                satisfies_wolfe: satisfies_wolfe(
                    initial,
                    sample,
                    parameters.sufficient_curvature_decrease,
                ),
            });
        }

        let denominator =
            2.0 * (sample.cost - initial.cost - initial.directional_derivative * sample.step_size);
        let interpolated =
            -initial.directional_derivative * sample.step_size * sample.step_size / denominator;
        let fallback = 0.5 * sample.step_size;
        let candidate = if interpolated.is_finite() && interpolated > 0.0 {
            interpolated
        } else {
            fallback
        };
        step_size = candidate.clamp(
            parameters.max_step_contraction * sample.step_size,
            parameters.min_step_contraction * sample.step_size,
        );
        if step_size * direction_infinity_norm < parameters.min_step_size {
            return None;
        }
    }
    None
}

fn wolfe(
    initial: LineSearchSample,
    initial_step_size: f64,
    direction_infinity_norm: f64,
    parameters: LineSearchParameters,
    evaluate: &mut impl FnMut(f64) -> Option<LineSearchSample>,
) -> Option<LineSearchResult> {
    let mut previous = initial;
    let mut best_armijo = None;
    let mut step_size = initial_step_size;
    for iteration in 1..=parameters.max_num_iterations {
        let current = evaluate(step_size)?;
        let current_armijo = current.cost.is_finite()
            && satisfies_armijo(initial, current, parameters.sufficient_decrease);
        if current_armijo
            && best_armijo.is_none_or(|best: LineSearchSample| current.cost < best.cost)
        {
            best_armijo = Some(current);
        }
        if !current_armijo || (iteration > 1 && current.cost >= previous.cost) {
            return zoom(
                initial,
                previous,
                current,
                iteration,
                direction_infinity_norm,
                parameters,
                best_armijo,
                evaluate,
            );
        }
        if satisfies_wolfe(initial, current, parameters.sufficient_curvature_decrease) {
            return Some(LineSearchResult {
                sample: current,
                iterations: iteration,
                satisfies_wolfe: true,
            });
        }
        if current.directional_derivative >= 0.0 {
            return zoom(
                initial,
                current,
                previous,
                iteration,
                direction_infinity_norm,
                parameters,
                best_armijo,
                evaluate,
            );
        }
        previous = current;
        step_size *= parameters.max_step_expansion;
    }
    best_armijo.map(|sample| LineSearchResult {
        sample,
        iterations: parameters.max_num_iterations,
        satisfies_wolfe: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn zoom(
    initial: LineSearchSample,
    mut low: LineSearchSample,
    mut high: LineSearchSample,
    completed_iterations: usize,
    direction_infinity_norm: f64,
    parameters: LineSearchParameters,
    mut best_armijo: Option<LineSearchSample>,
    evaluate: &mut impl FnMut(f64) -> Option<LineSearchSample>,
) -> Option<LineSearchResult> {
    for iteration in completed_iterations + 1..=parameters.max_num_iterations {
        let step_size = 0.5 * (low.step_size + high.step_size);
        if (high.step_size - low.step_size).abs() * direction_infinity_norm
            < parameters.min_step_size
        {
            break;
        }
        let current = evaluate(step_size)?;
        let current_armijo = current.cost.is_finite()
            && satisfies_armijo(initial, current, parameters.sufficient_decrease);
        if current_armijo && best_armijo.is_none_or(|best| current.cost < best.cost) {
            best_armijo = Some(current);
        }
        if !current_armijo || current.cost >= low.cost {
            high = current;
        } else {
            if satisfies_wolfe(initial, current, parameters.sufficient_curvature_decrease) {
                return Some(LineSearchResult {
                    sample: current,
                    iterations: iteration,
                    satisfies_wolfe: true,
                });
            }
            if current.directional_derivative * (high.step_size - low.step_size) >= 0.0 {
                high = low;
            }
            low = current;
        }
    }
    best_armijo.map(|sample| LineSearchResult {
        sample,
        iterations: parameters.max_num_iterations,
        satisfies_wolfe: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(step_size: f64) -> Option<LineSearchSample> {
        Some(LineSearchSample {
            step_size,
            cost: (step_size - 2.0).powi(2),
            directional_derivative: 2.0 * (step_size - 2.0),
        })
    }

    #[test]
    fn armijo_contracts_to_sufficient_decrease() {
        let initial = quadratic(0.0).unwrap();
        let result = search(
            LineSearchType::Armijo,
            initial,
            10.0,
            1.0,
            LineSearchParameters::default(),
            quadratic,
        )
        .unwrap();

        assert!(satisfies_armijo(initial, result.sample, 1e-4));
        assert!(result.sample.cost < initial.cost);
        assert!(result.iterations > 1);
    }

    #[test]
    fn wolfe_zoom_satisfies_both_conditions() {
        let initial = quadratic(0.0).unwrap();
        let result = search(
            LineSearchType::Wolfe,
            initial,
            10.0,
            1.0,
            LineSearchParameters::default(),
            quadratic,
        )
        .unwrap();

        assert!(satisfies_armijo(initial, result.sample, 1e-4));
        assert!(satisfies_wolfe(initial, result.sample, 0.9));
        assert!(result.satisfies_wolfe);
    }

    #[test]
    fn all_direction_methods_produce_descent_directions() {
        for direction_type in [
            LineSearchDirectionType::SteepestDescent,
            LineSearchDirectionType::NonlinearConjugateGradient(
                NonlinearConjugateGradientType::FletcherReeves,
            ),
            LineSearchDirectionType::NonlinearConjugateGradient(
                NonlinearConjugateGradientType::PolakRibiere,
            ),
            LineSearchDirectionType::Lbfgs,
        ] {
            let mut generator = LineSearchDirection::new(direction_type, 5, 1e-14).unwrap();
            let first_gradient = na::dvector![1.0, 10.0];
            let first_direction = generator.next(&first_gradient, None).unwrap();
            assert!(first_gradient.dot(&first_direction) < 0.0);

            let step = 0.1 * first_direction;
            let next_gradient = na::dvector![0.9, 0.0];
            let next_direction = generator.next(&next_gradient, Some(&step)).unwrap();
            assert!(next_gradient.dot(&next_direction) < 0.0);
        }
    }
}
