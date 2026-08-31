use std::ops::Mul;

use nalgebra as na;

use crate::SparseLinearSolver;

const MIN_DOGLEG_MU: f64 = 1e-8;
const MAX_DOGLEG_MU: f64 = 1.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DoglegType {
    #[default]
    Traditional,
    Subspace,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrustRegionStrategyType {
    #[default]
    LevenbergMarquardt,
    Dogleg(DoglegType),
}

#[derive(Clone, Debug)]
pub struct TrustRegionStep {
    pub step: na::DVector<f64>,
    pub scaled_step_norm: f64,
}

pub trait TrustRegionStrategy {
    fn compute_step(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
        linear_solver: &mut dyn SparseLinearSolver,
    ) -> Option<TrustRegionStep>;

    fn step_accepted(&mut self, step_quality: f64);
    fn step_rejected(&mut self);
    fn radius(&self) -> f64;
}

#[derive(Clone, Debug)]
pub struct LevenbergMarquardtStrategy {
    radius: f64,
    max_radius: f64,
    min_diagonal: f64,
    max_diagonal: f64,
    decrease_factor: f64,
    diagonal_squared: Option<na::DVector<f64>>,
    reuse_diagonal: bool,
}

impl LevenbergMarquardtStrategy {
    pub fn new(initial_radius: f64, max_radius: f64, min_diagonal: f64, max_diagonal: f64) -> Self {
        assert!(initial_radius > 0.0);
        assert!(max_radius >= initial_radius);
        assert!(min_diagonal > 0.0);
        assert!(max_diagonal >= min_diagonal);
        Self {
            radius: initial_radius,
            max_radius,
            min_diagonal,
            max_diagonal,
            decrease_factor: 2.0,
            diagonal_squared: None,
            reuse_diagonal: false,
        }
    }
}

impl TrustRegionStrategy for LevenbergMarquardtStrategy {
    fn compute_step(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
        linear_solver: &mut dyn SparseLinearSolver,
    ) -> Option<TrustRegionStep> {
        if !self.reuse_diagonal
            || self
                .diagonal_squared
                .as_ref()
                .is_none_or(|diagonal| diagonal.len() != jacobian.ncols())
        {
            self.diagonal_squared = Some(na::DVector::from_fn(jacobian.ncols(), |column, _| {
                jacobian
                    .val_of_col(column)
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .max(self.min_diagonal)
                    .min(self.max_diagonal)
            }));
        }
        let diagonal_squared = self.diagonal_squared.as_ref()?;
        let regularization: Vec<_> = diagonal_squared
            .iter()
            .map(|value| value / self.radius)
            .collect();
        let step = linear_solver.solve_regularized(residuals, jacobian, &regularization)?;
        let step = na::DVector::from_fn(jacobian.ncols(), |row, _| step[(row, 0)]);
        let scaled_step_norm = diagonal_squared
            .iter()
            .zip(step.iter())
            .map(|(diagonal, value)| diagonal * value * value)
            .sum::<f64>()
            .sqrt();
        self.reuse_diagonal = true;
        Some(TrustRegionStep {
            step,
            scaled_step_norm,
        })
    }

    fn step_accepted(&mut self, step_quality: f64) {
        let tmp = 2.0 * step_quality - 1.0;
        self.radius /= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
        self.radius = self.radius.min(self.max_radius);
        self.decrease_factor = 2.0;
        self.reuse_diagonal = false;
    }

    fn step_rejected(&mut self) {
        self.radius /= self.decrease_factor;
        self.decrease_factor *= 2.0;
        self.reuse_diagonal = true;
    }

    fn radius(&self) -> f64 {
        self.radius
    }
}

#[derive(Clone, Debug)]
pub struct DoglegStrategy {
    dogleg_type: DoglegType,
    radius: f64,
    max_radius: f64,
    min_diagonal: f64,
    max_diagonal: f64,
    mu: f64,
    last_step_norm: f64,
}

impl DoglegStrategy {
    pub fn new(
        dogleg_type: DoglegType,
        initial_radius: f64,
        max_radius: f64,
        min_diagonal: f64,
        max_diagonal: f64,
    ) -> Self {
        assert!(initial_radius > 0.0);
        assert!(max_radius >= initial_radius);
        assert!(min_diagonal > 0.0);
        assert!(max_diagonal >= min_diagonal);
        Self {
            dogleg_type,
            radius: initial_radius,
            max_radius,
            min_diagonal,
            max_diagonal,
            mu: MIN_DOGLEG_MU,
            last_step_norm: 0.0,
        }
    }

    fn traditional_step(
        &self,
        gradient: &na::DVector<f64>,
        alpha: f64,
        gauss_newton: &na::DVector<f64>,
    ) -> na::DVector<f64> {
        let gradient_norm = gradient.norm();
        let gauss_newton_norm = gauss_newton.norm();
        if gauss_newton_norm <= self.radius {
            return gauss_newton.clone();
        }
        if alpha * gradient_norm >= self.radius {
            return -(self.radius / gradient_norm) * gradient;
        }

        let cauchy = -alpha * gradient;
        let segment = gauss_newton - &cauchy;
        let c = cauchy.dot(&segment);
        let segment_squared_norm = segment.norm_squared();
        let discriminant = (c * c
            + segment_squared_norm * (self.radius * self.radius - cauchy.norm_squared()))
        .sqrt();
        let beta = if c <= 0.0 {
            (discriminant - c) / segment_squared_norm
        } else {
            (self.radius * self.radius - cauchy.norm_squared()) / (discriminant + c)
        };
        cauchy + beta * segment
    }

    fn subspace_step(
        &self,
        gradient: &na::DVector<f64>,
        gauss_newton: &na::DVector<f64>,
        diagonal: &na::DVector<f64>,
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<na::DVector<f64>> {
        if gauss_newton.norm() <= self.radius {
            return Some(gauss_newton.clone());
        }

        let first_basis = gradient / gradient.norm();
        let orthogonal = gauss_newton - &first_basis * first_basis.dot(gauss_newton);
        if orthogonal.norm() <= 1e-12 * gauss_newton.norm().max(1.0) {
            return Some(-(self.radius / gradient.norm()) * gradient);
        }
        let second_basis = &orthogonal / orthogonal.norm();
        let first_image = jacobian_product(jacobian, &first_basis.component_div(diagonal))?;
        let second_image = jacobian_product(jacobian, &second_basis.component_div(diagonal))?;
        let model = na::Matrix2::new(
            first_image.dot(&first_image),
            first_image.dot(&second_image),
            first_image.dot(&second_image),
            second_image.dot(&second_image),
        );
        let subspace_gradient =
            na::Vector2::new(first_basis.dot(gradient), second_basis.dot(gradient));
        let minimum = boundary_subspace_minimum(&model, &subspace_gradient, self.radius)?;
        Some(first_basis * minimum[0] + second_basis * minimum[1])
    }
}

impl TrustRegionStrategy for DoglegStrategy {
    fn compute_step(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
        linear_solver: &mut dyn SparseLinearSolver,
    ) -> Option<TrustRegionStep> {
        let diagonal_squared = na::DVector::from_fn(jacobian.ncols(), |column, _| {
            jacobian
                .val_of_col(column)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .max(self.min_diagonal)
                .min(self.max_diagonal)
        });
        let diagonal = diagonal_squared.map(f64::sqrt);
        let gradient_faer = jacobian.as_ref().transpose().mul(residuals);
        let gradient = na::DVector::from_fn(jacobian.ncols(), |row, _| {
            gradient_faer[(row, 0)] / diagonal[row]
        });
        let gradient_norm = gradient.norm();
        if gradient_norm == 0.0 {
            return Some(TrustRegionStep {
                step: na::DVector::zeros(jacobian.ncols()),
                scaled_step_norm: 0.0,
            });
        }
        let scaled_gradient = gradient.component_div(&diagonal);
        let gradient_image = jacobian_product(jacobian, &scaled_gradient)?;
        let alpha = gradient.norm_squared() / gradient_image.norm_squared();

        let gauss_newton_actual = loop {
            let regularization: Vec<_> = diagonal_squared
                .iter()
                .map(|value| self.mu * value)
                .collect();
            if let Some(step) =
                linear_solver.solve_regularized(residuals, jacobian, &regularization)
            {
                break na::DVector::from_fn(jacobian.ncols(), |row, _| step[(row, 0)]);
            }
            self.mu *= 10.0;
            if self.mu >= MAX_DOGLEG_MU {
                return None;
            }
        };
        let gauss_newton = gauss_newton_actual.component_mul(&diagonal);
        let scaled_step = match self.dogleg_type {
            DoglegType::Traditional => self.traditional_step(&gradient, alpha, &gauss_newton),
            DoglegType::Subspace => self
                .subspace_step(&gradient, &gauss_newton, &diagonal, jacobian)
                .unwrap_or_else(|| self.traditional_step(&gradient, alpha, &gauss_newton)),
        };
        self.last_step_norm = scaled_step.norm();
        Some(TrustRegionStep {
            step: scaled_step.component_div(&diagonal),
            scaled_step_norm: self.last_step_norm,
        })
    }

    fn step_accepted(&mut self, step_quality: f64) {
        if step_quality < 0.25 {
            self.radius *= 0.5;
        }
        if step_quality > 0.75 {
            self.radius = self
                .max_radius
                .min(self.radius.max(3.0 * self.last_step_norm));
        }
        self.mu = MIN_DOGLEG_MU.max(self.mu / 5.0);
    }

    fn step_rejected(&mut self) {
        self.radius *= 0.5;
    }

    fn radius(&self) -> f64 {
        self.radius
    }
}

fn jacobian_product(
    jacobian: &faer::sparse::SparseColMat<usize, f64>,
    vector: &na::DVector<f64>,
) -> Option<na::DVector<f64>> {
    if vector.len() != jacobian.ncols() {
        return None;
    }
    let vector = faer::Mat::from_fn(vector.len(), 1, |row, _| vector[row]);
    let product = jacobian.as_ref().mul(vector.as_ref());
    Some(na::DVector::from_fn(product.nrows(), |row, _| {
        product[(row, 0)]
    }))
}

fn boundary_subspace_minimum(
    model: &na::Matrix2<f64>,
    gradient: &na::Vector2<f64>,
    radius: f64,
) -> Option<na::Vector2<f64>> {
    let shifted_solution = |shift: f64| {
        (model + na::Matrix2::identity() * shift)
            .lu()
            .solve(&(-gradient))
    };
    let mut upper = 1.0;
    while shifted_solution(upper)?.norm() > radius {
        upper *= 2.0;
        if !upper.is_finite() {
            return None;
        }
    }
    let mut lower = 0.0;
    for _ in 0..80 {
        let middle = 0.5 * (lower + upper);
        if shifted_solution(middle)?.norm() > radius {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    shifted_solution(upper)
}

#[cfg(test)]
mod tests {
    use faer::sparse::Triplet;

    use super::*;
    use crate::DenseQRSolver;

    fn diagonal_problem() -> (faer::sparse::SparseColMat<usize, f64>, faer::Mat<f64>) {
        let diagonal = [
            1.0,
            2.0_f64.sqrt(),
            2.0,
            8.0_f64.sqrt(),
            4.0,
            32.0_f64.sqrt(),
        ];
        let jacobian = faer::sparse::SparseColMat::try_new_from_triplets(
            6,
            6,
            &diagonal
                .iter()
                .enumerate()
                .map(|(index, &value)| Triplet::new(index, index, value))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let residuals = faer::Mat::from_fn(6, 1, |row, _| -diagonal[row]);
        (jacobian, residuals)
    }

    #[test]
    fn dogleg_types_obey_radius_and_recover_gauss_newton_step() {
        let (jacobian, residuals) = diagonal_problem();
        for dogleg_type in [DoglegType::Traditional, DoglegType::Subspace] {
            let mut constrained = DoglegStrategy::new(dogleg_type, 2.0, 2.0, 1.0, 1.0);
            let step = constrained
                .compute_step(&residuals, &jacobian, &mut DenseQRSolver::new())
                .unwrap();
            assert!(step.scaled_step_norm <= 2.0 * (1.0 + 1e-12));
            constrained.step_rejected();
            assert!((constrained.radius() - 1.0).abs() < 1e-12);
            constrained.step_accepted(0.8);
            assert!((constrained.radius() - 2.0).abs() < 1e-12);

            let mut unconstrained = DoglegStrategy::new(dogleg_type, 10.0, 10.0, 1.0, 1.0);
            let step = unconstrained
                .compute_step(&residuals, &jacobian, &mut DenseQRSolver::new())
                .unwrap();
            assert!((step.step - na::DVector::from_element(6, 1.0)).norm() < 1e-6);
        }
    }

    #[test]
    fn subspace_dogleg_handles_one_dimensional_model() {
        let (jacobian, _) = diagonal_problem();
        let residuals = faer::Mat::from_fn(6, 1, |row, _| if row == 2 { -4.0 } else { 0.0 });
        let mut strategy = DoglegStrategy::new(DoglegType::Subspace, 0.25, 0.25, 1.0, 1.0);
        let step = strategy
            .compute_step(&residuals, &jacobian, &mut DenseQRSolver::new())
            .unwrap();

        assert!((step.step[2] - 0.25).abs() < 1e-6);
        assert!(step.step.remove_row(2).norm() < 1e-12);
    }
}
