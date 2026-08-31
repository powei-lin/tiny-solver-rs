use nalgebra as na;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConjugateGradientOptions {
    pub min_num_iterations: usize,
    pub max_num_iterations: usize,
    pub residual_reset_period: usize,
    pub residual_tolerance: f64,
    pub q_tolerance: f64,
}

pub(crate) fn solve(
    rhs: &na::DVector<f64>,
    options: ConjugateGradientOptions,
    mut right_multiply: impl FnMut(&na::DVector<f64>) -> Option<na::DVector<f64>>,
    mut precondition: impl FnMut(&na::DVector<f64>) -> na::DVector<f64>,
) -> Option<na::DVector<f64>> {
    if options.max_num_iterations == 0
        || options.min_num_iterations > options.max_num_iterations
        || options.residual_reset_period == 0
        || !options.residual_tolerance.is_finite()
        || !options.q_tolerance.is_finite()
        || options.q_tolerance < 0.0
    {
        return None;
    }

    let mut solution = na::DVector::zeros(rhs.len());
    let norm_rhs = rhs.norm();
    if norm_rhs == 0.0 {
        return Some(solution);
    }

    let tolerance = options.residual_tolerance * norm_rhs;
    let mut residual = rhs.clone();
    if options.residual_tolerance >= 0.0
        && options.min_num_iterations == 0
        && residual.norm() <= tolerance
    {
        return Some(solution);
    }

    let mut preconditioned = precondition(&residual);
    let mut rho = residual.dot(&preconditioned);
    if rho <= 0.0 || !rho.is_finite() {
        return None;
    }
    let mut direction = preconditioned.clone();
    let mut model_cost = 0.0;

    for iteration in 1..=options.max_num_iterations {
        let product = right_multiply(&direction)?;
        let curvature = direction.dot(&product);
        if curvature <= 0.0 || !curvature.is_finite() {
            return None;
        }
        let alpha = rho / curvature;
        if !alpha.is_finite() {
            return None;
        }
        solution.axpy(alpha, &direction, 1.0);

        if iteration % options.residual_reset_period == 0 {
            residual = rhs - right_multiply(&solution)?;
        } else {
            residual.axpy(-alpha, &product, 1.0);
        }
        let next_model_cost = -solution.dot(&(rhs + &residual));
        let zeta = iteration as f64 * (next_model_cost - model_cost) / next_model_cost;
        if iteration >= options.min_num_iterations && zeta < options.q_tolerance {
            return Some(solution);
        }
        model_cost = next_model_cost;

        let residual_norm = residual.norm();
        if iteration >= options.min_num_iterations
            && (residual_norm == 0.0
                || (options.residual_tolerance >= 0.0 && residual_norm <= tolerance))
        {
            return Some(solution);
        }

        preconditioned = precondition(&residual);
        let next_rho = residual.dot(&preconditioned);
        if next_rho <= 0.0 || !next_rho.is_finite() {
            return None;
        }
        let beta = next_rho / rho;
        if !beta.is_finite() {
            return None;
        }
        direction *= beta;
        direction += &preconditioned;
        rho = next_rho;
    }

    Some(solution)
}
