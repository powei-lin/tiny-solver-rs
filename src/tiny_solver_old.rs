use nalgebra as na;
use num_dual;
use std::ops::Mul;

pub struct SolverParameters {
    gradient_threshold: f64,
    relative_step_threshold: f64,
    error_threshold: f64,
    initial_scale_factor: f64,
    max_iterations: usize,
}
impl SolverParameters {
    pub fn defualt() -> SolverParameters {
        SolverParameters {
            gradient_threshold: 1e-16,
            relative_step_threshold: 1e-16,
            error_threshold: 1e-16,
            initial_scale_factor: 1e-3,
            max_iterations: 100,
        }
    }
}

#[derive(PartialEq, Debug)]
enum SolverStatus {
    Running,
    // Resulting solution may be OK to use.
    GradientTooSmall,         // eps > max(J'*f(x))
    RelativeStepSizeTooSmall, // eps > ||dx|| / ||x||
    ErrorTooSmall,            // eps > ||f(x)||
    HitMaxIterations,
    // Numerical issues
    // FAILED_TO_EVALUATE_COST_FUNCTION,
    // FAILED_TO_SOLVER_LINEAR_SYSTEM,
}

#[derive(Debug)]
pub struct ProblemResult {
    error_magnitude: f64,    // ||f(x)||
    gradient_magnitude: f64, // ||J'f(x)||
    num_failed_linear_solves: usize,
    iterations: usize,
    status: SolverStatus,
}
impl ProblemResult {
    fn new() -> ProblemResult {
        ProblemResult {
            error_magnitude: 0.0,
            gradient_magnitude: 0.0,
            num_failed_linear_solves: 0,
            iterations: 0,
            status: SolverStatus::Running,
        }
    }
}

pub trait TinySolver<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    fn cost_function(
        _params: na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_PARAMETERS>,
    ) -> na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_RESIDUALS>;

    fn solve_inplace(params: &mut na::SVector<f64, NUM_PARAMETERS>) -> ProblemResult {
        let solver_params = SolverParameters::defualt();
        let mut result = ProblemResult::new();
        let mut u: f64 = 0.0;
        let mut v = 2;
        let mut residual = na::SMatrix::<f64, NUM_RESIDUALS, 1>::zeros();
        let mut gradient = na::SMatrix::<f64, NUM_PARAMETERS, 1>::zeros();
        let mut jac: na::SMatrix<f64, NUM_RESIDUALS, NUM_PARAMETERS>;

        for step in 0..solver_params.max_iterations {
            result.iterations = step + 1;

            // This explicitly computes the normal equations, which is numerically
            // unstable. Nevertheless, it is often good enough and is fast.
            (residual, jac) = num_dual::jacobian(Self::cost_function, params.clone());
            gradient = jac.transpose().mul(-residual);
            let jtj = jac.transpose().mul(jac);
            // println!("residual \n{}\n", residual);
            // println!("jac \n{}\n", &jac);
            // println!("H:\n{}", jtj);
            // println!("gradient:\n{}", step);

            let max_gradient = gradient.abs().max();
            if max_gradient < solver_params.gradient_threshold {
                println!("gradient too small. {}", max_gradient);
                result.status = SolverStatus::GradientTooSmall;
                break;
            } else if residual.norm() < solver_params.error_threshold {
                result.status = SolverStatus::ErrorTooSmall;
                break;
            }

            // initialize u and v
            if step == 0 {
                u = solver_params.initial_scale_factor * jtj.diagonal().max();
                v = 2;
            }
            // println!("u: {}", u);
            let mut jtj_augmented = na::DMatrix::<f64>::zeros(NUM_PARAMETERS, NUM_PARAMETERS);
            jtj_augmented.copy_from(&jtj);
            jtj_augmented.set_diagonal(&jtj_augmented.diagonal().add_scalar(u));

            // println!("jtj {}", jtj_augmented);
            let dx = na::linalg::LU::new(jtj_augmented.clone())
                .solve(&gradient)
                .unwrap();
            let solution: na::SMatrix<f64, NUM_PARAMETERS, 1> = jtj_augmented.fixed_view(0, 0) * dx;
            let solved = (solution - gradient).abs().min() < solver_params.error_threshold;
            if solved {
                if dx.norm() < solver_params.relative_step_threshold * params.norm() {
                    result.status = SolverStatus::RelativeStepSizeTooSmall;
                    break;
                }
                let param_new = *params + dx;
                // Rho is the ratio of the actual reduction in error to the reduction
                // in error that would be obtained if the problem was linear. See [1]
                // for details.
                // TODO: Error handling on user eval.
                let residual_new =
                    Self::cost_function(param_new.map(num_dual::DualSVec64::from_re)).map(|x| x.re);
                let rho: f64 = (residual.norm_squared() - residual_new.norm_squared())
                    / dx.dot(&(u * dx + gradient));
                if rho > 0.0 {
                    // Accept the Gauss-Newton step because the linear model fits well.
                    *params = param_new;
                    let tmp: f64 = 2.0 * rho - 1.0;
                    u = u * (1.0_f64 / 3.0).max(1.0 - tmp.powi(3));
                    v = 2;
                    continue;
                }
            } else {
                result.num_failed_linear_solves += 1;
                println!("fail {}", solution - gradient);
            }
            // Reject the update because either the normal equations failed to solve
            // or the local linear model was not good (rho < 0). Instead, increase u
            // to move closer to gradient descent.
            u *= v as f64;
            v *= 2;
        }
        if result.status == SolverStatus::Running {
            result.status = SolverStatus::HitMaxIterations;
        }
        result.error_magnitude = residual.norm();
        result.gradient_magnitude = gradient.norm();

        result
    }
}
