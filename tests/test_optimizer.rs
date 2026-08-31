use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tiny_solver::factors::{Factor, PriorFactor};
use tiny_solver::{
    CallbackReturnType, DoglegType, EvaluationCallback, GaussNewtonOptimizer, IterationCallback,
    IterationSummary, LevenbergMarquardtOptimizer, LinearSolverType, Optimizer, OptimizerOptions,
    ParameterBlockOrdering, PreconditionerType, Problem, TerminationType, TrustRegionStrategyType,
    na,
};

struct StopAfterFirstIteration;

impl IterationCallback for StopAfterFirstIteration {
    fn on_iteration(&self, _summary: &IterationSummary) -> CallbackReturnType {
        CallbackReturnType::TerminateSuccessfully
    }
}

#[derive(Default)]
struct EvaluationCounter {
    residual_evaluations: AtomicUsize,
    jacobian_evaluations: AtomicUsize,
}

impl EvaluationCallback for EvaluationCounter {
    fn prepare_for_evaluation(&self, evaluate_jacobians: bool, _new_evaluation_point: bool) {
        self.residual_evaluations.fetch_add(1, Ordering::Relaxed);
        if evaluate_jacobians {
            self.jacobian_evaluations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn prior_problem() -> (Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = Problem::new();
    problem.add_residual_block(
        1,
        &["x"],
        Box::new(PriorFactor {
            v: na::dvector![3.0],
        }),
        None,
    );
    let initial_values = HashMap::from([("x".to_string(), na::dvector![0.0])]);
    (problem, initial_values)
}

struct PointCameraFactor;

impl<T: na::RealField> Factor<T> for PointCameraFactor {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        na::dvector![params[0][0].clone() + params[1][0].clone() - T::from_f64(3.0).unwrap()]
    }
}

struct RosenbrockFactor;

impl<T: na::RealField> Factor<T> for RosenbrockFactor {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let x = params[0][0].clone();
        let y = params[0][1].clone();
        na::dvector![
            T::from_f64(10.0).unwrap() * (y - x.clone() * x.clone()),
            T::one() - x
        ]
    }
}

fn schur_problem() -> (
    Problem,
    HashMap<String, na::DVector<f64>>,
    ParameterBlockOrdering,
) {
    let mut problem = Problem::new();
    problem.add_residual_block(1, &["point", "camera"], Box::new(PointCameraFactor), None);
    problem.add_residual_block(
        1,
        &["camera"],
        Box::new(PriorFactor {
            v: na::dvector![1.0],
        }),
        None,
    );
    let initial_values = HashMap::from([
        ("camera".to_string(), na::dvector![0.0]),
        ("point".to_string(), na::dvector![0.0]),
    ]);
    let mut ordering = ParameterBlockOrdering::new();
    ordering.add_element_to_group("point", 0);
    ordering.add_element_to_group("camera", 1);
    (problem, initial_values, ordering)
}

#[test]
fn gauss_newton_returns_a_detailed_summary() {
    let (problem, initial_values) = prior_problem();
    let result = GaussNewtonOptimizer::new().optimize_with_summary(
        &problem,
        &initial_values,
        Some(OptimizerOptions::default()),
    );

    assert_eq!(
        result.summary.termination_type,
        TerminationType::Convergence
    );
    assert_eq!(result.summary.initial_cost, 9.0);
    assert!(result.summary.final_cost < 1e-12);
    assert!(!result.summary.iterations.is_empty());
    assert!(result.parameters.is_some());
    assert!(result.summary.full_report().contains("Final cost"));
}

#[test]
fn callback_can_terminate_levenberg_marquardt_successfully() {
    let (problem, initial_values) = prior_problem();
    let mut options = OptimizerOptions::default();
    options.callbacks.push(Arc::new(StopAfterFirstIteration));

    let result = LevenbergMarquardtOptimizer::default().optimize_with_summary(
        &problem,
        &initial_values,
        Some(options),
    );

    assert_eq!(
        result.summary.termination_type,
        TerminationType::UserSuccess
    );
    assert_eq!(result.summary.iterations.len(), 1);
    assert!(result.summary.iterations[0].trust_region_radius.is_some());
    assert!(result.parameters.is_some());
}

#[test]
fn legacy_optimize_api_still_returns_parameters() {
    let (problem, initial_values) = prior_problem();
    let parameters = GaussNewtonOptimizer::new()
        .optimize(&problem, &initial_values, None)
        .unwrap();

    assert!((parameters["x"][0] - 3.0).abs() < 1e-12);
}

#[test]
fn dense_linear_solver_options_converge() {
    for linear_solver_type in [
        LinearSolverType::DenseQR,
        LinearSolverType::DenseNormalCholesky,
    ] {
        let (problem, initial_values) = prior_problem();
        let options = OptimizerOptions {
            linear_solver_type,
            ..OptimizerOptions::default()
        };
        let parameters = GaussNewtonOptimizer::new()
            .optimize(&problem, &initial_values, Some(options))
            .unwrap();

        assert!((parameters["x"][0] - 3.0).abs() < 1e-12);
    }
}

#[test]
fn schur_linear_solver_options_converge_with_both_optimizers() {
    for (linear_solver_type, preconditioner_type) in [
        (LinearSolverType::DenseSchur, PreconditionerType::Jacobi),
        (LinearSolverType::SparseSchur, PreconditionerType::Jacobi),
        (
            LinearSolverType::IterativeSchur,
            PreconditionerType::Identity,
        ),
        (LinearSolverType::IterativeSchur, PreconditionerType::Jacobi),
        (
            LinearSolverType::IterativeSchur,
            PreconditionerType::SchurJacobi,
        ),
    ] {
        for use_levenberg_marquardt in [false, true] {
            let (problem, initial_values, ordering) = schur_problem();
            let options = OptimizerOptions {
                linear_solver_type,
                parameter_block_ordering: Some(ordering),
                preconditioner_type,
                ..OptimizerOptions::default()
            };
            let result = if use_levenberg_marquardt {
                LevenbergMarquardtOptimizer::default().optimize_with_summary(
                    &problem,
                    &initial_values,
                    Some(options),
                )
            } else {
                GaussNewtonOptimizer::new().optimize_with_summary(
                    &problem,
                    &initial_values,
                    Some(options),
                )
            };

            let parameters = result.parameters.unwrap_or_else(|| {
                panic!(
                    "{linear_solver_type:?}/{preconditioner_type:?}, LM={use_levenberg_marquardt}: {}",
                    result.summary.message
                )
            });
            assert!((parameters["point"][0] - 2.0).abs() < 1e-6);
            assert!((parameters["camera"][0] - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn cgnr_preconditioners_converge_with_both_optimizers() {
    for preconditioner_type in [PreconditionerType::Identity, PreconditionerType::Jacobi] {
        for use_levenberg_marquardt in [false, true] {
            let (problem, initial_values, _) = schur_problem();
            let options = OptimizerOptions {
                linear_solver_type: LinearSolverType::Cgnr,
                preconditioner_type,
                ..OptimizerOptions::default()
            };
            let result = if use_levenberg_marquardt {
                LevenbergMarquardtOptimizer::default().optimize_with_summary(
                    &problem,
                    &initial_values,
                    Some(options),
                )
            } else {
                GaussNewtonOptimizer::new().optimize_with_summary(
                    &problem,
                    &initial_values,
                    Some(options),
                )
            };

            let parameters = result.parameters.unwrap_or_else(|| {
                panic!(
                    "CGNR/{preconditioner_type:?}, LM={use_levenberg_marquardt}: {}",
                    result.summary.message
                )
            });
            assert!((parameters["point"][0] - 2.0).abs() < 1e-6);
            assert!((parameters["camera"][0] - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn cgnr_rejects_schur_preconditioner() {
    let (problem, initial_values) = prior_problem();
    let options = OptimizerOptions {
        linear_solver_type: LinearSolverType::Cgnr,
        preconditioner_type: PreconditionerType::SchurJacobi,
        ..OptimizerOptions::default()
    };

    let result =
        GaussNewtonOptimizer::new().optimize_with_summary(&problem, &initial_values, Some(options));

    assert_eq!(result.summary.termination_type, TerminationType::Failure);
    assert!(result.parameters.is_none());
    assert!(result.summary.message.contains("Identity and Jacobi"));
}

#[test]
fn trust_region_strategies_converge_on_rosenbrock_valley() {
    for trust_region_strategy_type in [
        TrustRegionStrategyType::LevenbergMarquardt,
        TrustRegionStrategyType::Dogleg(DoglegType::Traditional),
        TrustRegionStrategyType::Dogleg(DoglegType::Subspace),
    ] {
        let mut problem = Problem::new();
        problem.add_residual_block(2, &["x"], Box::new(RosenbrockFactor), None);
        let initial_values = HashMap::from([("x".to_string(), na::dvector![-1.2, 1.0])]);
        let options = OptimizerOptions {
            max_iteration: 200,
            trust_region_strategy_type,
            min_abs_error_decrease_threshold: 1e-14,
            min_rel_error_decrease_threshold: 1e-14,
            min_error_threshold: 1e-14,
            ..OptimizerOptions::default()
        };
        let result = LevenbergMarquardtOptimizer::new(1e-6, 1e32, 1.0).optimize_with_summary(
            &problem,
            &initial_values,
            Some(options),
        );
        let parameters = result.parameters.unwrap_or_else(|| {
            panic!("{trust_region_strategy_type:?}: {}", result.summary.message)
        });

        assert!((parameters["x"][0] - 1.0).abs() < 1e-6);
        assert!((parameters["x"][1] - 1.0).abs() < 1e-6);
        assert!(result.summary.final_cost < 1e-12);
    }
}

#[test]
fn dogleg_rejects_iterative_linear_solvers() {
    let (problem, initial_values) = prior_problem();
    let options = OptimizerOptions {
        linear_solver_type: LinearSolverType::Cgnr,
        trust_region_strategy_type: TrustRegionStrategyType::Dogleg(DoglegType::Traditional),
        ..OptimizerOptions::default()
    };

    let result = LevenbergMarquardtOptimizer::default().optimize_with_summary(
        &problem,
        &initial_values,
        Some(options),
    );

    assert_eq!(result.summary.termination_type, TerminationType::Failure);
    assert!(result.parameters.is_none());
    assert!(result.summary.message.contains("exact factorization"));
}

#[test]
fn evaluation_callback_runs_before_residual_and_jacobian_evaluations() {
    let (problem, initial_values) = prior_problem();
    let counter = Arc::new(EvaluationCounter::default());
    let options = OptimizerOptions {
        evaluation_callback: Some(counter.clone()),
        ..OptimizerOptions::default()
    };

    GaussNewtonOptimizer::new().optimize_with_summary(&problem, &initial_values, Some(options));

    assert!(counter.residual_evaluations.load(Ordering::Relaxed) >= 3);
    assert!(counter.jacobian_evaluations.load(Ordering::Relaxed) >= 1);
}
