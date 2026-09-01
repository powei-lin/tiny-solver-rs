use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::time::Instant;

use tiny_solver::helper::read_bal;
use tiny_solver::{
    DoglegType, LinearSolverType, Optimizer, OptimizerOptions, PreconditionerType,
    TrustRegionOptimizer, TrustRegionStrategyType,
};

fn parse_solver(value: &str) -> Result<LinearSolverType, IoError> {
    match value {
        "dense_schur" => Ok(LinearSolverType::DenseSchur),
        "sparse_schur" => Ok(LinearSolverType::SparseSchur),
        "iterative_schur" => Ok(LinearSolverType::IterativeSchur),
        "sparse_normal_cholesky" => Ok(LinearSolverType::SparseCholesky),
        "dense_qr" => Ok(LinearSolverType::DenseQR),
        "dense_normal_cholesky" => Ok(LinearSolverType::DenseNormalCholesky),
        "cgnr" => Ok(LinearSolverType::Cgnr),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("unknown linear solver '{value}'"),
        )),
    }
}

fn parse_preconditioner(value: &str) -> Result<PreconditionerType, IoError> {
    match value {
        "identity" => Ok(PreconditionerType::Identity),
        "jacobi" => Ok(PreconditionerType::Jacobi),
        "schur_jacobi" => Ok(PreconditionerType::SchurJacobi),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("unknown preconditioner '{value}'"),
        )),
    }
}

fn parse_trust_region_strategy(value: &str) -> Result<TrustRegionStrategyType, IoError> {
    match value {
        "lm" => Ok(TrustRegionStrategyType::LevenbergMarquardt),
        "traditional_dogleg" => Ok(TrustRegionStrategyType::Dogleg(DoglegType::Traditional)),
        "subspace_dogleg" => Ok(TrustRegionStrategyType::Dogleg(DoglegType::Subspace)),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("unknown trust-region strategy '{value}'"),
        )),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    let input = arguments
        .get(1)
        .map(String::as_str)
        .unwrap_or("ceres-solver/data/problem-16-22106-pre.txt");
    let solver_name = arguments
        .get(2)
        .map(String::as_str)
        .unwrap_or("iterative_schur");
    let iterations = arguments
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(5);
    let preconditioner_name = arguments.get(4).map(String::as_str).unwrap_or("jacobi");
    let trust_region_name = arguments.get(5).map(String::as_str).unwrap_or("lm");
    let linear_solver_type = parse_solver(solver_name)?;
    let preconditioner_type = parse_preconditioner(preconditioner_name)?;
    let trust_region_strategy_type = parse_trust_region_strategy(trust_region_name)?;

    let load_start = Instant::now();
    let dataset = read_bal(input)?;
    let load_time = load_start.elapsed();
    let num_cameras = dataset.cameras.len();
    let num_points = dataset.points.len();
    let num_observations = dataset.observations.len();

    let build_start = Instant::now();
    let (problem, initial_values, ordering) = dataset.into_problem();
    let build_time = build_start.elapsed();
    let options = OptimizerOptions {
        max_iteration: iterations,
        linear_solver_type,
        parameter_block_ordering: Some(ordering),
        preconditioner_type,
        trust_region_strategy_type,
        eta: 1e-2,
        ..OptimizerOptions::default()
    };

    let solve_start = Instant::now();
    let result = TrustRegionOptimizer::default().optimize_with_summary(
        &problem,
        &initial_values,
        Some(options),
    );
    let solve_time = solve_start.elapsed();

    println!("input={input}");
    println!("cameras={num_cameras}");
    println!("points={num_points}");
    println!("observations={num_observations}");
    println!("solver={solver_name}");
    println!("preconditioner={preconditioner_name}");
    println!("trust_region_strategy={trust_region_name}");
    println!("iterations_requested={iterations}");
    println!("iterations_completed={}", result.summary.iterations.len());
    println!("load_seconds={:.6}", load_time.as_secs_f64());
    println!("build_seconds={:.6}", build_time.as_secs_f64());
    println!("solve_seconds={:.6}", solve_time.as_secs_f64());
    println!("initial_cost={:.12e}", 0.5 * result.summary.initial_cost);
    println!("final_cost={:.12e}", 0.5 * result.summary.final_cost);
    println!("termination={:?}", result.summary.termination_type);
    println!("message={}", result.summary.message);

    if result.parameters.is_none() {
        return Err(IoError::other("BAL solve failed").into());
    }
    Ok(())
}
