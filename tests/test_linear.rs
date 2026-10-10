#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use faer::sparse::{SparseColMat, Triplet};
    use nalgebra as na;
    use tiny_solver::linear::sparse::{LinearSolverType, SparseLinearSolver};
    use tiny_solver::optimizer::OptimizerOptions;
    use tiny_solver::{Optimizer, SparseCholeskySolver, SparseQRSolver};

    #[test]
    fn solve_jtj_agrees_between_cholesky_and_qr() {
        // symmetric positive definite
        let jtj = SparseColMat::<usize, f64>::try_new_from_triplets(
            2,
            2,
            &[
                Triplet::new(0, 0, 4.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 1, 3.0),
            ],
        )
        .unwrap();
        let jtr = faer::mat![[1.0], [2.0]];

        let dx_cholesky = SparseCholeskySolver::new().solve_jtj(&jtr, &jtj).unwrap();
        let dx_qr = SparseQRSolver::new().solve_jtj(&jtr, &jtj).unwrap();

        // jtj * dx = jtr  =>  dx = [1/11, 7/11]
        for (i, expected) in [1.0 / 11.0, 7.0 / 11.0].into_iter().enumerate() {
            assert!((dx_cholesky[(i, 0)] - expected).abs() < 1e-12);
            assert!((dx_qr[(i, 0)] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn levenberg_marquardt_with_sparse_qr() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );
        let initial_values = HashMap::from([("x".to_string(), na::dvector![0.0])]);
        let options = OptimizerOptions {
            linear_solver_type: LinearSolverType::SparseQR,
            ..Default::default()
        };

        let result = tiny_solver::LevenbergMarquardtOptimizer::default()
            .optimize(&problem, &initial_values, Some(options))
            .unwrap();

        assert!(
            (result["x"][0] - 3.0).abs() < 1e-3,
            "x = {}",
            result["x"][0]
        );
    }
}
