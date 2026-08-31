use std::ops::Mul;

use faer::linalg::solvers::{Solve, SolveLstsq};

use super::sparse::SparseLinearSolver;

#[derive(Debug, Clone, Copy, Default)]
pub struct DenseQRSolver;

impl DenseQRSolver {
    pub fn new() -> Self {
        Self
    }
}

impl SparseLinearSolver for DenseQRSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        Some(
            jacobians
                .as_ref()
                .to_dense()
                .col_piv_qr()
                .solve_lstsq(-residuals),
        )
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        Some(jtj.as_ref().to_dense().col_piv_qr().solve_lstsq(jtr))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DenseNormalCholeskySolver;

impl DenseNormalCholeskySolver {
    pub fn new() -> Self {
        Self
    }
}

impl SparseLinearSolver for DenseNormalCholeskySolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let jtj = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .ok()?
            .mul(jacobians.as_ref());
        let jtr = jacobians.as_ref().transpose().mul(-residuals);
        self.solve_jtj(&jtr, &jtj)
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let dense = jtj.as_ref().to_dense();
        let cholesky = dense.llt(faer::Side::Lower).ok()?;
        Some(cholesky.solve(jtr))
    }
}

#[cfg(test)]
mod tests {
    use faer::{Mat, mat, sparse::Triplet};

    use super::*;
    use crate::linear::{SparseCholeskySolver, SparseQRSolver};

    struct TestProblem {
        jacobian: faer::sparse::SparseColMat<usize, f64>,
        residuals: Mat<f64>,
        jtj: faer::sparse::SparseColMat<usize, f64>,
        jtr: Mat<f64>,
    }

    fn test_problem() -> TestProblem {
        let jacobian = faer::sparse::SparseColMat::try_new_from_triplets(
            3,
            2,
            &[
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 0, 1.0),
                Triplet::new(2, 1, 1.0),
            ],
        )
        .unwrap();
        let residuals = mat![[-1.0], [-2.0], [-3.0]];
        let jtj = faer::sparse::SparseColMat::try_new_from_triplets(
            2,
            2,
            &[
                Triplet::new(0, 0, 2.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 2.0),
            ],
        )
        .unwrap();
        let jtr = mat![[4.0], [5.0]];
        TestProblem {
            jacobian,
            residuals,
            jtj,
            jtr,
        }
    }

    fn assert_solution(solution: Mat<f64>) {
        assert_eq!(solution.nrows(), 2);
        assert_eq!(solution.ncols(), 1);
        assert!((solution[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((solution[(1, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn all_solvers_agree_on_overdetermined_system() {
        let problem = test_problem();
        let mut solvers: Vec<Box<dyn SparseLinearSolver>> = vec![
            Box::new(DenseQRSolver::new()),
            Box::new(DenseNormalCholeskySolver::new()),
            Box::new(SparseQRSolver::new()),
            Box::new(SparseCholeskySolver::new()),
        ];

        for solver in &mut solvers {
            assert_solution(solver.solve(&problem.residuals, &problem.jacobian).unwrap());
            assert_solution(solver.solve_jtj(&problem.jtr, &problem.jtj).unwrap());
        }
    }
}
