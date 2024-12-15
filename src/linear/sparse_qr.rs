use super::sparse::SparseLinearSolver;
use faer::solvers::SpSolverLstsq;
use faer::sparse::linalg::solvers;

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    symbolic_pattern: Option<solvers::SymbolicQr<usize>>,
}

impl SparseQRSolver {
    pub fn new() -> Self {
        SparseQRSolver {
            symbolic_pattern: None,
        }
    }
}
impl Default for SparseQRSolver {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseLinearSolver for SparseQRSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicQr::try_new(jacobians.symbolic()).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(cholesky) = solvers::Qr::try_new_with_symbolic(sym.clone(), jacobians.as_ref()) {
            let dx = cholesky.solve_lstsq(-residuals);
            Some(dx)
        } else {
            None
        }
    }
}
