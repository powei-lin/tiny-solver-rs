use super::sparse::{SparseLinearSolver, SparsePattern};
use faer::linalg::solvers::SolveLstsq;
// use faer::prelude::{SpSolver, SpSolverLstsq};
use faer::sparse::linalg::solvers;

#[derive(Debug, Clone)]
pub struct SparseQRSolver {
    pattern: Option<SparsePattern>,
    symbolic_pattern: Option<solvers::SymbolicQr<usize>>,
}

impl SparseQRSolver {
    pub fn new() -> Self {
        SparseQRSolver {
            pattern: None,
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
        let pattern = SparsePattern::new(jacobians);
        if self.pattern.as_ref() != Some(&pattern) {
            self.symbolic_pattern = solvers::SymbolicQr::try_new(jacobians.symbolic()).ok();
            self.pattern = self.symbolic_pattern.as_ref().map(|_| pattern);
        }

        let sym = self.symbolic_pattern.as_ref()?;
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), jacobians.as_ref()) {
            Some(qr.solve_lstsq(-residuals))
        } else {
            None
        }
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let pattern = SparsePattern::new(jtj);
        if self.pattern.as_ref() != Some(&pattern) {
            self.symbolic_pattern = solvers::SymbolicQr::try_new(jtj.symbolic()).ok();
            self.pattern = self.symbolic_pattern.as_ref().map(|_| pattern);
        }

        let sym = self.symbolic_pattern.as_ref()?;
        if let Ok(qr) = solvers::Qr::try_new_with_symbolic(sym.clone(), jtj.as_ref()) {
            Some(qr.solve_lstsq(jtr))
        } else {
            None
        }
    }
}
