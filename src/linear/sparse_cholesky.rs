use std::fmt::Debug;
use std::ops::Mul;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers;

use super::sparse::{SparseLinearSolver, SparsePattern};

// #[pyclass]
#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    pattern: Option<SparsePattern>,
    symbolic_pattern: Option<solvers::SymbolicLlt<usize>>,
}

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver {
            pattern: None,
            symbolic_pattern: None,
        }
    }
}
impl Default for SparseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseLinearSolver for SparseCholeskySolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let jtj = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobians.as_ref());
        let jtr = jacobians.as_ref().transpose().mul(-residuals);

        self.solve_jtj(&jtr, &jtj)
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let pattern = SparsePattern::new(jtj);
        if self.pattern.as_ref() != Some(&pattern) {
            self.symbolic_pattern =
                solvers::SymbolicLlt::try_new(jtj.symbolic(), faer::Side::Lower).ok();
            self.pattern = self.symbolic_pattern.as_ref().map(|_| pattern);
        }

        let sym = self.symbolic_pattern.as_ref()?;
        if let Ok(cholesky) =
            solvers::Llt::try_new_with_symbolic(sym.clone(), jtj.as_ref(), faer::Side::Lower)
        {
            let dx = cholesky.solve(jtr);

            Some(dx)
        } else {
            None
        }
    }
}
