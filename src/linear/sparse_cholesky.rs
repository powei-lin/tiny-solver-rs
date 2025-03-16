use std::fmt::Debug;
use std::ops::Mul;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers;

use super::sparse::SparseLinearSolver;

// #[pyclass]
#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    symbolic_pattern: Option<solvers::SymbolicLlt<usize>>,
}

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver {
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
        // initialize the pattern
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern =
                Some(solvers::SymbolicLlt::try_new(jtj.symbolic(), faer::Side::Lower).unwrap());
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
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
