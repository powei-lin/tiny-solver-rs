use std::fmt::Debug;
use std::ops::Mul;

use faer::prelude::SpSolver;
use faer::sparse::linalg::solvers;

use super::sparse::SparseLinearSolver;

// #[pyclass]

#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    symbolic_pattern: Option<solvers::SymbolicCholesky<usize>>,
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
        let hessian = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobians.as_ref());
        let b = jacobians.as_ref().transpose().mul(-residuals);

        // initialize the pattern
        if self.symbolic_pattern.is_none() {
            self.symbolic_pattern = Some(
                solvers::SymbolicCholesky::try_new(hessian.symbolic(), faer::Side::Lower).unwrap(),
            );
        }

        let sym = self.symbolic_pattern.as_ref().unwrap();
        if let Ok(cholesky) = solvers::Cholesky::try_new_with_symbolic(
            sym.clone(),
            hessian.as_ref(),
            faer::Side::Lower,
        ) {
            let dx = cholesky.solve(b);

            Some(dx)
        } else {
            None
        }
    }
}
