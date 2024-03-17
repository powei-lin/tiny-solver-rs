use std::ops::Mul;

use faer::prelude::SpSolver;
use faer::sparse::linalg::solvers;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub enum LinearSolver {
    SparseCholesky,
    SparseQR,
}
impl Default for LinearSolver {
    fn default() -> Self {
        LinearSolver::SparseCholesky
    }
}

pub fn sparse_cholesky(
    residuals: &faer::Mat<f64>,
    jacobians: &faer::sparse::SparseColMat<usize, f64>,
    symbolic_pattern: &mut Option<solvers::SymbolicCholesky<usize>>,
) -> faer::Mat<f64> {
    let hessian = jacobians
        .as_ref()
        .transpose()
        .to_col_major()
        .unwrap()
        .mul(jacobians.as_ref());
    let b = jacobians.as_ref().transpose().mul(-residuals);
    let sym = if symbolic_pattern.is_some() {
        symbolic_pattern.as_ref().unwrap()
    } else {
        // initialize the pattern
        *symbolic_pattern = Some(
            solvers::SymbolicCholesky::try_new(hessian.symbolic(), faer::Side::Lower).unwrap(),
        );
        symbolic_pattern.as_ref().unwrap()
    };
    let dx =
        solvers::Cholesky::try_new_with_symbolic(sym.clone(), hessian.as_ref(), faer::Side::Lower)
            .unwrap()
            .solve(b);
    dx
}

pub fn sparse_qr(
    residuals: &faer::Mat<f64>,
    jacobians: &faer::sparse::SparseColMat<usize, f64>,
) -> faer::Mat<f64> {
    let dx = jacobians.sp_qr().unwrap().solve(-residuals);
    dx
}
