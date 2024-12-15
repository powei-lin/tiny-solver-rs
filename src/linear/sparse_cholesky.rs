use std::ops::Mul;

use faer::prelude::SpSolver;
use faer::sparse::linalg::solvers;

// #[pyclass]
#[derive(Default, Clone)]
pub enum LinearSolver {
    #[default]
    SparseCholesky,
    SparseQR,
}

pub fn sparse_cholesky(
    residuals: &faer::Mat<f64>,
    jacobians: &faer::sparse::SparseColMat<usize, f64>,
    symbolic_pattern: &mut Option<solvers::SymbolicCholesky<usize>>,
) -> Option<faer::Mat<f64>> {
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
    if let Ok(cholesky) =
        solvers::Cholesky::try_new_with_symbolic(sym.clone(), hessian.as_ref(), faer::Side::Lower)
    {
        let dx = cholesky.solve(b);

        Some(dx)
    } else {
        None
    }
}

pub fn sparse_qr(
    residuals: &faer::Mat<f64>,
    jacobians: &faer::sparse::SparseColMat<usize, f64>,
) -> faer::Mat<f64> {
    jacobians.sp_qr().unwrap().solve(-residuals)
}
