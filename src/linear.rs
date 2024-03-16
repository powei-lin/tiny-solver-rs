use std::ops::Mul;

use faer::prelude::SpSolver;

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
) -> faer::prelude::Mat<f64> {
    let hessian = jacobians
        .as_ref()
        .transpose()
        .to_col_major()
        .unwrap()
        .mul(jacobians.as_ref());
    let b = jacobians.as_ref().transpose().mul(-residuals);
    let dx = hessian.sp_cholesky(faer::Side::Lower).unwrap().solve(b);
    dx
}

pub fn sparse_qr(
    residuals: &faer::Mat<f64>,
    jacobians: &faer::sparse::SparseColMat<usize, f64>,
) -> faer::prelude::Mat<f64> {
    let dx = jacobians.sp_qr().unwrap().solve(-residuals);
    dx
}
