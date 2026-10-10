#[derive(Default, Clone)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

pub trait SparseLinearSolver {
    /// Solves the least-squares problem `min ||J * dx + r||` for `dx`.
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;
    /// Solves `jtj * dx = jtr` for `dx`, where the caller passes the normal
    /// equations already assembled, i.e. `jtj = J^T * J` and `jtr = J^T * -r`.
    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;
}
