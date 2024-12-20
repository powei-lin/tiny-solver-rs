#[derive(Default, Clone)]
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
}

pub trait SparseLinearSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;
}