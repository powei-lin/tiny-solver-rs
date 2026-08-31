use std::ops::Mul;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSolverType {
    DenseQR,
    DenseNormalCholesky,
    DenseSchur,
    SparseSchur,
    IterativeSchur,
    Cgnr,
    #[default]
    SparseCholesky,
    SparseQR,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SparsePattern {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
}

impl SparsePattern {
    pub(crate) fn new(matrix: &faer::sparse::SparseColMat<usize, f64>) -> Self {
        let symbolic = matrix.symbolic();
        Self {
            nrows: symbolic.nrows(),
            ncols: symbolic.ncols(),
            col_ptr: symbolic.col_ptr().to_vec(),
            row_idx: symbolic.row_idx().to_vec(),
        }
    }
}

pub trait SparseLinearSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;
    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>>;

    fn solve_regularized(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        regularization: &[f64],
    ) -> Option<faer::Mat<f64>> {
        if regularization.len() != jacobians.ncols() {
            return None;
        }
        let mut jtj = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .ok()?
            .mul(jacobians.as_ref());
        for (index, &value) in regularization.iter().enumerate() {
            jtj[(index, index)] += value;
        }
        let jtr = jacobians.as_ref().transpose().mul(-residuals);
        self.solve_jtj(&jtr, &jtj)
    }
}
