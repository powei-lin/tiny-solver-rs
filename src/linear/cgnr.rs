use std::ops::Mul;

use nalgebra as na;

use super::conjugate_gradient::{self, ConjugateGradientOptions};
use super::{PreconditionerType, SparseLinearSolver};

#[derive(Clone, Copy, Debug)]
pub struct CgnrOptions {
    pub preconditioner_type: PreconditionerType,
    pub min_num_iterations: usize,
    pub max_num_iterations: usize,
    pub residual_reset_period: usize,
    pub residual_tolerance: f64,
    pub q_tolerance: f64,
}

impl Default for CgnrOptions {
    fn default() -> Self {
        Self {
            preconditioner_type: PreconditionerType::Jacobi,
            min_num_iterations: 0,
            max_num_iterations: 500,
            residual_reset_period: 10,
            residual_tolerance: -1.0,
            q_tolerance: 1e-1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CgnrSolver {
    parameter_block_sizes: Vec<usize>,
    options: CgnrOptions,
}

impl CgnrSolver {
    pub fn new(parameter_block_sizes: Vec<usize>, options: CgnrOptions) -> Self {
        Self {
            parameter_block_sizes,
            options,
        }
    }

    fn cg_options(&self) -> ConjugateGradientOptions {
        ConjugateGradientOptions {
            min_num_iterations: self.options.min_num_iterations,
            max_num_iterations: self.options.max_num_iterations,
            residual_reset_period: self.options.residual_reset_period,
            residual_tolerance: self.options.residual_tolerance,
            q_tolerance: self.options.q_tolerance,
        }
    }

    fn solve_columns(
        &self,
        rhs: &faer::Mat<f64>,
        preconditioner: &BlockPreconditioner,
        right_multiply: impl Fn(&na::DVector<f64>) -> Option<na::DVector<f64>>,
    ) -> Option<faer::Mat<f64>> {
        let mut solution = na::DMatrix::zeros(rhs.nrows(), rhs.ncols());
        for col in 0..rhs.ncols() {
            let rhs_column = na::DVector::from_fn(rhs.nrows(), |row, _| rhs[(row, col)]);
            let column = conjugate_gradient::solve(
                &rhs_column,
                self.cg_options(),
                &right_multiply,
                |residual| preconditioner.apply(residual),
            )?;
            solution.column_mut(col).copy_from(&column);
        }
        Some(faer::Mat::from_fn(
            solution.nrows(),
            solution.ncols(),
            |row, col| solution[(row, col)],
        ))
    }

    fn jacobian_normal_product(
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
        vector: &na::DVector<f64>,
    ) -> Option<na::DVector<f64>> {
        if vector.len() != jacobian.ncols() {
            return None;
        }
        let matrix = jacobian.as_ref();
        let mut intermediate = na::DVector::<f64>::zeros(jacobian.nrows());
        for col in 0..jacobian.ncols() {
            let rows = matrix.symbolic().row_idx_of_col_raw(col);
            for (&row, &value) in rows.iter().zip(matrix.val_of_col(col)) {
                intermediate[row] += value * vector[col];
            }
        }
        let mut result = na::DVector::<f64>::zeros(jacobian.ncols());
        for col in 0..jacobian.ncols() {
            let rows = matrix.symbolic().row_idx_of_col_raw(col);
            for (&row, &value) in rows.iter().zip(matrix.val_of_col(col)) {
                result[col] += value * intermediate[row];
            }
        }
        Some(result)
    }

    fn sparse_symmetric_product(
        matrix: &faer::sparse::SparseColMat<usize, f64>,
        vector: &na::DVector<f64>,
    ) -> Option<na::DVector<f64>> {
        if matrix.nrows() != matrix.ncols() || vector.len() != matrix.ncols() {
            return None;
        }
        let matrix_ref = matrix.as_ref();
        let mut result = na::DVector::zeros(matrix.nrows());
        for col in 0..matrix.ncols() {
            let rows = matrix_ref.symbolic().row_idx_of_col_raw(col);
            for (&row, &value) in rows.iter().zip(matrix_ref.val_of_col(col)) {
                result[row] += value * vector[col];
                if row != col && matrix_ref.get(col, row).is_none() {
                    result[col] += value * vector[row];
                }
            }
        }
        Some(result)
    }
}

enum BlockPreconditioner {
    Identity,
    BlockDiagonal {
        starts: Vec<usize>,
        inverses: Vec<na::DMatrix<f64>>,
    },
}

impl BlockPreconditioner {
    fn validate_blocks(block_sizes: &[usize], dimension: usize) -> Option<()> {
        (!block_sizes.is_empty()
            && !block_sizes.contains(&0)
            && block_sizes.iter().sum::<usize>() == dimension)
            .then_some(())
    }

    fn from_jacobian(
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
        block_sizes: &[usize],
        preconditioner_type: PreconditionerType,
    ) -> Option<Self> {
        Self::validate_blocks(block_sizes, jacobian.ncols())?;
        match preconditioner_type {
            PreconditionerType::Identity => Some(Self::Identity),
            PreconditionerType::Jacobi => {
                let matrix = jacobian.as_ref();
                Self::from_blocks(block_sizes, |start, size| {
                    na::DMatrix::from_fn(size, size, |row, col| {
                        let lhs_col = start + row;
                        let rhs_col = start + col;
                        let lhs_rows = matrix.symbolic().row_idx_of_col_raw(lhs_col);
                        let rhs_rows = matrix.symbolic().row_idx_of_col_raw(rhs_col);
                        let lhs_values = matrix.val_of_col(lhs_col);
                        let rhs_values = matrix.val_of_col(rhs_col);
                        let mut lhs_index = 0;
                        let mut rhs_index = 0;
                        let mut dot = 0.0;
                        while lhs_index < lhs_rows.len() && rhs_index < rhs_rows.len() {
                            match lhs_rows[lhs_index].cmp(&rhs_rows[rhs_index]) {
                                std::cmp::Ordering::Less => lhs_index += 1,
                                std::cmp::Ordering::Greater => rhs_index += 1,
                                std::cmp::Ordering::Equal => {
                                    dot += lhs_values[lhs_index] * rhs_values[rhs_index];
                                    lhs_index += 1;
                                    rhs_index += 1;
                                }
                            }
                        }
                        dot
                    })
                })
            }
            PreconditionerType::SchurJacobi => None,
        }
    }

    fn from_normal(
        normal: &faer::sparse::SparseColMat<usize, f64>,
        block_sizes: &[usize],
        preconditioner_type: PreconditionerType,
    ) -> Option<Self> {
        Self::validate_blocks(block_sizes, normal.ncols())?;
        match preconditioner_type {
            PreconditionerType::Identity => Some(Self::Identity),
            PreconditionerType::Jacobi => Self::from_blocks(block_sizes, |start, size| {
                na::DMatrix::from_fn(size, size, |row, col| {
                    normal
                        .as_ref()
                        .get(start + row, start + col)
                        .or_else(|| normal.as_ref().get(start + col, start + row))
                        .copied()
                        .unwrap_or(0.0)
                })
            }),
            PreconditionerType::SchurJacobi => None,
        }
    }

    fn from_blocks(
        block_sizes: &[usize],
        mut block: impl FnMut(usize, usize) -> na::DMatrix<f64>,
    ) -> Option<Self> {
        let mut starts = Vec::with_capacity(block_sizes.len());
        let mut inverses = Vec::with_capacity(block_sizes.len());
        let mut start = 0;
        for &size in block_sizes {
            starts.push(start);
            inverses.push(block(start, size).cholesky()?.inverse());
            start += size;
        }
        Some(Self::BlockDiagonal { starts, inverses })
    }

    fn apply(&self, residual: &na::DVector<f64>) -> na::DVector<f64> {
        match self {
            Self::Identity => residual.clone(),
            Self::BlockDiagonal { starts, inverses } => {
                let mut result = na::DVector::zeros(residual.len());
                for (&start, inverse) in starts.iter().zip(inverses) {
                    let transformed = inverse * residual.rows(start, inverse.nrows());
                    result
                        .rows_mut(start, inverse.nrows())
                        .copy_from(&transformed);
                }
                result
            }
        }
    }
}

impl SparseLinearSolver for CgnrSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let rhs = jacobians.as_ref().transpose().mul(-residuals);
        let preconditioner = BlockPreconditioner::from_jacobian(
            jacobians,
            &self.parameter_block_sizes,
            self.options.preconditioner_type,
        )?;
        self.solve_columns(&rhs, &preconditioner, |vector| {
            Self::jacobian_normal_product(jacobians, vector)
        })
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let preconditioner = BlockPreconditioner::from_normal(
            jtj,
            &self.parameter_block_sizes,
            self.options.preconditioner_type,
        )?;
        self.solve_columns(jtr, &preconditioner, |vector| {
            Self::sparse_symmetric_product(jtj, vector)
        })
    }
}

#[cfg(test)]
mod tests {
    use faer::{mat, sparse::Triplet};

    use super::*;

    #[test]
    fn identity_and_jacobi_solve_jacobian_and_normal_forms() {
        let jacobian = faer::sparse::SparseColMat::try_new_from_triplets(
            3,
            2,
            &[
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 0, 1.0),
                Triplet::new(2, 1, 1.0),
            ],
        )
        .unwrap();
        let residuals = mat![[-1.0], [-2.0], [-3.0]];
        let normal = faer::sparse::SparseColMat::try_new_from_triplets(
            2,
            2,
            &[
                Triplet::new(0, 0, 2.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 2.0),
            ],
        )
        .unwrap();
        let rhs = mat![[4.0], [5.0]];

        for preconditioner_type in [PreconditionerType::Identity, PreconditionerType::Jacobi] {
            let options = CgnrOptions {
                preconditioner_type,
                max_num_iterations: 20,
                residual_tolerance: 1e-12,
                q_tolerance: 0.0,
                ..CgnrOptions::default()
            };
            let mut solver = CgnrSolver::new(vec![1, 1], options);
            for solution in [
                solver.solve(&residuals, &jacobian).unwrap(),
                solver.solve_jtj(&rhs, &normal).unwrap(),
            ] {
                assert!((solution[(0, 0)] - 1.0).abs() < 1e-9);
                assert!((solution[(1, 0)] - 2.0).abs() < 1e-9);
            }
        }
    }
}
