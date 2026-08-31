use std::ops::Mul;

use nalgebra as na;

use super::SparseLinearSolver;
use super::conjugate_gradient::{self, ConjugateGradientOptions};
use super::schur::SchurSystem;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreconditionerType {
    Identity,
    #[default]
    Jacobi,
    SchurJacobi,
}

#[derive(Clone, Copy, Debug)]
pub struct IterativeSchurOptions {
    pub preconditioner_type: PreconditionerType,
    pub min_num_iterations: usize,
    pub max_num_iterations: usize,
    pub residual_reset_period: usize,
    pub residual_tolerance: f64,
    pub q_tolerance: f64,
}

impl Default for IterativeSchurOptions {
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
pub struct IterativeSchurSolver {
    elimination_block_sizes: Vec<usize>,
    retained_block_sizes: Vec<usize>,
    options: IterativeSchurOptions,
}

impl IterativeSchurSolver {
    pub fn new(
        elimination_block_sizes: Vec<usize>,
        retained_block_sizes: Vec<usize>,
        options: IterativeSchurOptions,
    ) -> Self {
        Self {
            elimination_block_sizes,
            retained_block_sizes,
            options,
        }
    }

    fn solve_column(
        &self,
        system: &SchurSystem,
        rhs: &na::DVector<f64>,
        preconditioner: &BlockPreconditioner,
    ) -> Option<na::DVector<f64>> {
        conjugate_gradient::solve(
            rhs,
            ConjugateGradientOptions {
                min_num_iterations: self.options.min_num_iterations,
                max_num_iterations: self.options.max_num_iterations,
                residual_reset_period: self.options.residual_reset_period,
                residual_tolerance: self.options.residual_tolerance,
                q_tolerance: self.options.q_tolerance,
            },
            |vector| system.right_multiply(vector),
            |residual| preconditioner.apply(residual),
        )
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
    fn try_new(
        system: &SchurSystem,
        block_sizes: &[usize],
        preconditioner_type: PreconditionerType,
    ) -> Option<Self> {
        if block_sizes.is_empty()
            || block_sizes.contains(&0)
            || block_sizes.iter().sum::<usize>() != system.retained_dimension()
        {
            return None;
        }
        if preconditioner_type == PreconditionerType::Identity {
            return Some(Self::Identity);
        }

        let mut starts = Vec::with_capacity(block_sizes.len());
        let mut inverses = Vec::with_capacity(block_sizes.len());
        let mut start = 0;
        for &size in block_sizes {
            starts.push(start);
            let diagonal = system.diagonal_block(
                start,
                size,
                preconditioner_type == PreconditionerType::SchurJacobi,
            )?;
            inverses.push(diagonal.cholesky()?.inverse());
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

impl SparseLinearSolver for IterativeSchurSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let jtj = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .ok()?
            .mul(jacobians.as_ref());
        let jtr = jacobians.as_ref().transpose().mul(-residuals);
        self.solve_jtj(&jtr, &jtj)
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let system = SchurSystem::try_new(&self.elimination_block_sizes, jtr, jtj)?;
        let preconditioner = BlockPreconditioner::try_new(
            &system,
            &self.retained_block_sizes,
            self.options.preconditioner_type,
        )?;
        let mut retained_solution =
            na::DMatrix::zeros(system.retained_dimension(), system.reduced_rhs().ncols());
        for col in 0..retained_solution.ncols() {
            let solution = self.solve_column(
                &system,
                &system.reduced_rhs().column(col).into_owned(),
                &preconditioner,
            )?;
            retained_solution.column_mut(col).copy_from(&solution);
        }
        system.back_substitute(&retained_solution)
    }
}

#[cfg(test)]
mod tests {
    use faer::{mat, sparse::Triplet};

    use super::*;

    #[test]
    fn all_preconditioners_solve_the_implicit_schur_system() {
        let normal = faer::sparse::SparseColMat::try_new_from_triplets(
            4,
            4,
            &[
                Triplet::new(0, 0, 4.0),
                Triplet::new(1, 1, 5.0),
                Triplet::new(0, 2, 1.0),
                Triplet::new(2, 0, 1.0),
                Triplet::new(0, 3, 2.0),
                Triplet::new(3, 0, 2.0),
                Triplet::new(1, 2, 0.5),
                Triplet::new(2, 1, 0.5),
                Triplet::new(1, 3, 1.0),
                Triplet::new(3, 1, 1.0),
                Triplet::new(2, 2, 6.0),
                Triplet::new(2, 3, 1.0),
                Triplet::new(3, 2, 1.0),
                Triplet::new(3, 3, 7.0),
            ],
        )
        .unwrap();
        let rhs = mat![[15.0], [15.5], [24.0], [35.0]];

        for preconditioner_type in [
            PreconditionerType::Identity,
            PreconditionerType::Jacobi,
            PreconditionerType::SchurJacobi,
        ] {
            let options = IterativeSchurOptions {
                preconditioner_type,
                max_num_iterations: 20,
                residual_tolerance: 1e-12,
                q_tolerance: 0.0,
                ..IterativeSchurOptions::default()
            };
            let solution = IterativeSchurSolver::new(vec![1, 1], vec![1, 1], options)
                .solve_jtj(&rhs, &normal)
                .unwrap();

            for (row, expected) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
                assert!((solution[(row, 0)] - expected).abs() < 1e-9);
            }
        }
    }
}
