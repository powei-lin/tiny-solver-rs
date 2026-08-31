use std::collections::BTreeMap;

use faer::sparse::Triplet;
use nalgebra as na;

use super::{SparseCholeskySolver, SparseLinearSolver};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchurComplementMode {
    Dense,
    Sparse,
}

#[derive(Clone, Debug)]
pub struct SchurComplementSolver {
    elimination_block_sizes: Vec<usize>,
    mode: SchurComplementMode,
    sparse_reduced_solver: SparseCholeskySolver,
    jacobian_row_layout: Option<JacobianRowLayout>,
}

impl SchurComplementSolver {
    pub fn new(elimination_dimension: usize, mode: SchurComplementMode) -> Self {
        Self::new_with_block_sizes(vec![elimination_dimension], mode)
    }

    pub fn new_with_block_sizes(
        elimination_block_sizes: Vec<usize>,
        mode: SchurComplementMode,
    ) -> Self {
        Self {
            elimination_block_sizes,
            mode,
            sparse_reduced_solver: SparseCholeskySolver::new(),
            jacobian_row_layout: None,
        }
    }

    fn solve_sparse_reduced(
        &mut self,
        dimension: usize,
        values: BTreeMap<(usize, usize), f64>,
        rhs: &na::DMatrix<f64>,
    ) -> Option<na::DMatrix<f64>> {
        let triplets: Vec<_> = values
            .into_iter()
            .filter_map(|((row, col), value)| {
                (value != 0.0).then_some(Triplet::new(row, col, value))
            })
            .collect();
        let sparse_lhs =
            faer::sparse::SparseColMat::try_new_from_triplets(dimension, dimension, &triplets)
                .ok()?;
        let faer_rhs = faer::Mat::from_fn(rhs.nrows(), rhs.ncols(), |row, col| rhs[(row, col)]);
        let solution = self
            .sparse_reduced_solver
            .solve_jtj(&faer_rhs, &sparse_lhs)?;
        Some(na::DMatrix::from_fn(
            solution.nrows(),
            solution.ncols(),
            |row, col| solution[(row, col)],
        ))
    }

    fn solve_system(&mut self, system: SchurSystem) -> Option<faer::Mat<f64>> {
        let retained_solution = match system.explicit_reduced(self.mode) {
            ReducedSystem::Dense(matrix) => matrix.cholesky()?.solve(system.reduced_rhs()),
            ReducedSystem::Sparse(values) => self.solve_sparse_reduced(
                system.retained_dimension(),
                values,
                system.reduced_rhs(),
            )?,
        };
        system.back_substitute(&retained_solution)
    }

    fn update_jacobian_row_layout(&mut self, jacobian: &faer::sparse::SparseColMat<usize, f64>) {
        if self
            .jacobian_row_layout
            .as_ref()
            .is_none_or(|layout| !layout.matches(jacobian))
        {
            self.jacobian_row_layout = Some(JacobianRowLayout::new(jacobian));
        }
    }
}

struct EliminatedBlock {
    start: usize,
    retained_columns: Vec<usize>,
    cross: na::DMatrix<f64>,
    inverse_times_cross: na::DMatrix<f64>,
    inverse_times_rhs: na::DMatrix<f64>,
}

struct EliminatedBlockAssembly {
    start: usize,
    diagonal: na::DMatrix<f64>,
    cross_columns: BTreeMap<usize, Vec<f64>>,
    dense_cross: Option<na::DMatrix<f64>>,
    rhs: na::DMatrix<f64>,
}

#[derive(Clone, Debug)]
struct JacobianRowLayout {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    rows: Vec<Vec<(usize, usize)>>,
}

impl JacobianRowLayout {
    fn new(jacobian: &faer::sparse::SparseColMat<usize, f64>) -> Self {
        let symbolic = jacobian.symbolic();
        let mut rows = vec![Vec::new(); jacobian.nrows()];
        for column in 0..jacobian.ncols() {
            for value_index in symbolic.col_range(column) {
                rows[symbolic.row_idx()[value_index]].push((column, value_index));
            }
        }
        Self {
            nrows: jacobian.nrows(),
            ncols: jacobian.ncols(),
            col_ptr: symbolic.col_ptr().to_vec(),
            row_idx: symbolic.row_idx().to_vec(),
            rows,
        }
    }

    fn matches(&self, jacobian: &faer::sparse::SparseColMat<usize, f64>) -> bool {
        let symbolic = jacobian.symbolic();
        self.nrows == jacobian.nrows()
            && self.ncols == jacobian.ncols()
            && self.col_ptr == symbolic.col_ptr()
            && self.row_idx == symbolic.row_idx()
    }
}

pub(crate) struct SchurSystem {
    eliminated_dimension: usize,
    retained_dimension: usize,
    retained_values: BTreeMap<(usize, usize), f64>,
    reduced_rhs: na::DMatrix<f64>,
    eliminated_blocks: Vec<EliminatedBlock>,
}

enum ReducedSystem {
    Dense(na::DMatrix<f64>),
    Sparse(BTreeMap<(usize, usize), f64>),
}

impl ReducedSystem {
    fn subtract_block(&mut self, columns: &[usize], block: &na::DMatrix<f64>) {
        for (local_col, &global_col) in columns.iter().enumerate() {
            for (local_row, &global_row) in columns.iter().enumerate() {
                let value = block[(local_row, local_col)];
                match self {
                    Self::Dense(matrix) => matrix[(global_row, global_col)] -= value,
                    Self::Sparse(values) => {
                        *values.entry((global_row, global_col)).or_default() -= value;
                    }
                }
            }
        }
    }
}

impl SchurSystem {
    pub(crate) fn try_new(
        elimination_block_sizes: &[usize],
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<Self> {
        let dimension = jtj.nrows();
        let eliminated: usize = elimination_block_sizes.iter().sum();
        if jtj.ncols() != dimension
            || jtr.nrows() != dimension
            || elimination_block_sizes.is_empty()
            || elimination_block_sizes.contains(&0)
            || eliminated >= dimension
        {
            return None;
        }

        let retained = dimension - eliminated;
        let rhs = na::DMatrix::from_fn(jtr.nrows(), jtr.ncols(), |row, col| jtr[(row, col)]);
        let mut row_to_block = vec![0; eliminated];
        let mut block_starts = Vec::with_capacity(elimination_block_sizes.len());
        let mut start = 0;
        for (block, &size) in elimination_block_sizes.iter().enumerate() {
            block_starts.push(start);
            row_to_block[start..start + size].fill(block);
            start += size;
        }

        let mut eliminated_assemblies: Vec<_> = elimination_block_sizes
            .iter()
            .enumerate()
            .map(|(block, &size)| EliminatedBlockAssembly {
                start: block_starts[block],
                diagonal: na::DMatrix::zeros(size, size),
                cross_columns: BTreeMap::new(),
                dense_cross: None,
                rhs: rhs.rows(block_starts[block], size).into_owned(),
            })
            .collect();
        let mut retained_values = BTreeMap::new();
        let normal = jtj.as_ref();
        for col in 0..dimension {
            let rows = normal.symbolic().row_idx_of_col_raw(col);
            for (&row, &value) in rows.iter().zip(normal.val_of_col(col)) {
                if row < eliminated && col < eliminated {
                    let row_block = row_to_block[row];
                    let col_block = row_to_block[col];
                    if row_block != col_block && value != 0.0 {
                        return None;
                    }
                    if row_block == col_block {
                        let local_row = row - block_starts[row_block];
                        let local_col = col - block_starts[row_block];
                        eliminated_assemblies[row_block].diagonal[(local_row, local_col)] = value;
                        eliminated_assemblies[row_block].diagonal[(local_col, local_row)] = value;
                    }
                } else if row < eliminated {
                    let block = row_to_block[row];
                    let local_row = row - block_starts[block];
                    eliminated_assemblies[block]
                        .cross_columns
                        .entry(col - eliminated)
                        .or_insert_with(|| vec![0.0; elimination_block_sizes[block]])[local_row] =
                        value;
                } else if col < eliminated {
                    let block = row_to_block[col];
                    let local_row = col - block_starts[block];
                    eliminated_assemblies[block]
                        .cross_columns
                        .entry(row - eliminated)
                        .or_insert_with(|| vec![0.0; elimination_block_sizes[block]])[local_row] =
                        value;
                } else {
                    retained_values.insert((row - eliminated, col - eliminated), value);
                    retained_values
                        .entry((col - eliminated, row - eliminated))
                        .or_insert(value);
                }
            }
        }

        Self::finish(
            eliminated,
            retained,
            retained_values,
            rhs.rows(eliminated, retained).into_owned(),
            eliminated_assemblies,
        )
    }

    fn try_new_from_jacobian(
        elimination_block_sizes: &[usize],
        residuals: &faer::Mat<f64>,
        jacobian: &faer::sparse::SparseColMat<usize, f64>,
        regularization: &[f64],
        dense_retained: bool,
        row_layout: &JacobianRowLayout,
    ) -> Option<Self> {
        let dimension = jacobian.ncols();
        let eliminated: usize = elimination_block_sizes.iter().sum();
        if residuals.nrows() != jacobian.nrows()
            || regularization.len() != dimension
            || regularization
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || elimination_block_sizes.is_empty()
            || elimination_block_sizes.contains(&0)
            || eliminated >= dimension
        {
            return None;
        }

        let retained = dimension - eliminated;
        let mut row_to_block = vec![0; eliminated];
        let mut block_starts = Vec::with_capacity(elimination_block_sizes.len());
        let mut start = 0;
        for (block, &size) in elimination_block_sizes.iter().enumerate() {
            block_starts.push(start);
            row_to_block[start..start + size].fill(block);
            start += size;
        }
        let mut eliminated_assemblies: Vec<_> = elimination_block_sizes
            .iter()
            .enumerate()
            .map(|(block, &size)| EliminatedBlockAssembly {
                start: block_starts[block],
                diagonal: na::DMatrix::zeros(size, size),
                cross_columns: BTreeMap::new(),
                dense_cross: dense_retained.then(|| na::DMatrix::zeros(size, retained)),
                rhs: na::DMatrix::zeros(size, residuals.ncols()),
            })
            .collect();
        let mut retained_values = BTreeMap::new();
        let mut dense_retained_values =
            dense_retained.then(|| na::DMatrix::<f64>::zeros(retained, retained));
        let mut reduced_rhs = na::DMatrix::zeros(retained, residuals.ncols());
        if !row_layout.matches(jacobian) {
            return None;
        }
        let values = jacobian.as_ref().val();
        let mut eliminated_entries = Vec::new();
        let mut retained_entries = Vec::new();

        for residual_row in 0..jacobian.nrows() {
            eliminated_entries.clear();
            retained_entries.clear();
            let mut eliminated_block = None;
            for &(column, value_index) in &row_layout.rows[residual_row] {
                let value = values[value_index];
                if value == 0.0 {
                    continue;
                }
                if column < eliminated {
                    let block = row_to_block[column];
                    if eliminated_block.is_some_and(|existing| existing != block) {
                        return None;
                    }
                    eliminated_block = Some(block);
                    let local_column = column - block_starts[block];
                    eliminated_entries.push((local_column, value));
                    for rhs_col in 0..residuals.ncols() {
                        eliminated_assemblies[block].rhs[(local_column, rhs_col)] -=
                            value * residuals[(residual_row, rhs_col)];
                    }
                } else {
                    let retained_column = column - eliminated;
                    retained_entries.push((retained_column, value));
                    for rhs_col in 0..residuals.ncols() {
                        reduced_rhs[(retained_column, rhs_col)] -=
                            value * residuals[(residual_row, rhs_col)];
                    }
                }
            }

            if let Some(block) = eliminated_block {
                let assembly = &mut eliminated_assemblies[block];
                for &(row, row_value) in &eliminated_entries {
                    for &(col, col_value) in &eliminated_entries {
                        assembly.diagonal[(row, col)] += row_value * col_value;
                    }
                    for &(retained_col, retained_value) in &retained_entries {
                        let value = row_value * retained_value;
                        if let Some(cross) = &mut assembly.dense_cross {
                            cross[(row, retained_col)] += value;
                        } else {
                            assembly
                                .cross_columns
                                .entry(retained_col)
                                .or_insert_with(|| vec![0.0; elimination_block_sizes[block]])
                                [row] += value;
                        }
                    }
                }
            }
            for &(row, row_value) in &retained_entries {
                for &(col, col_value) in &retained_entries {
                    let value = row_value * col_value;
                    if let Some(matrix) = &mut dense_retained_values {
                        matrix[(row, col)] += value;
                    } else {
                        *retained_values.entry((row, col)).or_default() += value;
                    }
                }
            }
        }

        for assembly in &mut eliminated_assemblies {
            for local in 0..assembly.diagonal.nrows() {
                assembly.diagonal[(local, local)] += regularization[assembly.start + local];
            }
        }
        for retained_col in 0..retained {
            let value = regularization[eliminated + retained_col];
            if let Some(matrix) = &mut dense_retained_values {
                matrix[(retained_col, retained_col)] += value;
            } else {
                *retained_values
                    .entry((retained_col, retained_col))
                    .or_default() += value;
            }
        }
        if let Some(matrix) = dense_retained_values {
            for col in 0..retained {
                for row in 0..retained {
                    let value = matrix[(row, col)];
                    if value != 0.0 {
                        retained_values.insert((row, col), value);
                    }
                }
            }
        }

        Self::finish(
            eliminated,
            retained,
            retained_values,
            reduced_rhs,
            eliminated_assemblies,
        )
    }

    fn finish(
        eliminated: usize,
        retained: usize,
        retained_values: BTreeMap<(usize, usize), f64>,
        mut reduced_rhs: na::DMatrix<f64>,
        eliminated_assemblies: Vec<EliminatedBlockAssembly>,
    ) -> Option<Self> {
        let mut eliminated_blocks = Vec::with_capacity(eliminated_assemblies.len());
        for assembly in eliminated_assemblies {
            let size = assembly.diagonal.nrows();
            let (columns, cross) = if let Some(dense_cross) = assembly.dense_cross {
                let columns: Vec<_> = (0..dense_cross.ncols())
                    .filter(|&column| dense_cross.column(column).iter().any(|value| *value != 0.0))
                    .collect();
                let cross = na::DMatrix::from_fn(size, columns.len(), |row, col| {
                    dense_cross[(row, columns[col])]
                });
                (columns, cross)
            } else {
                let columns: Vec<_> = assembly.cross_columns.keys().copied().collect();
                let cross = na::DMatrix::from_fn(size, columns.len(), |row, col| {
                    assembly.cross_columns[&columns[col]][row]
                });
                (columns, cross)
            };
            let cholesky = assembly.diagonal.cholesky()?;
            let inverse_times_cross = cholesky.solve(&cross);
            let inverse_times_rhs = cholesky.solve(&assembly.rhs);
            let rhs_update = cross.transpose() * &inverse_times_rhs;
            for (local_row, &global_row) in columns.iter().enumerate() {
                for col in 0..reduced_rhs.ncols() {
                    reduced_rhs[(global_row, col)] -= rhs_update[(local_row, col)];
                }
            }
            eliminated_blocks.push(EliminatedBlock {
                start: assembly.start,
                retained_columns: columns,
                cross,
                inverse_times_cross,
                inverse_times_rhs,
            });
        }

        Some(Self {
            eliminated_dimension: eliminated,
            retained_dimension: retained,
            retained_values,
            reduced_rhs,
            eliminated_blocks,
        })
    }

    fn explicit_reduced(&self, mode: SchurComplementMode) -> ReducedSystem {
        let mut reduced = match mode {
            SchurComplementMode::Dense => {
                let mut matrix =
                    na::DMatrix::zeros(self.retained_dimension, self.retained_dimension);
                for (&(row, col), &value) in &self.retained_values {
                    matrix[(row, col)] = value;
                }
                ReducedSystem::Dense(matrix)
            }
            SchurComplementMode::Sparse => ReducedSystem::Sparse(self.retained_values.clone()),
        };
        for block in &self.eliminated_blocks {
            let update = block.cross.transpose() * &block.inverse_times_cross;
            reduced.subtract_block(&block.retained_columns, &update);
        }
        reduced
    }

    pub(crate) fn retained_dimension(&self) -> usize {
        self.retained_dimension
    }

    pub(crate) fn reduced_rhs(&self) -> &na::DMatrix<f64> {
        &self.reduced_rhs
    }

    pub(crate) fn right_multiply(&self, vector: &na::DVector<f64>) -> Option<na::DVector<f64>> {
        if vector.len() != self.retained_dimension {
            return None;
        }
        let mut result = na::DVector::zeros(self.retained_dimension);
        for (&(row, col), &value) in &self.retained_values {
            result[row] += value * vector[col];
        }
        for block in &self.eliminated_blocks {
            let local_vector = na::DVector::from_iterator(
                block.retained_columns.len(),
                block.retained_columns.iter().map(|&col| vector[col]),
            );
            let transformed = &block.inverse_times_cross * local_vector;
            let update = block.cross.transpose() * transformed;
            for (local_row, &global_row) in block.retained_columns.iter().enumerate() {
                result[global_row] -= update[local_row];
            }
        }
        Some(result)
    }

    pub(crate) fn diagonal_block(
        &self,
        start: usize,
        size: usize,
        include_elimination: bool,
    ) -> Option<na::DMatrix<f64>> {
        if size == 0 || start + size > self.retained_dimension {
            return None;
        }
        let mut diagonal = na::DMatrix::from_fn(size, size, |row, col| {
            self.retained_values
                .get(&(start + row, start + col))
                .copied()
                .unwrap_or(0.0)
        });
        if include_elimination {
            for block in &self.eliminated_blocks {
                let selected: Vec<_> = block
                    .retained_columns
                    .iter()
                    .enumerate()
                    .filter_map(|(local, &global)| {
                        if (start..start + size).contains(&global) {
                            Some((local, global - start))
                        } else {
                            None
                        }
                    })
                    .collect();
                for &(cross_row, diagonal_row) in &selected {
                    for &(cross_col, diagonal_col) in &selected {
                        diagonal[(diagonal_row, diagonal_col)] -= block
                            .cross
                            .column(cross_row)
                            .dot(&block.inverse_times_cross.column(cross_col));
                    }
                }
            }
        }
        Some(diagonal)
    }

    pub(crate) fn back_substitute(
        &self,
        retained_solution: &na::DMatrix<f64>,
    ) -> Option<faer::Mat<f64>> {
        if retained_solution.nrows() != self.retained_dimension
            || retained_solution.ncols() != self.reduced_rhs.ncols()
        {
            return None;
        }
        let mut eliminated_solution =
            na::DMatrix::zeros(self.eliminated_dimension, retained_solution.ncols());
        for block in &self.eliminated_blocks {
            let retained_subset = na::DMatrix::from_fn(
                block.retained_columns.len(),
                retained_solution.ncols(),
                |row, col| retained_solution[(block.retained_columns[row], col)],
            );
            let solution = &block.inverse_times_rhs - &block.inverse_times_cross * retained_subset;
            eliminated_solution
                .rows_mut(block.start, solution.nrows())
                .copy_from(&solution);
        }

        let dimension = self.eliminated_dimension + self.retained_dimension;
        Some(faer::Mat::from_fn(
            dimension,
            retained_solution.ncols(),
            |row, col| {
                if row < self.eliminated_dimension {
                    eliminated_solution[(row, col)]
                } else {
                    retained_solution[(row - self.eliminated_dimension, col)]
                }
            },
        ))
    }
}

impl SparseLinearSolver for SchurComplementSolver {
    fn solve(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        self.update_jacobian_row_layout(jacobians);
        let regularization = vec![0.0; jacobians.ncols()];
        let system = SchurSystem::try_new_from_jacobian(
            &self.elimination_block_sizes,
            residuals,
            jacobians,
            &regularization,
            self.mode == SchurComplementMode::Dense,
            self.jacobian_row_layout.as_ref()?,
        )?;
        self.solve_system(system)
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        self.solve_system(SchurSystem::try_new(
            &self.elimination_block_sizes,
            jtr,
            jtj,
        )?)
    }

    fn solve_regularized(
        &mut self,
        residuals: &faer::Mat<f64>,
        jacobians: &faer::sparse::SparseColMat<usize, f64>,
        regularization: &[f64],
    ) -> Option<faer::Mat<f64>> {
        self.update_jacobian_row_layout(jacobians);
        let system = SchurSystem::try_new_from_jacobian(
            &self.elimination_block_sizes,
            residuals,
            jacobians,
            regularization,
            self.mode == SchurComplementMode::Dense,
            self.jacobian_row_layout.as_ref()?,
        )?;
        self.solve_system(system)
    }
}

#[cfg(test)]
mod tests {
    use faer::{mat, sparse::Triplet};

    use super::*;
    use crate::linear::DenseNormalCholeskySolver;

    #[test]
    fn direct_jacobian_assembly_matches_dense_normal_solution() {
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
        let regularization = [0.5, 0.25];

        for mode in [SchurComplementMode::Dense, SchurComplementMode::Sparse] {
            let mut schur = SchurComplementSolver::new_with_block_sizes(vec![1], mode);
            let solution = schur.solve(&residuals, &jacobian).unwrap();
            assert!((solution[(0, 0)] - 1.0).abs() < 1e-10);
            assert!((solution[(1, 0)] - 2.0).abs() < 1e-10);

            let schur_solution = schur
                .solve_regularized(&residuals, &jacobian, &regularization)
                .unwrap();
            let dense_solution = DenseNormalCholeskySolver::new()
                .solve_regularized(&residuals, &jacobian, &regularization)
                .unwrap();
            assert!((schur_solution - dense_solution).norm_l2() < 1e-10);
        }
    }

    #[test]
    fn dense_and_sparse_schur_recover_the_full_system_solution() {
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

        for mode in [SchurComplementMode::Dense, SchurComplementMode::Sparse] {
            let solution = SchurComplementSolver::new_with_block_sizes(vec![1, 1], mode)
                .solve_jtj(&rhs, &normal)
                .unwrap();
            for (row, expected) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
                assert!((solution[(row, 0)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rejects_coupling_between_elimination_blocks() {
        let normal = faer::sparse::SparseColMat::try_new_from_triplets(
            3,
            3,
            &[
                Triplet::new(0, 0, 2.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 1, 2.0),
                Triplet::new(2, 2, 1.0),
            ],
        )
        .unwrap();
        let rhs = mat![[1.0], [1.0], [1.0]];

        let solution =
            SchurComplementSolver::new_with_block_sizes(vec![1, 1], SchurComplementMode::Dense)
                .solve_jtj(&rhs, &normal);

        assert!(solution.is_none());
    }
}
