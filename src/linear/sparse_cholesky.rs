use std::fmt::Debug;
use std::ops::Mul;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::Solve;
use faer::perm::Perm;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::colamd; // 1. 引入 colamd
use faer::sparse::linalg::solvers;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};

use super::sparse::SparseLinearSolver;

#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    cache: Option<(solvers::SymbolicLlt<usize>, Perm<usize>)>,
}

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver { cache: None }
    }

    /// 輔助函數：執行矩陣重排 P * A * P^T
    fn permute_jtj(
        &self,
        jtj: &SparseColMat<usize, f64>,
        perm: faer::perm::PermRef<usize>,
    ) -> SparseColMat<usize, f64> {
        let dim = jtj.nrows();
        let nnz = jtj.compute_nnz();
        let mut col_ptrs = vec![0usize; dim + 1];
        let mut row_indices = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];

        let req = faer::sparse::utils::permute_self_adjoint_scratch::<usize>(dim);
        let mut mem = MemBuffer::try_new(req).expect("Mem alloc failed");
        let stack = MemStack::new(&mut mem);

        faer::sparse::utils::permute_self_adjoint_to_unsorted(
            &mut values,
            &mut col_ptrs,
            &mut row_indices,
            jtj.as_ref(),
            perm,
            faer::Side::Lower,
            faer::Side::Lower,
            stack,
        );

        SparseColMat::new(
            unsafe { SymbolicSparseColMat::new_unchecked(dim, dim, col_ptrs, None, row_indices) },
            values,
        )
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
        // ====================================================
        // 修改點：優先在 solve 階段使用 COLAMD (因為這裡有 J)
        // ====================================================
        if self.cache.is_none() {
            let n_cols = jacobians.ncols(); // 變數數量 (dim)
            let nnz = jacobians.compute_nnz();

            // 1. 準備 COLAMD 需要的空間
            let mut perm_fwd = vec![0usize; n_cols];
            let mut perm_inv = vec![0usize; n_cols];

            // 2. 準備 Stack (COLAMD 的 scratch space)
            // 注意：這裡是 colamd::order_scratch
            let req = colamd::order_scratch::<usize>(jacobians.nrows(), n_cols, nnz);
            let mut mem = MemBuffer::try_new(req).expect("COLAMD memory alloc failed");
            let stack = MemStack::new(&mut mem);

            // 3. 執行 COLAMD
            // 直接對 J (jacobians) 進行分析，而不是 J^T J
            colamd::order(
                &mut perm_fwd,
                &mut perm_inv,
                jacobians.symbolic(), // 傳入 J 的結構
                colamd::Control::default(),
                stack,
            )
            .expect("COLAMD ordering failed");

            let perm = Perm::new_checked(
                perm_fwd.into_boxed_slice(),
                perm_inv.into_boxed_slice(),
                n_cols,
            );

            // 4. 為了建立 SymbolicLlt，我們還是需要 "重排後的 J^T J 結構"
            // 這裡我們暫時先建構 J^T J (為了取得結構)，然後進行 permute
            // (注意：這一步其實可以優化，不用真的算數值，但為了程式碼簡潔，先用標準做法)
            let jtj = jacobians
                .as_ref()
                .transpose()
                .to_col_major()
                .unwrap()
                .mul(jacobians.as_ref());

            // 將 J^T J 重排： P * (J^T J) * P^T
            let jtj_permuted = self.permute_jtj(&jtj, perm.as_ref());

            // 5. 建立 Symbolic Analysis
            let symbolic =
                solvers::SymbolicLlt::try_new(jtj_permuted.symbolic(), faer::Side::Lower)
                    .expect("Symbolic analysis failed");

            self.cache = Some((symbolic, perm));
        }

        // 接續原本流程：計算 J^T J 和 -J^T r
        let jtj = jacobians
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobians.as_ref());
        let jtr = jacobians.as_ref().transpose().mul(-residuals);

        // 呼叫 solve_jtj
        // 因為 Cache 已經在上面被 COLAMD 填滿了，solve_jtj 會直接使用該 Cache
        self.solve_jtj(&jtr, &jtj)
    }

    fn solve_jtj(
        &mut self,
        jtr: &faer::Mat<f64>,
        jtj: &faer::sparse::SparseColMat<usize, f64>,
    ) -> Option<faer::Mat<f64>> {
        let dim = jtj.nrows();

        // Fallback 機制：
        // 如果使用者沒有呼叫 `solve`，而是直接呼叫 `solve_jtj` (手動傳 Hessian)，
        // 此時我們沒有 J，無法做 COLAMD，只好退而求其次做 AMD。
        if self.cache.is_none() {
            // ... (原本的 AMD 程式碼保持不變，作為備案) ...
            let nnz = jtj.compute_nnz();
            let mut perm_fwd = vec![0usize; dim];
            let mut perm_inv = vec![0usize; dim];
            let req = amd::order_scratch::<usize>(dim, nnz);
            let mut mem = MemBuffer::try_new(req).expect("AMD memory alloc failed");
            let stack = MemStack::new(&mut mem);

            amd::order(
                &mut perm_fwd,
                &mut perm_inv,
                jtj.symbolic(),
                amd::Control::default(),
                stack,
            )
            .expect("AMD failed");

            let perm = Perm::new_checked(
                perm_fwd.into_boxed_slice(),
                perm_inv.into_boxed_slice(),
                dim,
            );
            let jtj_permuted = self.permute_jtj(jtj, perm.as_ref());
            let symbolic =
                solvers::SymbolicLlt::try_new(jtj_permuted.symbolic(), faer::Side::Lower).unwrap();
            self.cache = Some((symbolic, perm));
        }

        // 以下數值求解邏輯完全不用動
        let (symbolic, perm) = self.cache.as_ref().unwrap();
        let jtj_permuted = self.permute_jtj(jtj, perm.as_ref());
        let jtr_permuted = perm.as_ref() * jtr;

        if let Ok(cholesky) = solvers::Llt::try_new_with_symbolic(
            symbolic.clone(),
            jtj_permuted.as_ref(),
            faer::Side::Lower,
        ) {
            let dx_permuted = cholesky.solve(&jtr_permuted);
            let dx = perm.as_ref().inverse() * dx_permuted;
            Some(dx)
        } else {
            None
        }
    }
}
