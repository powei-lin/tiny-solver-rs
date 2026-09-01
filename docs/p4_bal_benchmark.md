# P4 BAL 效能與精度驗收

日期：2026-08-31

## 測試環境

- 硬體：Apple M4，10 CPU cores
- 作業系統：macOS 26.6.2
- Rust：rustc 1.96.0
- CMake：4.2.1
- Ceres Solver：2.3.0，Eigen 3.5.0、AccelerateSparse、METIS 5.1.0
- 資料集：`ceres-solver/data/problem-16-22106-pre.txt`
- 問題規模：16 cameras、22,106 points、83,718 observations、66,462 parameters、167,436 residuals
- 非線性策略：Levenberg-Marquardt，5 iterations，`eta=1e-2`
- 執行緒：10
- 成本定義：`0.5 * ||r||^2`，與 Ceres report 一致

## 重現命令

建置 tiny-solver benchmark：

```bash
cargo build --release --example bal_benchmark
RAYON_NUM_THREADS=10 target/release/examples/bal_benchmark \
  ceres-solver/data/problem-16-22106-pre.txt dense_schur 5 jacobi
```

建置及執行 Ceres baseline：

```bash
cmake -S ceres-solver -B target/ceres-benchmark-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=ON
cmake --build target/ceres-benchmark-build --target bundle_adjuster -j 10

target/ceres-benchmark-build/bin/bundle_adjuster \
  --input=ceres-solver/data/problem-16-22106-pre.txt \
  --linear_solver=dense_schur \
  --linear_solver_ordering=user \
  --num_iterations=5 \
  --num_threads=10 \
  --eta=1e-2
```

## Dense Schur 對照

各執行三次，時間取中位數。

| 實作 | 總時間（秒） | Final cost | 相對 Ceres 時間 |
|---|---:|---:|---:|
| Ceres Dense Schur | 0.207652 | 1.803391e4 | 1.00x |
| tiny-solver Dense Schur | 0.733068 | 1.803399372413e4 | 3.53x |

Final cost 的相對差約為 `4.6e-6`。數值精度驗收通過；原定 2–3x 效能門檻尚未通過，目前超出 3x 上限約 18%。

## 其他 P4 backend

以下為單次 5-iteration 診斷量測；不作跨語言 median 比較。

| 實作 | 前置條件器 | 時間（秒） | Final cost |
|---|---|---:|---:|
| tiny-solver Sparse Schur | Jacobi | 6.708458 | 1.803399372413e4 |
| tiny-solver Iterative Schur | Schur Jacobi | 1.696650 | 1.895472088494e4 |
| tiny-solver CGNR | Jacobi | 17.483200 | 1.935085275834e4 |
| Ceres Iterative Schur | Schur Jacobi | 0.234553 | 1.834333e4 |

只有 16 個 camera blocks 時，縮減系統很小，Dense Schur 明顯優於 Sparse Schur；這符合 Ceres 的 solver 選擇建議。

## 本輪改善

相同資料與 5 iterations 下，tiny-solver Dense Schur 從 2.997 秒降至 0.733 秒，約加速 4.1x。主要變更：

- 無鎖、固定 row order 的 residual/Jacobian 組裝
- 線性時間 CSC symbolic builder 與 source-to-CSC scatter
- Snavely 封閉式 Jacobian，並以 dynamic autodiff 驗證零與非零 angle-axis
- borrowed parameter fast path，避免每筆 observation 複製 camera/point vectors
- 直接由 Jacobian 組裝 Schur blocks，不建立完整 global normal matrix
- Dense Schur 使用 dense retained/cross accumulators
- 快取 Jacobian row layout
- LM 重用 candidate cost，避免重複 residual evaluation

## 結論與後續

P4 的 solver 功能與 BAL 數值精度已通過；效能驗收仍為開放項。Sampling profile 顯示剩餘成本已分散在 scalar sparse Jacobian 建構、每個 residual 的小型配置、parameter lookup，以及 Schur block finalization，不再是一個可由局部修補消除的熱點。

下一個效能階段應導入 block-sparse evaluator：直接以 residual block 產生 `2x3`、`2x9` blocks 並組裝 Schur 系統，略過 scalar CSC Jacobian。完成前，P4 應標記為「功能完成、效能部分完成」。
