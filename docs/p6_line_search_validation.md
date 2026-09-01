# P6 Line Search 與 Inner Iterations 驗收

日期：2026-08-31

## 已實作功能

### Line Search

- `LineSearchOptimizer`
- Armijo sufficient-decrease search，使用受限二次回縮
- Strong Wolfe search，包含 bracketing 與 zoom
- Steepest Descent
- Nonlinear Conjugate Gradient
  - Fletcher-Reeves
  - Polak-Ribiere
- L-BFGS two-loop recursion，預設 rank 20
- 非下降方向自動重啟為 steepest descent
- L-BFGS secant condition 防護

### Inner Iterations

- `CoordinateDescentMinimizer`
- 使用 `ParameterBlockOrdering` 指定 ordered independent sets
- 每個 parameter block 執行 regularized local Gauss-Newton
- local step 只有在降低相關 residual cost 時才提交
- inner relative progress 低於 tolerance 後自動停用
- `SolverSummary` 回報 inner iteration 次數與累計時間

## 使用方式

```rust
let options = OptimizerOptions {
    line_search_direction_type: LineSearchDirectionType::Lbfgs,
    line_search_type: LineSearchType::Wolfe,
    max_lbfgs_rank: 20,
    ..OptimizerOptions::default()
};

let result = LineSearchOptimizer::new()
    .optimize_with_summary(&problem, &initial_values, Some(options));
```

Inner iterations 透過 `OptimizerOptions::inner_iteration_ordering` 啟用，僅支援 trust-region minimization。每個 group 必須是 independent set，也就是同一 residual block 不得同時引用 group 內兩個 parameter blocks。

## Ceres 相容限制

- L-BFGS 必須搭配 Wolfe line search。
- Line-search minimization 不支援 parameter bounds。
- Inner iterations 與 evaluation callback 不可同時使用。
- 尚未實作 automatic inner-iteration ordering；目前需由使用者提供 ordering。
- Dense BFGS 與 Hestenes-Stiefel NCG 留作延伸項目。

## 驗證

- Armijo：驗證 sufficient-decrease condition 與步長回縮。
- Strong Wolfe：驗證 Armijo 與 strong-curvature conditions。
- Rosenbrock valley：Steepest Descent、Fletcher-Reeves、Polak-Ribiere、L-BFGS 均穩定下降或收斂至 `(1, 1)`。
- Powell singular function：L-BFGS 與 Levenberg-Marquardt 均收斂至同一零解。
- Inner iterations：驗證單次小 trust-region step 後可進一步降低 cost。
- Invalid ordering：同組參數共同出現在一個 residual 時會明確拒絕。
