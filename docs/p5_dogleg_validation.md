# P5 Dogleg 信賴域策略驗收

日期：2026-08-31

## API

`OptimizerOptions::trust_region_strategy_type` 可選：

- `TrustRegionStrategyType::LevenbergMarquardt`
- `TrustRegionStrategyType::Dogleg(DoglegType::Traditional)`
- `TrustRegionStrategyType::Dogleg(DoglegType::Subspace)`

`TrustRegionOptimizer` 是既有 `LevenbergMarquardtOptimizer` 的中性別名；舊 API 保持可用。

```rust
let options = OptimizerOptions {
    trust_region_strategy_type:
        TrustRegionStrategyType::Dogleg(DoglegType::Traditional),
    ..OptimizerOptions::default()
};
let result = TrustRegionOptimizer::default()
    .optimize_with_summary(&problem, &initial_values, Some(options));
```

Dogleg 與 Ceres 一樣只接受精確 factorization linear solvers；CGNR 與 Iterative Schur 會在求解前回傳設定錯誤。

## 實作範圍

- 共用 `TrustRegionStrategy` contract
- Ceres-style Levenberg-Marquardt diagonal 與 radius 更新
- Traditional Dogleg：Cauchy 點、Gauss-Newton 點與邊界線段交點
- Subspace Dogleg：gradient/Gauss-Newton 正交基底上的 2D trust-region 最小化
- 橢圓 trust region scaling
- Gauss-Newton rank-deficiency regularization（初始 `mu=1e-8`）
- Ceres 0.25/0.75 step-quality radius 更新規則

## 驗證

單元與整合測試涵蓋：

- Traditional/Subspace 步長不超過 trust-region radius
- 寬鬆 radius 時回復 Gauss-Newton 解
- 一維 subspace valley 的正確邊界步長
- Rosenbrock 問題由 `(-1.2, 1)` 收斂至 `(1, 1)`
- iterative linear solver 設定被 Dogleg 拒絕

BAL `problem-16-22106-pre.txt`、Dense Schur、5 iterations、10 threads 的單次診斷結果：

| 實作 | 策略 | 總時間（秒） | Final cost |
|---|---|---:|---:|
| Ceres | Traditional Dogleg | 0.217698 | 1.803390e4 |
| tiny-solver | Traditional Dogleg | 0.735078 | 1.803390352394e4 |
| Ceres | Subspace Dogleg | 0.231580 | 1.803390e4 |
| tiny-solver | Subspace Dogleg | 0.725787 | 1.803390352393e4 |

兩種 Dogleg 的數值結果均與 Ceres 顯示精度一致。
