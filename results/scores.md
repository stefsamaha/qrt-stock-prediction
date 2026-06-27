# Experiment Log

## Baseline Random Forest (raw features, single split)
- Train/val split 80/20, shuffle=False
- Validation accuracy: 50.06%
- Notes: single split too noisy for reliable measurement

## LightGBM (engineered features, single split)  
- Validation accuracy: 49.88%
- Notes: single split unreliable

## LightGBM (engineered features, TimeSeriesCV 5 folds)
- Fold 1: 51.55%
- Fold 2: 51.32%
- Fold 3: 50.91%
- Fold 4: 51.00%
- Fold 5: 49.77%
- Mean: 50.91% | Std: 0.61%
- Benchmark: 51.31%
- Notes: competitive with benchmark. Fold 5 underperforms — 
  possible non-stationarity in recent data

  ## Submission 1 — LightGBM v1
- Public score: 50.73%
- Method: LightGBM, all engineered features, trained on full training set
- Benchmark: 51.31%
- Gap to benchmark: -0.58%
- Notes: first submission, pipeline validated end to end

## Experiment 2 — Sector Features + Volatility Normalization (CV only)
- Mean CV accuracy: 51.19%
- Std: 0.71%
- Previous mean: 50.91%
- Improvement: +0.28%
- New features: sector_mean_ret_1/2/3/5, ret_vs_sector_1/2/3/5, 
  vol_adj_ret_1/2, vol_adj_momentum, vol_regime
- Notes: consistent improvement across folds — submit

## Submission 2 — LightGBM v2 (sector + vol features)
- Public score: 51.30%
- CV mean: 51.19%
- Benchmark: 51.31%
- Gap to benchmark: -0.01%
- Improvement from v1: +0.57%
- Notes: sector conditional mean and volatility normalization drove the improvement. Matched benchmark on second submission.

## Experiment 3 — Hyperparameter Tuning
- CV mean: 51.04% — worse than v2, not submitted
- Notes: over-regularized, v2 params already well-calibrated

## Experiment 4 — Ensemble (LightGBM + RF + Logistic Regression)
- CV mean: 51.43%
- Std: 0.80%
- Improvement over v2: +0.24%
- Method: soft voting, average predicted probabilities
- Notes: consistent improvement across folds 1-4, fold 5 still weak

## Submission 3 — Ensemble LightGBM + RF + LR
- Public score: 51.32%
- CV mean: 51.43%
- Benchmark: 51.31%
- Status: BENCHMARK BEATEN ✓
- Improvement from v1: +0.59%

## Experiment 5 — Cross-sectional rank + interaction features (v6)
- Added 11 features: rank_ret_1/2/3/5, rank_momentum, rank_volatility,
  reversal_x_volume, conditional_reversal, return_acceleration, vol_acceleration
- Mean CV: 51.42% | Std: 1.05%
- Equal-weight v6 vs weighted (0.35/0.25/0.40): flat, -0.0001 (noise)

## Key research findings (today)
1. Fold 5 consistently underperforms (~49.5%) but on inspection has
   normal volatility (0.0246) and class balance (0.4987). Accounting for
   within-day return correlation inflating effective SE, it is within
   noise — NOT a regime break. Stopped optimizing against it.
2. Ensemble diversity confirmed: pairwise prediction correlations
   0.61–0.77. LR most decorrelated (0.61 w/ LGBM). Ensemble earns its keep.
3. Three independent levers tested — features, hyperparameters, ensemble
   weights — all plateau at ~51.4% CV. Conclusion: at signal ceiling for
   this approach. CV sits ~0.001 above leaderboard (trustworthy).

   