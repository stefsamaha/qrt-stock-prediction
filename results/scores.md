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

