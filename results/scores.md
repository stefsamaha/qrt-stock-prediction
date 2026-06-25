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

