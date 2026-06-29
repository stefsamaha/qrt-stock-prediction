# QRT Stock Return Prediction - ENS Data Challenge #23

Predicting the sign of next-day residual US stock returns using 20 days of historical returns and volume data. Built as part of the QRT Quantitative Research Data Challenge hosted on ENS Challenge Data.

## Problem

Binary classification: given 20 days of residual returns and relative volumes for a stock, predict whether tomorrow's residual return will be positive or negative. 418,595 training observations across multiple stocks, sectors, and industries.

**Benchmark:** Random Forest — 51.31% accuracy

## Approach

- Exploratory data analysis of return distributions, autocorrelation, and volume patterns
- Feature engineering: realized volatility, return autocorrelation, volume-return interactions, cross-sectional sector signals
- Model comparison: Logistic Regression, Random Forest, LightGBM, ensembles
- Strict look-ahead bias controls throughout

## Structure
 
 ```
data/           # raw data files (not tracked)
eda/            # exploratory analysis notebooks
features/       # feature engineering pipeline
models/         # baseline and experimental models
notebooks/      # end-to-end experiment notebooks
results/        # experiment scores and notes
submissions/    # competition submission files
```

## Results

Final model: soft-voting ensemble of LightGBM, Random Forest, Logistic 
Regression, and a small regularized MLP, over 79 engineered features.

| Model | Public Accuracy |
|-------|----------------|
| Benchmark (Random Forest) | 51.31% |
| LightGBM baseline (v1) | 50.73% |
| + sector & volatility features (v2) | 51.30% |
| + ensemble LGBM/RF/LR (v4) | 51.32% |
| + cross-sectional rank features (v6) | 51.52% |
| + decorrelated MLP (v7, final) | **51.54%** |

Final result is **+0.23% above the benchmark**. All scores are public 
leaderboard, validated against 5-fold time-series cross-validation.

## What Actually Moved the Score

Of every technique tried, only two produced real out-of-sample gains:

- **Cross-sectional rank features** (+0.20%). Ranking each stock against 
  all others on the same day. Since the target is a residual 
  (market-neutral) return, relative position predicts better than absolute 
  return — the features were realigned to match the target's structure.
- **A decorrelated MLP** (+0.02%, marginal). Added only after confirming 
  its predictions correlated 0.49 with LightGBM — low enough to contribute 
  genuine diversity rather than redundancy.

## What Did Not Work — and Why That Matters

Five techniques were tested with honest cross-validation and rejected. The 
negative results are the substance of this project:

- **Hyperparameter tuning hurt.** Aggressive regularization dropped CV to 
  51.04%. The default-ish parameters were already well-calibrated; the model 
  was not overfitting, so constraining it further only added bias.
- **Stacking hurt.** A logistic meta-learner on out-of-fold base predictions 
  underperformed simple averaging. On a ~1%-signal problem, the meta-layer 
  overfit the base models' validation noise — variance from learning weights 
  exceeded the bias reduction. Equal-weight averaging is more robust here.
- **Ensemble reweighting and per-date thresholds were flat.** Upweighting 
  the most decorrelated model and forcing a per-day median split both moved 
  results by <0.0001 — confirming the base probabilities were already well 
  calibrated cross-sectionally.
- **Stock-level target encoding showed no out-of-sample lift.** Leakage-safe 
  out-of-fold encoding of stock identity had meaningful in-sample spread 
  (per-stock target means ranged 0.30–0.70) but zero CV improvement. This 
  empirically confirms residual returns carry **no persistent per-stock 
  directional bias** — consistent with market efficiency. A naive full-data 
  encoding would have shown a spurious 3–4% gain; the out-of-fold setup 
  correctly rejected it.

## Validation Methodology

- **Time-series cross-validation** (`TimeSeriesSplit`, 5 folds, no shuffling) 
  throughout, to prevent look-ahead bias in a temporal problem.
- **Winsorization bounds and target encodings computed on training folds 
  only**, then applied to validation/test — no information leakage from the 
  evaluation set into preprocessing.
- **CV–leaderboard calibration tracked** across submissions. The gap stayed 
  small and sign-variable (~±0.001), confirming the local CV was a 
  trustworthy proxy for the leaderboard.

## Key Diagnostic: Fold 5

One CV fold consistently underperformed (~49.5%). Investigation showed it 
had normal realized volatility (0.0246) and normal class balance (0.4987) — 
not a regime break. Accounting for within-day return correlation inflating 
the effective standard error, the fold was within noise. It was not treated 
as a signal to optimize against. The final model's public score later 
exceeded its CV mean, confirming the test period resembled the stronger folds.

## Conclusion

The signal ceiling for tree-based and linear models on this representation is 
~51.5%. Six independent levers were tested; one produced a real gain, one a 
marginal gain, four were flat or negative. The project's value is less the 
score than the disciplined separation of genuine signal from noise on a 
low-signal-to-noise financial prediction problem.

## Setup

    pip install -r requirements.txt

Data files available via ENS Challenge Data platform (registration required): https://challengedata.ens.fr/challenges/23