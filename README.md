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

| Model | Public Accuracy |
|-------|----------------|
| Benchmark (RF) | 51.31% |
| ... | ... |

*Results updated as experiments progress.*

## Setup

    pip install -r requirements.txt

Data files available via ENS Challenge Data platform (registration required): https://challengedata.ens.fr/challenges/23