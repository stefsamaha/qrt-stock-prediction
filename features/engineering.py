import pandas as pd
import numpy as np

def build_features(df):
    """
    Feature engineering pipeline for QRT stock return prediction.
    All features derived from RET_1-20 and VOLUME_1-20.
    
    Design philosophy:
    - Short-term reversal: RET_1 negatively predicts next day (EDA confirmed -0.017 correlation)
    - Medium-term momentum: RET_5-10 shows weak positive correlation
    - Volume-return interaction: high volume amplifies signal
    - Realized volatility: controls for varying stock risk regimes
    """
    
    features = pd.DataFrame(index=df.index)
    
    ret_cols = [f'RET_{i}' for i in range(1, 21)]
    vol_cols = [f'VOLUME_{i}' for i in range(1, 21)]
    
    # ── 1. RAW LAGS (baseline) ──────────────────────────────────────────
    # Keep raw returns as features — the benchmark uses these directly
    for col in ret_cols:
        features[col] = df[col]
    for col in vol_cols:
        features[col] = df[col]
    
    # ── 2. SHORT-TERM REVERSAL ──────────────────────────────────────────
    # EDA showed RET_1 has strongest negative correlation (-0.017)
    # Reversal effect: yesterday's loser tends to bounce back
    features['reversal_1'] = -df['RET_1']  # flip sign: negative return → positive prediction
    features['reversal_2'] = -df['RET_2']
    features['reversal_1_2_avg'] = -(df['RET_1'] + df['RET_2']) / 2
    
    # ── 3. MOMENTUM FEATURES ───────────────────────────────────────────
    # EDA showed weak positive correlation at RET_7 and RET_10
    # Momentum: stock trending up over past weeks tends to continue
    features['momentum_5_10'] = df[[f'RET_{i}' for i in range(5, 11)]].mean(axis=1)
    features['momentum_10_20'] = df[[f'RET_{i}' for i in range(10, 21)]].mean(axis=1)
    features['momentum_full'] = df[ret_cols].mean(axis=1)
    
    # ── 4. REALIZED VOLATILITY ─────────────────────────────────────────
    # Std of past returns — high vol stocks behave differently
    # High volatility = less predictable, noisier signal
    features['realized_vol_5'] = df[[f'RET_{i}' for i in range(1, 6)]].std(axis=1)
    features['realized_vol_20'] = df[ret_cols].std(axis=1)
    
    # ── 5. SIGN CONSISTENCY ────────────────────────────────────────────
    # How many of last 5 days were positive?
    # Captures trend persistence better than average return
    features['sign_consistency_5'] = df[[f'RET_{i}' for i in range(1, 6)]].apply(
        lambda x: (x > 0).sum(), axis=1
    )
    features['sign_consistency_10'] = df[[f'RET_{i}' for i in range(1, 11)]].apply(
        lambda x: (x > 0).sum(), axis=1
    )
    
    # ── 6. VOLUME-RETURN INTERACTION ───────────────────────────────────
    # High volume on a down day = stronger reversal signal
    # High volume on an up day = stronger momentum signal
    # This is more informative than volume or return alone
    features['vol_ret_interaction_1'] = df['VOLUME_1'] * df['RET_1']
    features['vol_ret_interaction_2'] = df['VOLUME_2'] * df['RET_2']
    
    # ── 7. VOLUME TREND ────────────────────────────────────────────────
    # Is volume increasing or decreasing recently?
    # Rising volume often precedes larger price moves
    features['volume_trend'] = (
        df[[f'VOLUME_{i}' for i in range(1, 6)]].mean(axis=1) -
        df[[f'VOLUME_{i}' for i in range(6, 11)]].mean(axis=1)
    )
    features['volume_recent_avg'] = df[[f'VOLUME_{i}' for i in range(1, 6)]].mean(axis=1)
    
    # ── 8. SECTOR DUMMIES ──────────────────────────────────────────────
    # Sector membership as categorical features
    # Different sectors have different return dynamics
    features['SECTOR'] = df['SECTOR']
    features['INDUSTRY'] = df['INDUSTRY']

    # ── 9. SECTOR-CONDITIONAL FEATURES ────────────────────────────────
    # For each stock, compute its return relative to its sector average
    # This captures whether a stock is outperforming or underperforming peers
    # The benchmark's strongest feature was a sector conditional mean
    for lag in [1, 2, 3, 5]:
        col = f'RET_{lag}'
        sector_mean = df.groupby(['DATE', 'SECTOR'])[col].transform('mean')
        features[f'sector_mean_ret_{lag}'] = sector_mean
        features[f'ret_vs_sector_{lag}'] = df[col] - sector_mean

    # ── 10. VOLATILITY-NORMALIZED RETURNS ─────────────────────────────
    # Divide return signals by realized volatility
    # A -2% move in a 0.5% vol stock is huge. Same move in 5% vol stock is normal.
    epsilon = 1e-8  # prevent division by zero
    features['vol_adj_ret_1'] = df['RET_1'] / (features['realized_vol_5'] + epsilon)
    features['vol_adj_ret_2'] = df['RET_2'] / (features['realized_vol_5'] + epsilon)
    features['vol_adj_momentum'] = (
        df[[f'RET_{i}' for i in range(5, 11)]].mean(axis=1) / 
        (features['realized_vol_20'] + epsilon)
    )
    
    # ── 11. VOLATILITY REGIME ──────────────────────────────────────────
    # Is volatility increasing or decreasing recently?
    # Ratio > 1 means volatility picking up — often precedes larger moves
    features['vol_regime'] = (
        features['realized_vol_5'] / (features['realized_vol_20'] + epsilon)
    )

    # ── 12. CROSS-SECTIONAL RANK FEATURES ─────────────────────────────
    # Rank each stock against ALL other stocks on the same day.
    # Target is residual (market-neutral) return, so relative position
    # matters more than absolute return. This is the core of
    # cross-sectional equity prediction.
    for lag in [1, 2, 3, 5]:
        col = f'RET_{lag}'
        features[f'rank_ret_{lag}'] = df.groupby('DATE')[col].rank(pct=True)

    # Rank of recent 5-day momentum across all stocks that day
    recent_mom = df[[f'RET_{i}' for i in range(1, 6)]].mean(axis=1)
    features['rank_momentum'] = recent_mom.groupby(df['DATE']).rank(pct=True)

    # Rank of volatility — which stocks are most volatile today
    vol_temp = df[[f'RET_{i}' for i in range(1, 21)]].std(axis=1)
    features['rank_volatility'] = vol_temp.groupby(df['DATE']).rank(pct=True)

    # ── 13. VOLUME-CONDITIONED REVERSAL ───────────────────────────────
    # Reversal is stronger after high-volume moves. High volume means
    # market makers were forced to displace price from fair value,
    # creating the mispricing that corrects the next day.
    features['reversal_x_volume'] = -df['RET_1'] * df['VOLUME_1'].abs()
    features['reversal_x_volume_2'] = -df['RET_2'] * df['VOLUME_2'].abs()

    high_vol_flag = (df['VOLUME_1'] > df['VOLUME_1'].median()).astype(int)
    features['conditional_reversal'] = -df['RET_1'] * high_vol_flag

    # ── 14. RETURN AND VOLATILITY ACCELERATION ────────────────────────
    # Is the trend speeding up or slowing down? Deceleration often
    # precedes reversal — a trend-exhaustion signal. Second-order
    # information that raw lagged returns do not contain.
    recent = df[[f'RET_{i}' for i in range(1, 4)]].mean(axis=1)
    older = df[[f'RET_{i}' for i in range(4, 8)]].mean(axis=1)
    features['return_acceleration'] = recent - older

    recent_vol = df[[f'RET_{i}' for i in range(1, 6)]].std(axis=1)
    older_vol = df[[f'RET_{i}' for i in range(6, 11)]].std(axis=1)
    features['vol_acceleration'] = recent_vol - older_vol

    return features


def build_features_train_test(train_df, test_df):
    """Build features for both train and test consistently."""
    X_train = build_features(train_df)
    X_test = build_features(test_df)
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"\nFeature list ({X_train.shape[1]} features):")
    print(list(X_train.columns))
    
    return X_train, X_test


