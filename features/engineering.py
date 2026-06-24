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