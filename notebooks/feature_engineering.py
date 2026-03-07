"""
FILE 1 — feature_engineering.py
════════════════════════════════════════════════════════════════════════════════
INPUT : combinedBankData.csv  (output of add_columns.ipynb)
        Columns: date, open, high, low, close, per_change, volume, amount,
                 bank, nepse_close, nepse_bull, nepse_ret_1d, nepse_ret_5d,
                 nepse_ret_21d, policy_rate, car, npl

OUTPUT: enriched_features.csv
        All input columns + 73 ML features + target columns
        Ready to feed directly into train_model.py

USAGE:
    python feature_engineering.py
    python feature_engineering.py --input path/to/combinedBankData.csv --output path/to/enriched_features.csv
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
ACTIVE_BANKS = [
    'ADBL','CZBIL','EBL','GBIME','HBL','KBL','MBL',
    'NABIL','NICA','NMB','PCBL','PRVU','SANIMA','SBL','SCB'
]

# All 73 features the model expects — DO NOT change order
MODEL_FEATURES = [
    # ── Calendar (3) ──────────────────────────────────────────────
    'fy_end', 'month', 'q2_results',

    # ── Sector / Cross-bank (19) ──────────────────────────────────
    'sec_drawdown', 'sec_bull_pct', 'sec_rvol', 'sec_bull_50pct',
    'sec_ret_5d', 'sec_ret_10d', 'sec_ret_21d', 'sec_ret_42d',
    'sec_breadth', 'sec_ret21_std', 'rsi_vs_sec', 'rel_drawdown',
    'rel_str_5d', 'rel_str_10d', 'rel_str_21d', 'rel_str_42d',
    'xrank_ret_5d', 'xrank_ret_63d', 'xrank_rvol21',

    # ── EMA trend (5) ─────────────────────────────────────────────
    'cross_21_50', 'cross_50_100', 'cross_50_200', 'p_ema50', 'p_ema200',

    # ── Oscillators (7) ───────────────────────────────────────────
    'macd_norm', 'rsi14_zscore', 'rsi21', 'stoch_k',
    'bb_pct_20', 'bb_width_14', 'bb_width_20',

    # ── Volatility (7) ────────────────────────────────────────────
    'rvol5', 'rvol10', 'rvol21', 'rvol63',
    'vol_regime', 'atr14_pct', 'atr_regime',

    # ── Volume (4) ────────────────────────────────────────────────
    'vol_r5', 'vol_r60', 'obv_trend', 'bull_x_volume',

    # ── Returns (8) ───────────────────────────────────────────────
    'ret_2d', 'ret_3d', 'ret_21d', 'ret_63d',
    'ret_skew21', 'ret_autocorr', 'ret21_vadjusted', 'gap_pct',

    # ── 52-week levels (2) ────────────────────────────────────────
    'range52_pos', 'dist_high52',

    # ── Interactions (4) ──────────────────────────────────────────
    'bull_x_rsi', 'bull_x_macd', 'easy_x_bull', 'breadth_x_bull',

    # ── NEW: NEPSE relative (3) ───────────────────────────────────
    'p_vs_nepse_21d', 'p_vs_nepse_5d', 'nepse_x_bull',

    # ── NEW: Fundamentals per-bank (8) ───────────────────────────
    'car', 'npl', 'car_zscore', 'npl_zscore',
    'npl_trend', 'car_trend', 'npl_x_bear', 'car_x_bull',

    # ── NEW: Fundamental cross-bank (3) ──────────────────────────
    'xrank_car', 'xrank_npl', 'rel_npl',
]

TARGET_COLS = ['target_dir', 'target_mag', 'target_mom5', 'fwd_ret_5d', 'fwd_ret_21d']


# ── STEP 1: LOAD & VALIDATE ───────────────────────────────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    print(f"  Loading: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['bank']  = df['bank'].str.strip()
    df['date']  = pd.to_datetime(df['date'])

    required = ['date','open','high','low','close','volume','amount',
                'bank','nepse_close','nepse_bull','nepse_ret_1d',
                'nepse_ret_5d','nepse_ret_21d','policy_rate','car','npl']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from notebook output: {missing}\n"
                         f"Run add_columns.ipynb first.")

    # Basic data quality
    df['open']  = df['open'].clip(lower=df['low'], upper=df['high'])
    df.loc[df['amount'] == 0, 'amount'] = np.nan
    df = df.drop_duplicates(subset=['bank','date'], keep='last')
    df = df.sort_values(['bank','date']).reset_index(drop=True)

    print(f"  Loaded: {len(df):,} rows | {df['bank'].nunique()} banks | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── STEP 2: PER-BANK TECHNICAL FEATURES ──────────────────────────────────────
def add_bank_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add all per-bank technical features. Call per bank group."""
    g = g.sort_values('date').copy()
    c, v, h, l, o = g['close'], g['volume'], g['high'], g['low'], g['open']
    log = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)

    # EMAs
    ema21  = c.ewm(span=21,  min_periods=10).mean()
    ema50  = c.ewm(span=50,  min_periods=25).mean()
    ema100 = c.ewm(span=100, min_periods=50).mean()
    ema200 = c.ewm(span=200, min_periods=100).mean()

    # RSI-14
    d     = c.diff()
    gain  = d.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss  = (-d.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean().add(1e-9)
    rsi14 = 100 - (100 / (1 + gain / loss))

    # MACD
    macd_raw = c.ewm(span=12, min_periods=12).mean() - c.ewm(span=26, min_periods=26).mean()
    macd_sig = macd_raw.ewm(span=9, min_periods=9).mean()

    # ATR
    tr    = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, min_periods=14).mean()

    # 52-week
    high52 = h.rolling(252, min_periods=60).max()
    low52  = l.rolling(252, min_periods=60).min()

    # Regime flags
    drawdown = c / c.expanding().max() - 1
    bull200  = (c > ema200).astype(int)
    bull50   = (c > ema50).astype(int)

    # ── Calendar ──────────────────────────────────────────────────
    g['month']      = g['date'].dt.month
    g['fy_end']     = g['month'].isin([5, 6, 7]).astype(int)
    g['q2_results'] = g['month'].isin([1, 2]).astype(int)

    # ── EMA crossovers ────────────────────────────────────────────
    g['cross_21_50']  = ema21  / ema50  - 1
    g['cross_50_100'] = ema50  / ema100 - 1
    g['cross_50_200'] = ema50  / ema200 - 1
    g['p_ema50']      = c / ema50  - 1
    g['p_ema200']     = c / ema200 - 1

    # ── MACD normalised (scale-invariant) ─────────────────────────
    macd_mu  = macd_raw.rolling(63, min_periods=30).mean()
    macd_std = macd_raw.rolling(63, min_periods=30).std()
    g['macd_norm'] = (macd_raw - macd_mu) / (macd_std + 1e-9)

    # ── RSI-21 + z-scored RSI-14 ──────────────────────────────────
    d21   = c.diff()
    g21   = d21.clip(lower=0).ewm(alpha=1/21, min_periods=21).mean()
    l21   = (-d21.clip(upper=0)).ewm(alpha=1/21, min_periods=21).mean().add(1e-9)
    g['rsi21'] = 100 - (100 / (1 + g21 / l21))
    rsi14_mu  = rsi14.rolling(63, min_periods=30).mean()
    rsi14_std = rsi14.rolling(63, min_periods=30).std()
    g['rsi14_zscore'] = (rsi14 - rsi14_mu) / (rsi14_std + 1e-9)

    # ── Stochastic %K ─────────────────────────────────────────────
    lo14 = l.rolling(14, min_periods=10).min()
    hi14 = h.rolling(14, min_periods=10).max()
    g['stoch_k'] = (c - lo14) / (hi14 - lo14 + 1e-9) * 100

    # ── Bollinger Bands ───────────────────────────────────────────
    bb_ma20  = c.rolling(20, min_periods=12).mean()
    bb_std20 = c.rolling(20, min_periods=12).std()
    g['bb_pct_20']   = (c - (bb_ma20 - 2*bb_std20)) / (4*bb_std20 + 1e-9)
    g['bb_width_14'] = 4*c.rolling(14, min_periods=8).std() / (c.rolling(14, min_periods=8).mean() + 1e-9)
    g['bb_width_20'] = 4*bb_std20 / (bb_ma20 + 1e-9)

    # ── Realised volatility ───────────────────────────────────────
    for w in [5, 10, 21, 63]:
        g[f'rvol{w}'] = log.rolling(w, min_periods=w//2).std() * np.sqrt(252)
    g['vol_regime'] = g['rvol21'].rolling(252, min_periods=60).rank(pct=True)
    g['atr14_pct']  = atr14 / (c + 1e-9)
    g['atr_regime'] = atr14 / atr14.rolling(63, min_periods=30).mean()

    # ── Volume ratios ─────────────────────────────────────────────
    g['vol_r5']  = v / (v.rolling(5,  min_periods=3).mean()  + 1e-9)
    g['vol_r60'] = v / (v.rolling(60, min_periods=30).mean() + 1e-9)
    obv = (np.sign(log) * v).fillna(0).cumsum()
    g['obv_trend'] = (obv > obv.ewm(span=20, min_periods=10).mean()).astype(int)

    # ── Returns ───────────────────────────────────────────────────
    g['ret_2d']  = c.pct_change(2)
    g['ret_3d']  = c.pct_change(3)
    g['ret_21d'] = c.pct_change(21)
    g['ret_63d'] = c.pct_change(63)
    g['gap_pct'] = (o - c.shift(1)) / (c.shift(1) + 1e-9)
    g['ret21_vadjusted'] = g['ret_21d'] / (g['rvol21'] + 1e-9)
    g['ret_skew21'] = log.rolling(21, min_periods=15).skew()
    g['ret_autocorr'] = log.rolling(10, min_periods=8).apply(
        lambda x: x[:5].mean() - x[5:].mean() if len(x) == 10 else np.nan, raw=True)

    # ── 52-week position ──────────────────────────────────────────
    g['range52_pos'] = (c - low52) / (high52 - low52 + 1e-9)
    g['dist_high52'] = c / (high52 + 1e-9) - 1

    # ── Interaction features ──────────────────────────────────────
    g['bull_x_rsi']    = bull200 * rsi14
    g['bull_x_macd']   = bull200 * (macd_raw > macd_sig).astype(int)
    g['bull_x_volume'] = bull200 * g['vol_r60']
    g['easy_x_bull']   = (g['policy_rate'] < 5.0).astype(int) * bull200

    # ── NEPSE relative performance ────────────────────────────────
    g['p_vs_nepse_21d'] = g['ret_21d'] - g['nepse_ret_21d']
    g['p_vs_nepse_5d']  = g['ret_3d']  - g['nepse_ret_5d']
    g['nepse_x_bull']   = g['nepse_bull'].fillna(0) * bull200

    # ── Fundamental dynamics (CAR / NPL) ─────────────────────────
    g['car_zscore'] = (g['car'] - g['car'].rolling(252, min_periods=60).mean()) / \
                      (g['car'].rolling(252, min_periods=60).std() + 1e-9)
    g['npl_zscore'] = (g['npl'] - g['npl'].rolling(252, min_periods=60).mean()) / \
                      (g['npl'].rolling(252, min_periods=60).std() + 1e-9)
    g['npl_trend']  = g['npl'] - g['npl'].shift(252)
    g['car_trend']  = g['car'] - g['car'].shift(252)
    g['npl_x_bear'] = (g['npl'] > 3.0).astype(int) * (1 - bull200)
    g['car_x_bull'] = (g['car'] > 14.0).astype(int) * bull200

    # ── Temp columns for sector aggregation ──────────────────────
    g['_rsi14']    = rsi14
    g['_bull200']  = bull200
    g['_bull50']   = bull50
    g['_drawdown'] = drawdown
    g['_near_high']= (g['dist_high52'] > -0.05).astype(int)

    return g


# ── STEP 3: CROSS-BANK SECTOR FEATURES ───────────────────────────────────────
def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across all banks on the same date. Requires full panel."""

    # Return-based sector features (use ret_2d/3d/21d/63d as proxies for 5d/10d/21d/42d)
    for src, dst in [('ret_2d','5d'), ('ret_3d','10d'), ('ret_21d','21d'), ('ret_63d','42d')]:
        avg = df.groupby('date')[src].transform('mean')
        df[f'sec_ret_{dst}']  = avg
        df[f'rel_str_{dst}']  = df[src] - avg

    df['sec_ret21_std']  = df.groupby('date')['ret_21d'].transform('std')
    df['rsi_vs_sec']     = df['_rsi14'] - df.groupby('date')['_rsi14'].transform('mean')
    df['sec_bull_pct']   = df.groupby('date')['_bull200'].transform('mean')
    df['sec_bull_50pct'] = df.groupby('date')['_bull50'].transform('mean')
    df['sec_drawdown']   = df.groupby('date')['_drawdown'].transform('mean')
    df['rel_drawdown']   = df['_drawdown'] - df['sec_drawdown']
    df['sec_rvol']       = df.groupby('date')['rvol21'].transform('mean')
    df['sec_breadth']    = df.groupby('date')['_near_high'].transform('mean')

    # Cross-sectional percentile ranks
    df['xrank_ret_5d']  = df.groupby('date')['ret_21d'].rank(pct=True)
    df['xrank_ret_63d'] = df.groupby('date')['ret_63d'].rank(pct=True)
    df['xrank_rvol21']  = df.groupby('date')['rvol21'].rank(pct=True)

    # Interaction
    df['breadth_x_bull'] = df['sec_breadth'] * df['_bull200']

    # Fundamental cross-bank
    df['xrank_car']   = df.groupby('date')['car'].rank(pct=True)
    df['xrank_npl']   = df.groupby('date')['npl'].rank(pct=True)
    df['rel_npl']     = df['npl'] - df.groupby('date')['npl'].transform('mean')

    # Drop temp columns
    df.drop(columns=[c for c in df.columns if c.startswith('_')], inplace=True)
    return df


# ── STEP 4: TARGETS ───────────────────────────────────────────────────────────
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-looking targets (used for training only, NaN at inference time)."""
    for hz in [5, 21]:
        df[f'fwd_ret_{hz}d'] = (df.groupby('bank')['close']
            .transform(lambda x: x.shift(-hz) / x - 1)
            .clip(-0.35, 0.35))
    df['target_dir']  = (df['fwd_ret_21d'] > 0).astype(int)
    df['target_mag']  = df['fwd_ret_21d']
    df['target_mom5'] = (df['fwd_ret_5d']  > 0).astype(int)
    return df


# ── STEP 5: NULL HANDLING ─────────────────────────────────────────────────────
def handle_nulls(df: pd.DataFrame, min_year: int = 2013,
                 drop_targets: bool = True) -> pd.DataFrame:
    """
    Filter ≥ min_year (removes warm-up period for early-listed banks),
    then dropna on MODEL_FEATURES only.
    Nulls are purely rolling-window warm-up artefacts — no imputation needed.
    """
    before = len(df)
    df = df[df['date'].dt.year >= min_year].copy()
    drop_cols = MODEL_FEATURES + (TARGET_COLS if drop_targets else [])
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.dropna(subset=drop_cols)
    after = len(df)
    print(f"  Null handling: {before:,} → {after:,} rows  "
          f"(dropped {before-after:,} warm-up rows, {(before-after)/before:.1%})")
    return df.reset_index(drop=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(input_path: str, output_path: str) -> pd.DataFrame:
    t0 = time.time()
    print("\n" + "="*65)
    print(" STEP 1 — Feature Engineering")
    print("="*65)

    # Load
    print("\n[1/5] Load & validate...")
    df = load_and_validate(input_path)

    # Per-bank features
    print("\n[2/5] Per-bank technical features...")
    groups = [add_bank_features(g) for _, g in df.groupby('bank')]
    df = pd.concat(groups).sort_values(['bank', 'date']).reset_index(drop=True)
    print(f"  Done. Shape: {df.shape}")

    # Sector features
    print("\n[3/5] Cross-bank sector features...")
    df = add_sector_features(df)
    print(f"  Done. Shape: {df.shape}")

    # Targets
    print("\n[4/5] Forward-looking targets...")
    df = add_targets(df)

    # Null handling
    print("\n[5/5] Null handling (≥2013 + dropna)...")
    df = handle_nulls(df, min_year=2013, drop_targets=True)

    # Validate all features present
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"Features missing after engineering: {missing}")

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path}")
    print(f"  Final shape: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Feature cols: {len(MODEL_FEATURES)}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Banks: {sorted(df['bank'].unique())}")
    print(f"  Elapsed: {time.time()-t0:.1f}s\n")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature engineering for NEPSE ML pipeline')
    parser.add_argument('--input',  default='../data/processed/combinedBankData.csv',
                        help='Path to notebook output CSV')
    parser.add_argument('--output', default='enriched_features.csv',
                        help='Path to save enriched features CSV')
    args = parser.parse_args()
    run(args.input, args.output)
