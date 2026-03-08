"""
FILE 3 — inference_pipeline.py
════════════════════════════════════════════════════════════════════════════════
The full inference path from raw scraped data → trading signals.

Takes exactly what you'd scrape from NEPSE at runtime:
  - OHLCV for one or more commercial banks (last ~300 trading days)
  - NEPSE index history (last ~300 trading days)
  - CAR / NPL values (latest annual figures per bank)
  - Current policy rate

Runs the complete feature engineering pipeline (same as training)
and returns a signal for each bank:  LONG / HOLD / SKIP

USAGE — as a standalone script:
    python inference_pipeline.py

USAGE — as a library (import into your backend):
    from inference_pipeline import NEPSEInferencePipeline
    pipeline = NEPSEInferencePipeline('model_artifacts/')
    signals  = pipeline.predict(ohlcv_df, nepse_df, fundamentals)
    print(signals)

INPUT CONTRACT:
    ohlcv_df : pd.DataFrame with columns:
               date (date), bank (str), open, high, low, close, volume, amount
               Must have at least 250 rows per bank (for EMA-200 warm-up)

    nepse_df : pd.DataFrame with columns:
               date (date), nepse_close (float)
               Must cover the same date range as ohlcv_df

    fundamentals : dict  {
        'NABIL': {'car': 12.71, 'npl': 1.54},
        'HBL':   {'car': 12.04, 'npl': 2.06},
        ...
    }
    policy_rate : float  (current NRB policy rate, e.g. 4.5)

OUTPUT:
    pd.DataFrame with one row per bank:
    bank | date | close | prob_direction | prob_momentum | ensemble_score | signal | car | npl
════════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import pickle
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_ROWS_REQUIRED = 250   # minimum OHLCV rows needed per bank for reliable features
POLICY_RATE_FALLBACK = 4.5


class NEPSEInferencePipeline:
    """
    End-to-end inference pipeline for NEPSE commercial bank signals.

    Load once, call predict() any time new data arrives.
    All feature engineering is identical to training — no drift.
    """

    def __init__(self, model_dir: str = 'model_artifacts', verbose: bool = True):
        self.model_dir = model_dir
        self.verbose = verbose
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all model artifacts from disk."""
        self._log(f"Loading model artifacts from: {self.model_dir}/")

        def load(name):
            path = os.path.join(self.model_dir, f'model_{name}.pkl')
            with open(path, 'rb') as f:
                return pickle.load(f)

        self.clf_dir = load('clf_dir')
        self.clf_mom = load('clf_mom')
        self.reg_mag = load('reg_mag')
        self.scaler  = load('scaler')

        meta_path = os.path.join(self.model_dir, 'model_meta.json')
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.features        = self.meta['model_features']
        self.long_thresh     = self.meta['signal_thresholds']['LONG']
        self.hold_thresh     = self.meta['signal_thresholds']['HOLD']

        self._log(f"  Models loaded. Features: {len(self.features)}")
        self._log(f"  Trained on data up to: {self.meta['train_end']}")
        self._log(f"  Validated AUC: {self.meta['metrics']['te_auc_ens']:.4f}")

    # ── PUBLIC API ────────────────────────────────────────────────────────────
    def predict(
        self,
        ohlcv_df:     pd.DataFrame,
        nepse_df:     pd.DataFrame,
        fundamentals: Dict[str, Dict[str, float]],
        policy_rate:  float = POLICY_RATE_FALLBACK,
        verbose:      bool  = True,
    ) -> pd.DataFrame:
        """
        Main inference function.

        Parameters
        ----------
        ohlcv_df     : OHLCV data for one or more banks (≥250 rows per bank)
        nepse_df     : NEPSE index history
        fundamentals : {'NABIL': {'car': 12.71, 'npl': 1.54}, ...}
        policy_rate  : current NRB policy rate
        verbose      : print signal table to stdout

        Returns
        -------
        pd.DataFrame : signals for each bank
        """
        # Validate and clean inputs
        ohlcv   = self._prepare_ohlcv(ohlcv_df)
        nepse   = self._prepare_nepse(nepse_df)
        banks   = sorted(ohlcv['bank'].unique())

        if verbose:
            print(f"\nRunning inference for {len(banks)} banks: {banks}")
            print(f"  OHLCV rows: {len(ohlcv):,} | "
                  f"Date range: {ohlcv['date'].min().date()} → {ohlcv['date'].max().date()}")

        # Merge NEPSE + policy rate + fundamentals
        df = self._merge_external(ohlcv, nepse, fundamentals, policy_rate)

        # Per-bank technical features
        groups = [self._add_bank_features(g) for _, g in df.groupby('bank')]
        df = pd.concat(groups).sort_values(['bank', 'date']).reset_index(drop=True)

        # Sector features (requires all banks together)
        df = self._add_sector_features(df)

        # Get latest row per bank (the actual inference point)
        latest = df.groupby('bank').last().reset_index()

        # Check all features available
        missing = [f for f in self.features if f not in latest.columns]
        if missing:
            raise RuntimeError(f"Features missing at inference: {missing}\n"
                               f"Check that ohlcv_df has enough history (≥250 rows).")

        X = latest[self.features].copy()

        # Run models
        prob_dir = self.clf_dir.predict_proba(X)[:, 1]
        prob_mom = self.clf_mom.predict_proba(X)[:, 1]
        mag      = self.reg_mag.predict(X)

        # Ensemble score
        mag_norm = self.scaler.transform(mag.reshape(-1, 1)).ravel()
        ens      = 0.6 * prob_dir + 0.4 * mag_norm

        # Build output
        signals = latest[['bank', 'date', 'close', 'car', 'npl']].copy()
        signals['prob_direction'] = np.round(prob_dir, 4)
        signals['prob_momentum']  = np.round(prob_mom, 4)
        signals['predicted_mag']  = np.round(mag * 100, 2)       # as %
        signals['ensemble_score'] = np.round(ens, 4)
        signals['signal'] = signals['ensemble_score'].apply(self._score_to_signal)
        signals = signals.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

        if verbose:
            self._print_signals(signals)

        return signals

    # ── INPUT PREPARATION ────────────────────────────────────────────────────
    def _prepare_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ['date', 'bank', 'open', 'high', 'low', 'close', 'volume']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"ohlcv_df missing columns: {missing}")

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['bank'] = df['bank'].str.strip().str.upper()
        df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])
        if 'amount' not in df.columns:
            df['amount'] = np.nan
        df.loc[df.get('amount', pd.Series(dtype=float)) == 0, 'amount'] = np.nan
        df = df.drop_duplicates(subset=['bank', 'date'], keep='last')
        df = df.sort_values(['bank', 'date']).reset_index(drop=True)

        # Warn on insufficient history
        for bk, g in df.groupby('bank'):
            if len(g) < MIN_ROWS_REQUIRED:
                self._log(
                    f"  WARNING: {bk} has only {len(g)} rows "
                    f"(need >= {MIN_ROWS_REQUIRED}). Features may be NaN."
                )
        return df

    def _prepare_nepse(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'nepse_close' not in df.columns and 'close' in df.columns:
            df = df.rename(columns={'close': 'nepse_close'})
        required = ['date', 'nepse_close']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"nepse_df missing columns: {missing}")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Compute NEPSE features
        c = df['nepse_close']
        df['nepse_ema200']  = c.ewm(span=200, min_periods=100).mean()
        df['nepse_bull']    = (c > df['nepse_ema200']).astype(int)
        df['nepse_ret_1d']  = c.pct_change(1)
        df['nepse_ret_5d']  = c.pct_change(5)
        df['nepse_ret_21d'] = c.pct_change(21)
        return df.drop(columns=['nepse_ema200'])

    def _merge_external(self, ohlcv, nepse, fundamentals, policy_rate):
        df = ohlcv.merge(
            nepse[['date','nepse_close','nepse_bull',
                   'nepse_ret_1d','nepse_ret_5d','nepse_ret_21d']],
            on='date', how='left'
        )
        df['policy_rate'] = policy_rate

        # Forward-fill NEPSE for any market holidays
        nepse_cols = ['nepse_close','nepse_bull','nepse_ret_1d','nepse_ret_5d','nepse_ret_21d']
        df[nepse_cols] = df[nepse_cols].ffill()

        # Attach fundamentals (CAR / NPL)
        for bk in df['bank'].unique():
            mask = df['bank'] == bk
            if bk in fundamentals:
                df.loc[mask, 'car'] = fundamentals[bk].get('car', np.nan)
                df.loc[mask, 'npl'] = fundamentals[bk].get('npl', np.nan)
            else:
                # Use sector median as fallback
                df.loc[mask, 'car'] = np.nan
                df.loc[mask, 'npl'] = np.nan
                self._log(f"  WARNING: No fundamentals for {bk}. Using NaN (model will estimate).")

        # Fill any missing fundamental with cross-bank median
        for col in ['car', 'npl']:
            med = df.groupby('date')[col].transform('median')
            df[col] = df[col].fillna(med)

        return df

    # ── FEATURE ENGINEERING (identical to training) ───────────────────────────
    def _add_bank_features(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('date').copy()
        c, v, h, l, o = g['close'], g['volume'], g['high'], g['low'], g['open']
        log = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)

        ema21  = c.ewm(span=21,  min_periods=10).mean()
        ema50  = c.ewm(span=50,  min_periods=25).mean()
        ema100 = c.ewm(span=100, min_periods=50).mean()
        ema200 = c.ewm(span=200, min_periods=100).mean()
        d      = c.diff()
        gain   = d.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
        loss   = (-d.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean().add(1e-9)
        rsi14  = 100 - (100 / (1 + gain / loss))
        macd_raw = c.ewm(span=12, min_periods=12).mean() - c.ewm(span=26, min_periods=26).mean()
        macd_sig = macd_raw.ewm(span=9, min_periods=9).mean()
        tr     = pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr14  = tr.ewm(span=14, min_periods=14).mean()
        high52 = h.rolling(252, min_periods=60).max()
        low52  = l.rolling(252, min_periods=60).min()
        drawdown = c / c.expanding().max() - 1
        bull200  = (c > ema200).astype(int)
        bull50   = (c > ema50).astype(int)

        g['month']      = g['date'].dt.month
        g['fy_end']     = g['month'].isin([5,6,7]).astype(int)
        g['q2_results'] = g['month'].isin([1,2]).astype(int)
        g['cross_21_50']  = ema21/ema50-1;  g['cross_50_100'] = ema50/ema100-1
        g['cross_50_200'] = ema50/ema200-1; g['p_ema50'] = c/ema50-1; g['p_ema200'] = c/ema200-1
        macd_mu  = macd_raw.rolling(63, min_periods=30).mean()
        macd_std = macd_raw.rolling(63, min_periods=30).std()
        g['macd_norm'] = (macd_raw - macd_mu) / (macd_std + 1e-9)
        d21 = c.diff()
        g21 = d21.clip(lower=0).ewm(alpha=1/21, min_periods=21).mean()
        l21 = (-d21.clip(upper=0)).ewm(alpha=1/21, min_periods=21).mean().add(1e-9)
        g['rsi21'] = 100 - (100 / (1 + g21/l21))
        g['rsi14_zscore'] = (rsi14 - rsi14.rolling(63,min_periods=30).mean()) / \
                            (rsi14.rolling(63,min_periods=30).std() + 1e-9)
        lo14 = l.rolling(14, min_periods=10).min(); hi14 = h.rolling(14, min_periods=10).max()
        g['stoch_k'] = (c - lo14) / (hi14 - lo14 + 1e-9) * 100
        bb_ma  = c.rolling(20, min_periods=12).mean()
        bb_std = c.rolling(20, min_periods=12).std()
        g['bb_pct_20']   = (c - (bb_ma-2*bb_std)) / (4*bb_std+1e-9)
        g['bb_width_14'] = 4*c.rolling(14,min_periods=8).std() / (c.rolling(14,min_periods=8).mean()+1e-9)
        g['bb_width_20'] = 4*bb_std / (bb_ma+1e-9)
        for w in [5,10,21,63]: g[f'rvol{w}'] = log.rolling(w,min_periods=w//2).std()*np.sqrt(252)
        g['vol_regime'] = g['rvol21'].rolling(252, min_periods=60).rank(pct=True)
        g['atr14_pct']  = atr14 / (c+1e-9)
        g['atr_regime'] = atr14 / atr14.rolling(63, min_periods=30).mean()
        g['vol_r5']  = v / (v.rolling(5,  min_periods=3).mean()  + 1e-9)
        g['vol_r60'] = v / (v.rolling(60, min_periods=30).mean() + 1e-9)
        obv = (np.sign(log)*v).fillna(0).cumsum()
        g['obv_trend'] = (obv > obv.ewm(span=20, min_periods=10).mean()).astype(int)
        g['ret_2d']  = c.pct_change(2); g['ret_3d']  = c.pct_change(3)
        g['ret_21d'] = c.pct_change(21);g['ret_63d'] = c.pct_change(63)
        g['gap_pct'] = (o - c.shift(1)) / (c.shift(1)+1e-9)
        g['ret21_vadjusted'] = g['ret_21d'] / (g['rvol21']+1e-9)
        g['ret_skew21']  = log.rolling(21, min_periods=15).skew()
        g['ret_autocorr']= log.rolling(10, min_periods=8).apply(
            lambda x: x[:5].mean()-x[5:].mean() if len(x)==10 else np.nan, raw=True)
        g['range52_pos'] = (c - low52) / (high52 - low52 + 1e-9)
        g['dist_high52'] = c / (high52+1e-9) - 1
        g['bull_x_rsi']    = bull200 * rsi14
        g['bull_x_macd']   = bull200 * (macd_raw > macd_sig).astype(int)
        g['bull_x_volume'] = bull200 * g['vol_r60']
        g['easy_x_bull']   = (g['policy_rate'] < 5.0).astype(int) * bull200
        g['p_vs_nepse_21d'] = g['ret_21d'] - g['nepse_ret_21d'].fillna(0)
        g['p_vs_nepse_5d']  = g['ret_3d']  - g['nepse_ret_5d'].fillna(0)
        g['nepse_x_bull']   = g['nepse_bull'].fillna(0) * bull200
        g['car_zscore'] = (g['car'] - g['car'].rolling(252,min_periods=60).mean()) / \
                          (g['car'].rolling(252,min_periods=60).std()+1e-9)
        g['npl_zscore'] = (g['npl'] - g['npl'].rolling(252,min_periods=60).mean()) / \
                          (g['npl'].rolling(252,min_periods=60).std()+1e-9)
        g['npl_trend'] = g['npl'] - g['npl'].shift(252)
        g['car_trend'] = g['car'] - g['car'].shift(252)
        g['npl_x_bear'] = (g['npl'] > 3.0).astype(int) * (1 - bull200)
        g['car_x_bull'] = (g['car'] > 14.0).astype(int) * bull200
        g['_rsi14']    = rsi14;    g['_bull200']  = bull200
        g['_bull50']   = bull50;   g['_drawdown'] = drawdown
        g['_near_high']= (g['dist_high52'] > -0.05).astype(int)
        return g

    def _add_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for src, dst in [('ret_2d','5d'),('ret_3d','10d'),('ret_21d','21d'),('ret_63d','42d')]:
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
        df['xrank_ret_5d']   = df.groupby('date')['ret_21d'].rank(pct=True)
        df['xrank_ret_63d']  = df.groupby('date')['ret_63d'].rank(pct=True)
        df['xrank_rvol21']   = df.groupby('date')['rvol21'].rank(pct=True)
        df['breadth_x_bull'] = df['sec_breadth'] * df['_bull200']
        df['xrank_car']      = df.groupby('date')['car'].rank(pct=True)
        df['xrank_npl']      = df.groupby('date')['npl'].rank(pct=True)
        df['rel_npl']        = df['npl'] - df.groupby('date')['npl'].transform('mean')
        df.drop(columns=[c for c in df.columns if c.startswith('_')], inplace=True)
        return df

    # ── SIGNAL LOGIC ─────────────────────────────────────────────────────────
    def _score_to_signal(self, score: float) -> str:
        if score >= self.long_thresh: return 'LONG'
        if score >= self.hold_thresh: return 'HOLD'
        return 'SKIP'

    def _print_signals(self, signals: pd.DataFrame):
        print(f"\n{'='*72}")
        print(f"  NEPSE Signals  ·  {signals['date'].max().date()}")
        print(f"{'='*72}")
        print(f"  {'Bank':8s} {'Close':>7} {'P(dir)':>8} {'P(mom)':>8} "
              f"{'Mag%':>7} {'Ens':>7}  {'CAR':>6} {'NPL':>6}  Signal")
        print(f"  {'-'*70}")
        for _, r in signals.iterrows():
            icon = '▲' if r['signal']=='LONG' else '→' if r['signal']=='HOLD' else '▼'
            print(f"  {r['bank']:8s} {r['close']:7.0f} {r['prob_direction']:8.3f} "
                  f"{r['prob_momentum']:8.3f} {r['predicted_mag']:7.2f}% "
                  f"{r['ensemble_score']:7.3f}  {r['car']:6.1f} {r['npl']:6.2f}  "
                  f"{icon} {r['signal']}")
        print(f"{'='*72}\n")

    def _log(self, message: str):
        if self.verbose:
            print(message)


# ── DEMO — runs with real uploaded data ──────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Use the actual uploaded data to demonstrate the full inference path
    print("\n" + "="*65)
    print(" INFERENCE PIPELINE DEMO")
    print(" Using real NEPSE data (combinedBankData.csv + nepse_index.csv)")
    print("="*65)

    # ── Load raw data (simulates what your scraper would provide) ────────────
    bank_df = pd.read_csv('/mnt/user-data/uploads/combinedBankData.csv')
    bank_df.columns  = bank_df.columns.str.strip()
    bank_df['bank']  = bank_df['bank'].str.strip()
    bank_df['date']  = pd.to_datetime(bank_df['date'])
    bank_df = bank_df.sort_values(['bank','date'])

    # Simulate scraping: take last 300 rows per bank (what you'd get at runtime)
    ohlcv_scraped = (bank_df.groupby('bank')
                            .tail(300)
                            .reset_index(drop=True)
                            [['date','bank','open','high','low','close','volume','amount']])

    # NEPSE index
    nepse_raw = pd.read_csv('/mnt/user-data/uploads/nepse_index.csv')
    nepse_raw.columns = nepse_raw.columns.str.strip()
    nepse_raw = nepse_raw.rename(columns={'timestamp':'date','close':'nepse_close'})
    nepse_raw['date'] = pd.to_datetime(nepse_raw['date'])
    nepse_df  = nepse_raw[['date','nepse_close']].tail(400)

    # Fundamentals — latest known annual figures (from your CAR/NPL xlsx)
    # At inference time your backend reads these from the DB (FinancialHistory table)
    fundamentals = {
        'ADBL':  {'car': 13.4,  'npl': 3.44},
        'CZBIL': {'car': 13.1,  'npl': 4.07},
        'EBL':   {'car': 12.6,  'npl': 0.71},
        'GBIME': {'car': 12.5,  'npl': 4.37},
        'HBL':   {'car': 12.0,  'npl': 4.91},
        'KBL':   {'car': 12.3,  'npl': 4.95},
        'MBL':   {'car': 13.8,  'npl': 3.63},
        'NABIL': {'car': 12.4,  'npl': 3.85},
        'NICA':  {'car': 11.2,  'npl': 3.41},
        'NMB':   {'car': 12.8,  'npl': 3.27},
        'PCBL':  {'car': 11.9,  'npl': 4.67},
        'PRVU':  {'car': 13.6,  'npl': 4.78},
        'SANIMA':{'car': 13.7,  'npl': 1.73},
        'SBL':   {'car': 12.2,  'npl': 2.22},
        'SCB':   {'car': 17.2,  'npl': 2.14},
    }

    # ── Run inference ────────────────────────────────────────────────────────
    pipeline = NEPSEInferencePipeline('model_artifacts')
    signals  = pipeline.predict(
        ohlcv_df    = ohlcv_scraped,
        nepse_df    = nepse_df,
        fundamentals= fundamentals,
        policy_rate = 4.5,
        verbose     = True,
    )

    # Output can be written to DB (Prediction table) or returned via API
    signals.to_csv('latest_signals.csv', index=False)
    print(f"Signals saved to: latest_signals.csv")
