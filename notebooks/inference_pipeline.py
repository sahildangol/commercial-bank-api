"""
Inference pipeline for Nepal bank stocks (direction + magnitude).

This module is used by the FastAPI service. It expects the same inputs as the
existing API scraper (OHLCV for one or more banks + NEPSE index history), but
uses the new HistGB models + lighter feature engineering.

Output columns (one row per bank):
    bank | date | close | prob_direction | prob_momentum | predicted_mag |
    ensemble_score | signal | car | npl

Additional debug columns may be included when available (prob_up,
return_magnitude, confidence, etc.).
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MIN_ROWS_REQUIRED = 252
POLICY_RATE_FALLBACK = 4.5


class NEPSEInferencePipeline:
    """End-to-end inference pipeline for Nepal commercial bank signals."""

    def __init__(self, model_dir: str = "models", verbose: bool = True):
        self.model_dir = model_dir
        self.verbose = verbose
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        self._log(f"Loading model artifacts from: {self.model_dir}/")

        self.cls_model = joblib.load(os.path.join(self.model_dir, "direction_classifier.pkl"))
        self.reg_model = joblib.load(os.path.join(self.model_dir, "return_regressor.pkl"))
        self.le = joblib.load(os.path.join(self.model_dir, "label_encoder.pkl"))

        meta_path = os.path.join(self.model_dir, "model_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta.get("feature_cols") or self.meta.get("model_features", [])
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.bank_classes = self.meta.get("bank_classes", [])

        # Normalize meta for downstream consumers.
        self.meta.setdefault("model_features", list(self.feature_cols))
        self.meta.setdefault("n_features", len(self.feature_cols))
        self.meta.setdefault("train_end", self.meta.get("train_cutoff"))
        self.meta.setdefault("metrics", {})

        self._sector_history = self._load_csv("sector_ret_history.csv", date_cols=["date"])
        self._fundamentals = self._load_csv("fundamental_lookup.csv")
        if self._fundamentals is not None and "bank" in self._fundamentals.columns:
            self._fundamentals["bank"] = self._fundamentals["bank"].astype(str).str.strip().str.upper()
            self._fundamentals = self._fundamentals.set_index("bank")

        self._log(f"  Models loaded. Features: {len(self.feature_cols)}")
        self._log(f"  Threshold: {self.threshold:.3f}")
        if self.bank_classes:
            self._log(f"  Known banks: {', '.join(self.bank_classes)}")

    def predict(
        self,
        ohlcv_df: pd.DataFrame,
        nepse_df: pd.DataFrame,
        fundamentals: Dict[str, Dict[str, float]] | None,
        policy_rate: float = POLICY_RATE_FALLBACK,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run inference for one or more banks."""
        _ = policy_rate  # unused in this model family

        ohlcv = self._prepare_ohlcv(ohlcv_df)
        nepse = self._prepare_nepse(nepse_df)
        banks = sorted(ohlcv["bank"].unique())

        if verbose:
            print(f"\nRunning inference for {len(banks)} banks: {banks}")
            print(
                f"  OHLCV rows: {len(ohlcv):,} | "
                f"Date range: {ohlcv['date'].min().date()} to {ohlcv['date'].max().date()}"
            )

        sector_ret_live = self._compute_sector_ret_live(ohlcv)

        rows: list[dict[str, object]] = []
        for bank in banks:
            bank_df = ohlcv[ohlcv["bank"] == bank].sort_values("date")
            if bank_df.empty:
                continue

            last_date = bank_df["date"].iloc[-1]
            yesterday_date = bank_df["date"].iloc[-2] if len(bank_df) >= 2 else last_date

            car_value, npl_value = self._resolve_fundamentals(bank, fundamentals)
            sector_ret = self._lookup_sector_ret(yesterday_date, sector_ret_live)

            features = self._compute_features(
                bank=bank,
                ohlcv=bank_df,
                nepse=nepse,
                sector_ret=sector_ret,
                car=car_value,
                le=self.le,
            )

            nan_features = [
                k
                for k, v in features.items()
                if v is None or (isinstance(v, float) and np.isnan(v))
            ]
            if nan_features:
                self._log(
                    f"  WARNING: {bank} has NaN features {nan_features} "
                    f"(need {MIN_ROWS_REQUIRED} rows for full history)."
                )

            X = pd.DataFrame([features])
            for col in self.feature_cols:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[self.feature_cols]

            prob_up = float(self.cls_model.predict_proba(X)[0, 1])
            ret_mag = float(self.reg_model.predict(X)[0])

            direction = "UP" if prob_up > self.threshold else "DOWN"
            confidence = self._confidence_label(prob_up)
            signal_strength = self._signal_strength_label(ret_mag)

            close = float(bank_df["close"].iloc[-1])

            rows.append(
                {
                    "bank": bank,
                    "date": last_date,
                    "close": close,
                    "prob_direction": round(prob_up, 4),
                    "prob_momentum": round(prob_up, 4),
                    "predicted_mag": round(ret_mag * 100, 2),
                    "ensemble_score": round(prob_up, 4),
                    "signal": direction,
                    "car": car_value,
                    "npl": npl_value,
                    "direction": direction,
                    "prob_up": round(prob_up, 4),
                    "prob_down": round(1 - prob_up, 4),
                    "return_magnitude": round(ret_mag, 5),
                    "return_magnitude_pct": f"{ret_mag*100:+.3f}%",
                    "confidence": confidence,
                    "signal_strength": signal_strength,
                    "threshold_used": self.threshold,
                    "nan_features": nan_features,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "bank",
                    "date",
                    "close",
                    "prob_direction",
                    "prob_momentum",
                    "predicted_mag",
                    "ensemble_score",
                    "signal",
                    "car",
                    "npl",
                ]
            )

        signals = pd.DataFrame(rows).sort_values("prob_direction", ascending=False).reset_index(
            drop=True
        )

        if verbose:
            self._print_signals(signals)

        return signals

    def _prepare_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["date", "bank", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"ohlcv_df missing columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["bank"] = df["bank"].astype(str).str.strip().str.upper()

        # keep OHLC valid
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["open"] = df["open"].clip(lower=df["low"], upper=df["high"])

        if "amount" not in df.columns:
            df["amount"] = np.nan
        df = df.drop_duplicates(subset=["bank", "date"], keep="last")
        df = df.sort_values(["bank", "date"]).reset_index(drop=True)

        for bk, g in df.groupby("bank"):
            if len(g) < MIN_ROWS_REQUIRED:
                self._log(
                    f"  WARNING: {bk} has only {len(g)} rows "
                    f"(need >= {MIN_ROWS_REQUIRED})."
                )
        return df

    def _prepare_nepse(self, df: pd.DataFrame) -> pd.DataFrame:
        if "nepse_close" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"close": "nepse_close"})

        required = ["date", "nepse_close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"nepse_df missing columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["nepse_close"] = df["nepse_close"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)

        if "nepse_ret_1d" not in df.columns:
            df["nepse_ret_1d"] = df["nepse_close"].pct_change()

        if "nepse_bull" not in df.columns:
            ema200 = df["nepse_close"].ewm(span=200, min_periods=100).mean()
            df["nepse_bull"] = (df["nepse_close"] > ema200).astype(float)
            df["nepse_bull"] = df["nepse_bull"].fillna(0.0)

        return df

    def _compute_sector_ret_live(self, ohlcv: pd.DataFrame) -> float:
        returns: list[float] = []
        for _, g in ohlcv.groupby("bank"):
            g = g.sort_values("date")
            if len(g) >= 2:
                r = (g["close"].iloc[-1] / g["close"].iloc[-2]) - 1
                returns.append(float(r))
        if not returns:
            return 0.0
        return float(np.mean(returns))

    def _lookup_sector_ret(self, target_date: pd.Timestamp, fallback: float) -> float:
        if self._sector_history is None or "date" not in self._sector_history.columns:
            return fallback

        matches = self._sector_history[self._sector_history["date"] == target_date]
        if matches.empty:
            return fallback
        value = matches["sector_ret"].iloc[0]
        if pd.isna(value):
            return fallback
        return float(value)

    def _resolve_fundamentals(
        self,
        bank: str,
        fundamentals: Dict[str, Dict[str, float]] | None,
    ) -> tuple[float, float | None]:
        car_value = None
        npl_value = None

        if fundamentals and bank in fundamentals:
            car_value = fundamentals[bank].get("car")
            npl_value = fundamentals[bank].get("npl")

        if car_value is None and self._fundamentals is not None and bank in self._fundamentals.index:
            try:
                car_value = float(self._fundamentals.loc[bank, "car"])
            except Exception:
                car_value = None

        if npl_value is None and self._fundamentals is not None and bank in self._fundamentals.index:
            try:
                npl_value = float(self._fundamentals.loc[bank, "npl"])
            except Exception:
                npl_value = None

        if car_value is None or pd.isna(car_value):
            car_value = 13.0
            self._log(f"  WARNING: No CAR data for {bank}, using fallback {car_value}")

        return float(car_value), npl_value

    @staticmethod
    def _compute_features(
        bank: str,
        ohlcv: pd.DataFrame,
        nepse: pd.DataFrame,
        sector_ret: float,
        car: float,
        le,
    ) -> dict[str, float | int | None]:
        ohlcv = ohlcv.sort_values("date").copy()
        c = ohlcv["close"]
        ret = c.pct_change()

        ret_lag1 = float(ret.iloc[-2]) if len(ret) >= 2 else np.nan
        ret_3d_lag1 = float(ret.iloc[-4:-1].sum()) if len(ret) >= 4 else np.nan

        if len(c) >= 21:
            ma20 = c.rolling(20).mean().iloc[-2]
            std20 = c.rolling(20).std().iloc[-2]
            c_lag = c.iloc[-2]
            bb_pct_lag1 = float((c_lag - (ma20 - 2 * std20)) / (4 * std20 + 1e-9))
        else:
            bb_pct_lag1 = np.nan

        if len(ohlcv) >= 15:
            lo14 = ohlcv["low"].iloc[-15:-1].min()
            hi14 = ohlcv["high"].iloc[-15:-1].max()
            c_lag = c.iloc[-2]
            stoch_k_lag1 = float((c_lag - lo14) / (hi14 - lo14 + 1e-9) * 100)
        else:
            stoch_k_lag1 = np.nan

        if len(ohlcv) >= 15:
            hi = ohlcv["high"]
            lo = ohlcv["low"]
            cp = c.shift(1)
            tr = pd.concat([hi - lo, (hi - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-2]
            atr_ratio_lag1 = float(atr14 / (c.iloc[-2] + 1e-9))
        else:
            atr_ratio_lag1 = np.nan

        if len(ret) >= 22:
            vol_5 = float(ret.iloc[-6:-1].std())
            vol_21 = float(ret.iloc[-22:-1].std())
            vol_ratio_5_21 = vol_5 / (vol_21 + 1e-9)
        else:
            vol_ratio_5_21 = np.nan

        if len(c) >= 21:
            ma21 = float(c.iloc[-21:].mean())
            close_to_ma21 = float(c.iloc[-1] / ma21) - 1.0
        else:
            close_to_ma21 = np.nan

        window = min(252, len(c))
        lo252 = float(c.iloc[-window:].min())
        dist_52w_low = float(c.iloc[-1] / lo252) - 1.0 if lo252 > 0 else np.nan

        if len(ohlcv) >= 6:
            vol_today = float(ohlcv["volume"].iloc[-2])
            vol_ma21 = float(ohlcv["volume"].iloc[max(-23, -len(ohlcv)) : -1].mean())
            vol_surge_lag1 = vol_today / (vol_ma21 + 1e-9)
        else:
            vol_surge_lag1 = np.nan

        nepse = nepse.sort_values("date").copy()
        if "nepse_ret_1d" not in nepse.columns:
            nepse["nepse_ret_1d"] = nepse["nepse_close"].pct_change()

        nepse_ret_lag1 = float(nepse["nepse_ret_1d"].iloc[-2]) if len(nepse) >= 2 else np.nan
        nepse_bull = (
            float(nepse["nepse_bull"].iloc[-1])
            if "nepse_bull" in nepse.columns
            else 0.0
        )

        if hasattr(le, "classes_") and bank in le.classes_:
            bank_enc = int(le.transform([bank])[0])
        else:
            bank_enc = -1

        month = int(ohlcv["date"].iloc[-1].month)

        return {
            "close_to_ma21": close_to_ma21,
            "nepse_ret_lag1": nepse_ret_lag1,
            "month": month,
            "bb_pct_lag1": bb_pct_lag1,
            "dist_52w_low": dist_52w_low,
            "ret_3d_lag1": ret_3d_lag1,
            "stoch_k_lag1": stoch_k_lag1,
            "atr_ratio_lag1": atr_ratio_lag1,
            "vol_surge_lag1": vol_surge_lag1,
            "bank_enc": bank_enc,
            "nepse_bull": nepse_bull,
            "sector_ret_lag1": sector_ret,
            "car": car,
            "ret_lag1": ret_lag1,
            "vol_ratio_5_21": vol_ratio_5_21,
        }

    @staticmethod
    def _confidence_label(prob_up: float) -> str:
        margin = abs(prob_up - 0.5)
        if margin >= 0.12:
            return "HIGH"
        if margin >= 0.06:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _signal_strength_label(ret_mag: float) -> str:
        abs_ret = abs(ret_mag)
        if abs_ret >= 0.02:
            return "STRONG"
        if abs_ret >= 0.01:
            return "MODERATE"
        return "WEAK"

    def _print_signals(self, signals: pd.DataFrame) -> None:
        print(f"\n{'='*72}")
        print(f"  NEPSE Signals  -  {signals['date'].max().date()}")
        print(f"{'='*72}")
        print(
            f"  {'Bank':8s} {'Close':>7} {'P(up)':>7} {'Ret%':>7} "
            f"{'Conf':>6} {'Str':>6}  Signal"
        )
        print(f"  {'-'*70}")
        for _, r in signals.iterrows():
            print(
                f"  {r['bank']:8s} {r['close']:7.0f} {r['prob_direction']:7.3f} "
                f"{r['predicted_mag']:7.2f}% {r['confidence']:>6} {r['signal_strength']:>6}  "
                f"{r['signal']}"
            )
        print(f"{'='*72}\n")

    def _load_csv(self, filename: str, date_cols: list[str] | None = None) -> pd.DataFrame | None:
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, parse_dates=date_cols or [])
        except Exception:
            return None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
