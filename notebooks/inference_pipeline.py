"""
Jupyter-friendly TFT inference pipeline for NEPSE OHLCV data.

This module serves two purposes:
1. It powers the FastAPI inference service through the existing
   ``NEPSEInferencePipeline`` interface.
2. It can be pasted into or imported from a notebook and used directly via
   ``run_inference(...)``.

The pipeline recreates the model's inference dataset from the checkpoint's
saved dataset parameters, so we do not need access to the original training
dataset at prediction time.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

POLICY_RATE_FALLBACK = 4.5
DEFAULT_ARTIFACT_DIR = Path("models")
DEFAULT_WEEKMASK = "Sun Mon Tue Wed Thu"
DEFAULT_CAR = 12.0
DEFAULT_NPL = 2.0
DEFAULT_STATE_DICT_NAME = "tft_model.pth"
DEFAULT_CKPT_NAME = "tft_best_model.ckpt"
NOTEBOOK_TARGET = "t5"
NOTEBOOK_KNOWN_REALS = ["time_idx"]
NOTEBOOK_UNKNOWN_REALS = ["close_x", "volume_x", "t1", "t3", "t5"]
NOTEBOOK_ENCODER_LENGTH = 30
NOTEBOOK_PREDICTION_LENGTH = 5
NOTEBOOK_MODEL_KWARGS = {
    "learning_rate": 1e-3,
    "hidden_size": 32,
    "attention_head_size": 4,
    "dropout": 0.2,
    "hidden_continuous_size": 16,
}
HISTORY_WINDOW_DAYS = 5
SUMMARY_HORIZON_DAYS = 5
API_SIGNAL_COLUMNS = [
    "bank",
    "date",
    "close",
    "prob_direction",
    "prob_momentum",
    "predicted_mag",
    "model_score",
    "signal",
    "direction",
    "prob_up",
    "prob_down",
    "return_magnitude",
    "return_magnitude_pct",
    "confidence",
    "signal_strength",
    "threshold_used",
    "nan_features",
    "car",
    "npl",
    "forecast_next_5d",
    "timeline_10d",
]

try:
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    torch = None
    TimeSeriesDataSet = None
    TemporalFusionTransformer = None
    GroupNormalizer = None
    MAE = None
    _IMPORT_ERROR: ModuleNotFoundError | None = exc
else:  # pragma: no cover - simple assignment
    _IMPORT_ERROR = None


@dataclass(slots=True)
class ForecastArtifacts:
    forecast: pd.DataFrame
    processed_history: pd.DataFrame
    target_interpretation: str
    device: str
    max_encoder_length: int
    max_prediction_length: int
    last_observed_date: pd.Timestamp
    last_observed_close: float
    used_nepse_fallback: bool
    bank: str
    car: float
    npl: float


class NEPSEInferencePipeline:
    """Production-oriented inference wrapper around a saved TFT checkpoint."""

    def __init__(
        self,
        model_dir: str = str(DEFAULT_ARTIFACT_DIR),
        verbose: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        self.artifact_dir = Path(model_dir)
        self.verbose = verbose
        self.threshold = 0.5
        self.device = self._resolve_device()
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.meta = self._load_meta()
        self.meta.setdefault("model_features", self._combine_feature_columns(self.meta))
        self.meta.setdefault("n_features", len(self.meta["model_features"]))

        self._model = None
        self._state_dict: dict[str, Any] | None = None
        self._dataset_parameters: dict[str, Any] | None = None
        self._uses_state_dict = self.checkpoint_path.suffix.lower() == ".pth"

        default_target = NOTEBOOK_TARGET if self._uses_state_dict else "target_logret"
        default_encoder = NOTEBOOK_ENCODER_LENGTH if self._uses_state_dict else 30
        default_pred = NOTEBOOK_PREDICTION_LENGTH if self._uses_state_dict else 7
        default_known = NOTEBOOK_KNOWN_REALS if self._uses_state_dict else []
        default_unknown = NOTEBOOK_UNKNOWN_REALS if self._uses_state_dict else []

        self.target_name = str(self.meta.get("target", default_target))
        self.max_encoder_length = int(self.meta.get("encoder_len", default_encoder))
        self.max_prediction_length = int(self.meta.get("pred_len", default_pred))
        self.known_reals = list(self.meta.get("known_reals", default_known))
        self.unknown_reals = list(self.meta.get("unknown_reals", default_unknown))

        raw_banks = self.meta.get("banks") or self.meta.get("bank_classes") or self.meta.get("active_banks") or []
        self.supported_banks = [str(bank).strip().upper() for bank in raw_banks if str(bank).strip()]

    def predict(
        self,
        ohlcv_df: pd.DataFrame,
        nepse_df: pd.DataFrame | None = None,
        fundamentals: dict[str, dict[str, float | None]] | None = None,
        policy_rate: float = POLICY_RATE_FALLBACK,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Return API-friendly summary rows for each bank in the input frame."""

        self._ensure_runtime_dependencies()
        self._load_model()

        ohlcv = self._prepare_ohlcv(ohlcv_df)
        nepse, used_nepse_fallback = self._prepare_nepse(nepse_df, ohlcv)
        requested_banks = sorted(ohlcv["bank"].unique())

        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        for bank in requested_banks:
            if self.supported_banks and bank not in self.supported_banks:
                skipped.append(bank)
                continue

            try:
                rows.append(
                    self._build_signal_row(
                        ohlcv=ohlcv,
                        bank=bank,
                        nepse=nepse,
                        fundamentals=fundamentals,
                        policy_rate=policy_rate,
                        used_nepse_fallback=used_nepse_fallback,
                    )
                )
            except Exception as exc:
                skipped.append(f"{bank} ({exc})")
                continue

        if skipped and verbose:
            self._log(f"Skipping banks that could not be scored: {', '.join(sorted(skipped))}")

        if not rows:
            return pd.DataFrame(columns=API_SIGNAL_COLUMNS)

        signals = pd.DataFrame(rows).sort_values(["model_score", "predicted_mag"], ascending=[False, False])
        return signals.reset_index(drop=True)

    def _build_signal_row(
        self,
        ohlcv: pd.DataFrame,
        bank: str,
        nepse: pd.DataFrame,
        fundamentals: dict[str, dict[str, float | None]] | None,
        policy_rate: float,
        used_nepse_fallback: bool,
    ) -> dict[str, Any]:
        bank_history = ohlcv[ohlcv["bank"] == bank].copy()
        car_value, npl_value = self._resolve_fundamentals(bank, fundamentals)
        artifacts = self._forecast_single_bank(
            ohlcv_df=bank_history,
            full_ohlcv_df=ohlcv,
            bank=bank,
            nepse_df=nepse,
            car=car_value,
            npl=npl_value,
            policy_rate=policy_rate,
            batch_size=64,
            used_nepse_fallback=used_nepse_fallback,
        )
        forecast = artifacts.forecast
        forecast_window = forecast.head(min(SUMMARY_HORIZON_DAYS, len(forecast))).copy()
        summary_row = forecast_window.iloc[-1]
        terminal_row = forecast.iloc[-1]
        direction_return = float(summary_row["cumulative_return"])
        momentum_return = float(forecast_window["predicted_return"].mean())

        prob_direction = self._confidence_from_return(direction_return)
        prob_momentum = self._confidence_from_return(momentum_return)
        model_score = round((0.6 * prob_direction) + (0.4 * prob_momentum), 4)
        signal = "UP" if direction_return >= 0 else "DOWN"
        forecast_next_5d = [
            {
                "horizon_day": int(row["horizon_step"]),
                "forecast_date": pd.Timestamp(row["forecast_date"]).to_pydatetime(),
                "predicted_close": round(float(row["predicted_close"]), 6),
                "predicted_return": round(float(row["predicted_return"]), 6),
                "cumulative_return": round(float(row["cumulative_return"]), 6),
            }
            for _, row in forecast_window.iterrows()
        ]
        history_window = artifacts.processed_history.tail(HISTORY_WINDOW_DAYS).copy()
        history_points = [
            {
                "date": pd.Timestamp(row["date"]).to_pydatetime(),
                "point_type": "history",
                "horizon_day": None,
                "open": round(float(row["open"]), 6),
                "high": round(float(row["high"]), 6),
                "low": round(float(row["low"]), 6),
                "close": round(float(row["close"]), 6),
                "volume": round(float(row["volume"]), 6),
                "predicted_return": None,
                "cumulative_return": None,
            }
            for _, row in history_window.iterrows()
        ]
        forecast_points = [
            {
                "date": pd.Timestamp(row["forecast_date"]).to_pydatetime(),
                "point_type": "forecast",
                "horizon_day": int(row["horizon_step"]),
                "open": None,
                "high": None,
                "low": None,
                "close": round(float(row["predicted_close"]), 6),
                "volume": None,
                "predicted_return": round(float(row["predicted_return"]), 6),
                "cumulative_return": round(float(row["cumulative_return"]), 6),
            }
            for _, row in forecast_window.iterrows()
        ]
        timeline_10d = [*history_points, *forecast_points]

        return {
            "bank": bank,
            "date": artifacts.last_observed_date,
            "close": round(artifacts.last_observed_close, 6),
            "prob_direction": round(prob_direction, 4),
            "prob_momentum": round(prob_momentum, 4),
            "predicted_mag": round(direction_return * 100.0, 4),
            "model_score": model_score,
            "signal": signal,
            "direction": signal,
            "prob_up": round(prob_direction, 4),
            "prob_down": round(1.0 - prob_direction, 4),
            "return_magnitude": round(direction_return, 6),
            "return_magnitude_pct": f"{direction_return * 100:+.3f}%",
            "confidence": self._confidence_label(prob_direction),
            "signal_strength": self._signal_strength_label(direction_return),
            "threshold_used": self.threshold,
            "nan_features": None,
            "car": round(car_value, 4),
            "npl": round(npl_value, 4),
            "forecast_start": forecast_window["forecast_date"].iloc[0],
            "forecast_end": forecast_window["forecast_date"].iloc[-1],
            "forecast_close_5d": round(float(summary_row["predicted_close"]), 6),
            "forecast_close_terminal": round(float(terminal_row["predicted_close"]), 6),
            "forecast_next_5d": forecast_next_5d,
            "timeline_10d": timeline_10d,
            "target_interpretation": artifacts.target_interpretation,
            "used_nepse_fallback": used_nepse_fallback,
        }

    def forecast_dataframe(
        self,
        ohlcv_df: pd.DataFrame,
        bank: str,
        nepse_df: pd.DataFrame | None = None,
        car: float = DEFAULT_CAR,
        npl: float = DEFAULT_NPL,
        policy_rate: float = POLICY_RATE_FALLBACK,
        batch_size: int = 64,
        plot: bool = False,
    ) -> pd.DataFrame:
        """Return the full horizon forecast DataFrame for a single bank."""

        self._ensure_runtime_dependencies()
        self._load_model()

        bank = str(bank).strip().upper()
        if self.supported_banks and bank not in self.supported_banks:
            raise ValueError(
                f"Bank '{bank}' is not supported by this TFT checkpoint. "
                f"Supported banks: {', '.join(self.supported_banks)}"
            )
        ohlcv = self._prepare_ohlcv(ohlcv_df)
        bank_history = ohlcv[ohlcv["bank"] == bank].copy()
        if bank_history.empty:
            raise ValueError(f"No OHLCV rows found for bank '{bank}'.")

        nepse, used_nepse_fallback = self._prepare_nepse(nepse_df, ohlcv)
        artifacts = self._forecast_single_bank(
            ohlcv_df=bank_history,
            full_ohlcv_df=ohlcv,
            bank=bank,
            nepse_df=nepse,
            car=float(car),
            npl=float(npl),
            policy_rate=policy_rate,
            batch_size=batch_size,
            used_nepse_fallback=used_nepse_fallback,
        )
        if plot:
            self.plot_forecast(
                processed_history=artifacts.processed_history,
                forecast_df=artifacts.forecast,
                bank=bank,
            )
        return artifacts.forecast.copy()

    def predict_from_csv(
        self,
        csv_path: str,
        checkpoint_path: str | None = None,
        bank: str | None = None,
        nepse_csv_path: str | None = None,
        car: float = DEFAULT_CAR,
        npl: float = DEFAULT_NPL,
        policy_rate: float = POLICY_RATE_FALLBACK,
        batch_size: int = 64,
        plot: bool = False,
    ) -> pd.DataFrame:
        """Notebook-friendly CSV entrypoint."""

        if checkpoint_path is not None and Path(checkpoint_path) != self.checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
            self._model = None
            self._state_dict = None
            self._dataset_parameters = None
            self._uses_state_dict = self.checkpoint_path.suffix.lower() == ".pth"

        ohlcv_df = pd.read_csv(csv_path)
        inferred_bank = bank or self._infer_bank_symbol(ohlcv_df, csv_path)
        if "bank" not in ohlcv_df.columns:
            ohlcv_df["bank"] = inferred_bank

        nepse_df = pd.read_csv(nepse_csv_path) if nepse_csv_path else None
        return self.forecast_dataframe(
            ohlcv_df=ohlcv_df,
            bank=inferred_bank,
            nepse_df=nepse_df,
            car=car,
            npl=npl,
            policy_rate=policy_rate,
            batch_size=batch_size,
            plot=plot,
        )

    def plot_forecast(
        self,
        processed_history: pd.DataFrame,
        forecast_df: pd.DataFrame,
        bank: str,
        history_window: int = 60,
    ) -> None:
        """Plot recent actual closes against the forecast tail."""

        import matplotlib.pyplot as plt

        history_tail = processed_history.tail(history_window).copy()
        history_x = history_tail["date"].tolist()
        history_y = history_tail["close"].tolist()

        forecast_x = [history_tail["date"].iloc[-1], *forecast_df["forecast_date"].tolist()]
        forecast_y = [history_tail["close"].iloc[-1], *forecast_df["predicted_close"].tolist()]

        plt.figure(figsize=(12, 6))
        plt.plot(history_x, history_y, label="Actual Close", linewidth=2.0)
        plt.plot(forecast_x, forecast_y, label="Forecast Close", linewidth=2.0, linestyle="--")
        plt.title(f"TFT Forecast for {bank}")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _forecast_single_bank(
        self,
        ohlcv_df: pd.DataFrame,
        full_ohlcv_df: pd.DataFrame,
        bank: str,
        nepse_df: pd.DataFrame,
        car: float,
        npl: float,
        policy_rate: float,
        batch_size: int,
        used_nepse_fallback: bool = False,
    ) -> ForecastArtifacts:
        self._load_model()

        if self._uses_state_dict:
            history = self._build_notebook_feature_history(ohlcv_df=ohlcv_df, bank=bank)
            inference_frame = self._build_inference_frame(history, bank=bank, policy_rate=policy_rate)

            context_history = self._build_notebook_feature_history(ohlcv_df=full_ohlcv_df, bank=None)
            context_history = context_history.groupby("bank", group_keys=False).filter(
                lambda frame: len(frame) >= (self.max_encoder_length + self.max_prediction_length)
            )
            if context_history.empty or bank not in set(context_history["bank"].unique()):
                context_history = history.copy()

            training_dataset = self._build_state_dict_training_dataset(context_history)
            prediction_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                inference_frame,
                predict=True,
                stop_randomization=True,
            )
            dataloader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=0,
            )
            self._ensure_state_dict_model(training_dataset)
            predictions = self._model.predict(
                dataloader,
                mode="prediction",
                trainer_kwargs=self._trainer_kwargs(),
            )
        else:
            history = self._build_feature_history(
                ohlcv_df=ohlcv_df,
                bank=bank,
                nepse_df=nepse_df,
                car=car,
                npl=npl,
                policy_rate=policy_rate,
            )
            inference_frame = self._build_inference_frame(history, bank=bank, policy_rate=policy_rate)
            dataset = TimeSeriesDataSet.from_parameters(
                self._dataset_parameters,
                inference_frame,
                predict=True,
                stop_randomization=True,
            )
            dataloader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
            predictions = self._model.predict(
                dataloader,
                mode="prediction",
                trainer_kwargs=self._trainer_kwargs(),
            )

        raw_vector = self._flatten_predictions(predictions, self.max_prediction_length)
        target_interpretation = self._infer_target_interpretation(raw_vector)
        forecast = self._build_forecast_dataframe(
            history=history,
            raw_predictions=raw_vector,
            target_interpretation=target_interpretation,
        )

        return ForecastArtifacts(
            forecast=forecast,
            processed_history=history,
            target_interpretation=target_interpretation,
            device=str(self.device),
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            last_observed_date=pd.Timestamp(history["date"].iloc[-1]),
            last_observed_close=float(history["close"].iloc[-1]),
            used_nepse_fallback=used_nepse_fallback,
            bank=bank,
            car=float(car),
            npl=float(npl),
        )

    def _build_feature_history(
        self,
        ohlcv_df: pd.DataFrame,
        bank: str,
        nepse_df: pd.DataFrame,
        car: float,
        npl: float,
        policy_rate: float,
    ) -> pd.DataFrame:
        if self._uses_state_dict:
            return self._build_notebook_feature_history(ohlcv_df=ohlcv_df, bank=bank)

        history = ohlcv_df.sort_values("date").copy()
        history["bank"] = bank
        history["group_id"] = bank

        history["month"] = history["date"].dt.month.astype(int)
        history["day"] = history["date"].dt.day.astype(int)
        history["weekday"] = history["date"].dt.weekday.astype(int)
        history["day_of_week"] = history["weekday"]
        history["is_month_end"] = history["date"].dt.is_month_end.astype(int)
        history["is_quarter_end"] = history["date"].dt.is_quarter_end.astype(int)
        history["fy_end"] = history["month"].isin([5, 6, 7]).astype(int)
        history["policy_rate"] = float(policy_rate)
        history["is_gap"] = history["date"].diff().dt.days.fillna(1).gt(1).astype(int)

        history["log_return"] = np.log(history["close"] / history["close"].shift(1))
        history["target_logret"] = history["log_return"]
        history["ma_7"] = history["close"].rolling(7, min_periods=3).mean()
        history["ma_14"] = history["close"].rolling(14, min_periods=7).mean()
        history["rolling_volatility"] = (
            history["log_return"].rolling(14, min_periods=7).std() * np.sqrt(252.0)
        )
        history["volume_ma_7"] = history["volume"].rolling(7, min_periods=3).mean()

        ema50 = history["close"].ewm(span=50, adjust=False, min_periods=10).mean()
        ema200 = history["close"].ewm(span=200, adjust=False, min_periods=20).mean()

        history["momentum_20d"] = history["close"].pct_change(20)
        history["volatility_20d"] = (
            history["log_return"].rolling(20, min_periods=10).std() * np.sqrt(252.0)
        )
        volume_mean = history["volume"].rolling(20, min_periods=10).mean()
        volume_std = history["volume"].rolling(20, min_periods=10).std().replace(0.0, np.nan)
        history["volume_z"] = (history["volume"] - volume_mean) / volume_std
        history["ret_5d"] = history["close"].pct_change(5)
        history["ret_21d"] = history["close"].pct_change(21)
        history["risk_stability"] = history["ret_21d"] / (
            history["volatility_20d"].abs() + 1e-6
        )
        history["p_ema50"] = (history["close"] / ema50) - 1.0
        history["p_ema200"] = (history["close"] / ema200) - 1.0

        history["car"] = float(car)
        history["npl"] = float(npl)
        npl_mean = history["npl"].rolling(63, min_periods=5).mean()
        npl_std = history["npl"].rolling(63, min_periods=5).std().replace(0.0, np.nan)
        history["npl_zscore"] = ((history["npl"] - npl_mean) / npl_std).fillna(0.0)

        nepse = nepse_df.copy()
        merge_columns = ["date", "nepse_close", "nepse_bull", "nepse_ret_5d", "nepse_ret_21d"]
        history = history.merge(nepse[merge_columns], on="date", how="left")
        history["nepse_close"] = history["nepse_close"].ffill().bfill()
        history["nepse_bull"] = history["nepse_bull"].ffill().bfill().fillna(0.0)
        history["nepse_ret_5d"] = history["nepse_ret_5d"].ffill().bfill()
        history["nepse_ret_21d"] = history["nepse_ret_21d"].ffill().bfill()
        history["alpha_21d"] = history["ret_21d"] - history["nepse_ret_21d"]

        history = history.replace([np.inf, -np.inf], np.nan)

        # Drop rows after feature creation so the model only sees complete history.
        history = history.dropna(
            subset=[
                "target_logret",
                "log_return",
                "momentum_20d",
                "ret_5d",
                "ret_21d",
                "nepse_ret_5d",
                "nepse_ret_21d",
            ]
        ).reset_index(drop=True)

        numeric_fill_columns = [
            "ma_7",
            "ma_14",
            "rolling_volatility",
            "volume_ma_7",
            "volatility_20d",
            "volume_z",
            "risk_stability",
            "p_ema50",
            "p_ema200",
            "alpha_21d",
        ]
        history[numeric_fill_columns] = history[numeric_fill_columns].fillna(0.0)
        history["time_idx"] = np.arange(len(history), dtype=np.int64)

        minimum_rows = self.max_encoder_length
        if len(history) < minimum_rows:
            raise ValueError(
                f"{bank} has only {len(history)} usable rows after feature engineering. "
                f"Need at least {minimum_rows} rows for encoder inference."
            )

        return history

    def _build_notebook_feature_history(
        self,
        ohlcv_df: pd.DataFrame,
        bank: str | None = None,
    ) -> pd.DataFrame:
        frame = ohlcv_df.copy()
        if bank is not None:
            target_bank = str(bank).strip().upper()
            frame = frame[frame["bank"].astype(str).str.strip().str.upper() == target_bank]
        if frame.empty:
            raise ValueError(f"No OHLCV rows found for bank '{bank}'.")

        frame = frame.sort_values(["bank", "date"]).copy()
        frame["bank"] = frame["bank"].astype(str).str.strip().str.upper()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame = frame.dropna(subset=["date", "bank", "close"])
        if frame.empty:
            raise ValueError("No valid OHLCV rows available after numeric coercion.")

        groups: list[pd.DataFrame] = []
        for bank_name, group in frame.groupby("bank", sort=False):
            history = group.sort_values("date").copy()
            history["close_x"] = history["close"].astype(float)
            history["volume_x"] = history["volume"].fillna(0.0).astype(float)
            history["t1"] = (history["close_x"].shift(-1) - history["close_x"]) / history["close_x"]
            history["t3"] = (history["close_x"].shift(-3) - history["close_x"]) / history["close_x"]
            history["t5"] = (history["close_x"].shift(-5) - history["close_x"]) / history["close_x"]
            history[["t1", "t3", "t5"]] = (
                history[["t1", "t3", "t5"]]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            history["time_idx"] = np.arange(len(history), dtype=np.int64)
            groups.append(history)

        combined = pd.concat(groups, ignore_index=True)
        minimum_rows = self.max_encoder_length
        if bank is not None and len(combined) < minimum_rows:
            raise ValueError(
                f"{bank} has only {len(combined)} usable rows after feature engineering. "
                f"Need at least {minimum_rows} rows for encoder inference."
            )
        return combined.reset_index(drop=True)

    def _build_state_dict_training_dataset(self, history: pd.DataFrame) -> Any:
        return TimeSeriesDataSet(
            history,
            time_idx="time_idx",
            target=self.target_name,
            group_ids=["bank"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=[
                column for column in NOTEBOOK_UNKNOWN_REALS if column in history.columns
            ],
            time_varying_known_reals=[
                column for column in NOTEBOOK_KNOWN_REALS if column in history.columns
            ],
            target_normalizer=GroupNormalizer(groups=["bank"]),
        )

    def _ensure_state_dict_model(self, training_dataset: Any) -> None:
        if self._model is not None:
            return
        if self._state_dict is None:
            raise RuntimeError("State dict is not loaded for AutoTFT inference.")

        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **NOTEBOOK_MODEL_KWARGS,
            loss=MAE(),
        )
        try:
            missing, unexpected = model.load_state_dict(self._state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Unable to load state dict model from {self.checkpoint_path}: {exc}"
            ) from exc

        if missing or unexpected:
            self._log(
                "Loaded state dict with non-critical differences: "
                f"missing={missing}, unexpected={unexpected}"
            )

        model.to(self.device)
        model.eval()
        self._model = model

    def _build_inference_frame(
        self,
        history: pd.DataFrame,
        bank: str,
        policy_rate: float,
    ) -> pd.DataFrame:
        if self._uses_state_dict:
            future_dates = self._future_dates(
                last_date=pd.Timestamp(history["date"].iloc[-1]),
                steps=self.max_prediction_length,
            )
            future = pd.DataFrame({"date": future_dates})
            last_row = history.iloc[-1]

            future["bank"] = bank
            future["open"] = float(last_row["close"])
            future["high"] = float(last_row["close"])
            future["low"] = float(last_row["close"])
            future["close"] = float(last_row["close"])
            future["volume"] = float(last_row["volume"])
            future["close_x"] = float(last_row["close_x"])
            future["volume_x"] = float(last_row["volume_x"])
            future["t1"] = float(last_row["t1"])
            future["t3"] = float(last_row["t3"])
            future["t5"] = float(last_row["t5"])
            future["time_idx"] = np.arange(
                int(history["time_idx"].iloc[-1]) + 1,
                int(history["time_idx"].iloc[-1]) + 1 + self.max_prediction_length,
                dtype=np.int64,
            )

            inference_frame = pd.concat([history, future], ignore_index=True, sort=False)
            required_columns = self._required_dataset_columns()
            for column in required_columns:
                if column not in inference_frame.columns:
                    if column == "bank":
                        inference_frame[column] = bank
                    else:
                        inference_frame[column] = 0.0
            return inference_frame

        future_dates = self._future_dates(
            last_date=pd.Timestamp(history["date"].iloc[-1]),
            steps=self.max_prediction_length,
        )
        future = pd.DataFrame({"date": future_dates})
        last_row = history.iloc[-1]

        future["bank"] = bank
        future["group_id"] = bank
        future["open"] = float(last_row["close"])
        future["high"] = float(last_row["close"])
        future["low"] = float(last_row["close"])
        future["close"] = float(last_row["close"])
        future["volume"] = float(last_row["volume"])

        future["month"] = future["date"].dt.month.astype(int)
        future["day"] = future["date"].dt.day.astype(int)
        future["weekday"] = future["date"].dt.weekday.astype(int)
        future["day_of_week"] = future["weekday"]
        future["is_month_end"] = future["date"].dt.is_month_end.astype(int)
        future["is_quarter_end"] = future["date"].dt.is_quarter_end.astype(int)
        future["fy_end"] = future["month"].isin([5, 6, 7]).astype(int)
        future["policy_rate"] = float(policy_rate)
        future["is_gap"] = 0

        fill_forward_columns = [
            "log_return",
            "target_logret",
            "ma_7",
            "ma_14",
            "rolling_volatility",
            "volume_ma_7",
            "momentum_20d",
            "volatility_20d",
            "volume_z",
            "risk_stability",
            "ret_5d",
            "ret_21d",
            "p_ema50",
            "p_ema200",
            "car",
            "npl",
            "npl_zscore",
            "nepse_close",
            "nepse_bull",
            "nepse_ret_5d",
            "nepse_ret_21d",
            "alpha_21d",
        ]
        for column in fill_forward_columns:
            future[column] = float(last_row[column]) if column in last_row.index else 0.0

        # Prediction mode still requires a finite target placeholder.
        future["target_logret"] = float(last_row["target_logret"])
        future["time_idx"] = np.arange(
            int(history["time_idx"].iloc[-1]) + 1,
            int(history["time_idx"].iloc[-1]) + 1 + self.max_prediction_length,
            dtype=np.int64,
        )

        inference_frame = pd.concat([history, future], ignore_index=True, sort=False)
        required_columns = self._required_dataset_columns()
        for column in required_columns:
            if column not in inference_frame.columns:
                if column in {"bank", "group_id"}:
                    inference_frame[column] = bank
                else:
                    inference_frame[column] = 0.0
        return inference_frame

    def _build_forecast_dataframe(
        self,
        history: pd.DataFrame,
        raw_predictions: np.ndarray,
        target_interpretation: str,
    ) -> pd.DataFrame:
        last_close = float(history["close"].iloc[-1])
        future_dates = self._future_dates(
            last_date=pd.Timestamp(history["date"].iloc[-1]),
            steps=len(raw_predictions),
        )

        rows: list[dict[str, float | int | pd.Timestamp | str]] = []
        running_close = last_close

        for step, (forecast_date, raw_value) in enumerate(zip(future_dates, raw_predictions), start=1):
            predicted_target = float(raw_value)

            if target_interpretation == "log_return":
                step_return = float(np.expm1(predicted_target))
                running_close = max(last_close * float(np.exp(np.sum(raw_predictions[:step]))), 1e-6)
            elif target_interpretation == "simple_return":
                step_return = float(np.clip(predicted_target, -0.95, 10.0))
                running_close = max(running_close * (1.0 + step_return), 1e-6)
            elif target_interpretation == "log_price":
                running_close = max(float(np.exp(predicted_target)), 1e-6)
                step_return = (running_close / float(rows[-1]["predicted_close"])) - 1.0 if rows else (
                    (running_close / last_close) - 1.0
                )
            else:
                running_close = max(predicted_target, 1e-6)
                step_return = (running_close / float(rows[-1]["predicted_close"])) - 1.0 if rows else (
                    (running_close / last_close) - 1.0
                )

            cumulative_return = (running_close / last_close) - 1.0
            rows.append(
                {
                    "horizon_step": step,
                    "forecast_date": forecast_date,
                    "predicted_target": predicted_target,
                    "predicted_return": float(step_return),
                    "predicted_close": float(running_close),
                    "cumulative_return": float(cumulative_return),
                    "target_interpretation": target_interpretation,
                }
            )

        return pd.DataFrame(rows)

    def _required_dataset_columns(self) -> list[str]:
        required = {self.target_name, "time_idx", "bank"}
        required.update(self.known_reals)
        required.update(self.unknown_reals)
        if self._dataset_parameters is not None:
            required.update(self._dataset_parameters.get("group_ids", []))
            required.update(self._dataset_parameters.get("static_categoricals", []))
        return sorted(required)

    def _infer_target_interpretation(self, raw_predictions: np.ndarray) -> str:
        target_name = str(self.target_name).lower()
        if target_name in {"t1", "t3", "t5"}:
            return "simple_return"
        if "logret" in target_name and float(np.nanmin(raw_predictions)) < 0.0:
            return "log_return"
        if "logret" in target_name:
            return "simple_return"
        if "log" in target_name:
            return "log_price"
        if "close" in target_name or "price" in target_name:
            return "price_level"
        if float(np.nanmax(np.abs(raw_predictions))) <= 0.5:
            return "simple_return"
        return "price_level"

    def _flatten_predictions(self, output: Any, expected_length: int) -> np.ndarray:
        array = self._to_numpy(output)
        if array.ndim == 0:
            raise ValueError("Model returned a scalar prediction; expected a horizon vector.")

        if array.ndim == 1:
            flattened = array
        elif array.ndim == 2:
            flattened = array[0]
        else:
            candidate = array[0]
            while candidate.ndim > 1:
                candidate = candidate[..., 0]
            flattened = candidate

        flattened = np.asarray(flattened, dtype=float).reshape(-1)
        if len(flattened) < expected_length:
            raise ValueError(
                f"Model returned only {len(flattened)} forecast steps; expected {expected_length}."
            )
        return flattened[:expected_length]

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "output"):
            return self._to_numpy(value.output)
        if isinstance(value, dict):
            for key in ("prediction", "predictions", "output"):
                if key in value:
                    return self._to_numpy(value[key])
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Model returned an empty prediction container.")
            return self._to_numpy(value[0])
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _prepare_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"OHLCV data is missing required columns: {missing}")

        ohlcv = df.copy()
        ohlcv.columns = [str(column).strip() for column in ohlcv.columns]
        ohlcv["date"] = pd.to_datetime(ohlcv["date"], utc=False).dt.tz_localize(None)
        if "bank" not in ohlcv.columns:
            ohlcv["bank"] = "ASSET"

        ohlcv["bank"] = ohlcv["bank"].astype(str).str.strip().str.upper()
        for column in ("open", "high", "low", "close", "volume"):
            ohlcv[column] = pd.to_numeric(ohlcv[column], errors="coerce")

        ohlcv = ohlcv.dropna(subset=["date", "close"])
        ohlcv = ohlcv.sort_values(["bank", "date"]).drop_duplicates(
            subset=["bank", "date"], keep="last"
        )

        previous_close = ohlcv.groupby("bank")["close"].shift(1)
        ohlcv["open"] = ohlcv["open"].fillna(previous_close).fillna(ohlcv["close"])
        ohlcv["high"] = ohlcv[["high", "open", "close"]].max(axis=1)
        ohlcv["low"] = ohlcv[["low", "open", "close"]].min(axis=1)
        ohlcv["volume"] = ohlcv["volume"].fillna(0.0)
        return ohlcv.reset_index(drop=True)

    def _prepare_nepse(
        self,
        nepse_df: pd.DataFrame | None,
        ohlcv_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, bool]:
        used_fallback = False

        if nepse_df is None or nepse_df.empty:
            used_fallback = True
            nepse = (
                ohlcv_df.groupby("date", as_index=False)["close"]
                .mean()
                .rename(columns={"close": "nepse_close"})
            )
        else:
            nepse = nepse_df.copy()
            nepse.columns = [str(column).strip() for column in nepse.columns]
            if "nepse_close" not in nepse.columns and "close" in nepse.columns:
                nepse = nepse.rename(columns={"close": "nepse_close"})

        if "date" not in nepse.columns or "nepse_close" not in nepse.columns:
            raise ValueError("NEPSE data must contain 'date' and 'nepse_close'.")

        nepse["date"] = pd.to_datetime(nepse["date"], utc=False).dt.tz_localize(None)
        nepse["nepse_close"] = pd.to_numeric(nepse["nepse_close"], errors="coerce")
        nepse = nepse.dropna(subset=["date", "nepse_close"]).sort_values("date")
        nepse = nepse.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        nepse_ema200 = nepse["nepse_close"].ewm(span=200, adjust=False, min_periods=20).mean()
        nepse["nepse_bull"] = (nepse["nepse_close"] > nepse_ema200).astype(float)
        nepse["nepse_ret_5d"] = nepse["nepse_close"].pct_change(5)
        nepse["nepse_ret_21d"] = nepse["nepse_close"].pct_change(21)
        nepse = nepse.replace([np.inf, -np.inf], np.nan)
        nepse = nepse.dropna(subset=["nepse_ret_5d", "nepse_ret_21d"]).reset_index(drop=True)
        return nepse, used_fallback

    def _resolve_fundamentals(
        self,
        bank: str,
        fundamentals: dict[str, dict[str, float | None]] | None,
    ) -> tuple[float, float]:
        bank_key = str(bank).strip().upper()
        values = (fundamentals or {}).get(bank_key, {})
        car = values.get("car")
        npl = values.get("npl")

        if car is None or pd.isna(car):
            car = DEFAULT_CAR
        if npl is None or pd.isna(npl):
            npl = DEFAULT_NPL
        return float(car), float(npl)

    def _future_dates(self, last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
        offset = pd.offsets.CustomBusinessDay(weekmask=DEFAULT_WEEKMASK)
        return pd.date_range(start=last_date + offset, periods=steps, freq=offset)

    def _load_model(self) -> None:
        if self._uses_state_dict and self._state_dict is not None:
            return
        if (not self._uses_state_dict) and self._model is not None and self._dataset_parameters is not None:
            return

        self._ensure_runtime_dependencies()
        self._log(f"Loading TFT checkpoint from {self.checkpoint_path}")

        if self._uses_state_dict:
            loaded = torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
            if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(
                loaded.get("state_dict"), dict
            ):
                loaded = loaded["state_dict"]
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"Expected a state_dict dictionary in {self.checkpoint_path}, got {type(loaded)}."
                )
            self._state_dict = loaded
            self.target_name = NOTEBOOK_TARGET
            self.max_encoder_length = NOTEBOOK_ENCODER_LENGTH
            self.max_prediction_length = NOTEBOOK_PREDICTION_LENGTH
            self.known_reals = list(NOTEBOOK_KNOWN_REALS)
            self.unknown_reals = list(NOTEBOOK_UNKNOWN_REALS)
            self.meta["best_checkpoint"] = str(self.checkpoint_path)
            self.meta["target"] = self.target_name
            self.meta["encoder_len"] = self.max_encoder_length
            self.meta["pred_len"] = self.max_prediction_length
            self.meta["known_reals"] = self.known_reals
            self.meta["unknown_reals"] = self.unknown_reals
            self.meta["model_features"] = self._combine_feature_columns(self.meta)
            self.meta["n_features"] = len(self.meta["model_features"])
            return

        loaded_with_cpu_fallback = False
        try:
            self._model = TemporalFusionTransformer.load_from_checkpoint(
                str(self.checkpoint_path),
                map_location=self.device,
            )
        except RuntimeError as exc:
            if not self._should_use_cpu_checkpoint_fallback(exc):
                raise
            self._log("Falling back to manual CPU checkpoint load for a GPU-trained TFT artifact.")
            self._model = self._load_model_with_cpu_fallback()
            loaded_with_cpu_fallback = True

        if not (loaded_with_cpu_fallback and str(self.device) == "cpu"):
            try:
                self._model.to(self.device)
            except RuntimeError as exc:
                if not self._should_use_cpu_checkpoint_fallback(exc):
                    raise
                self._log("CUDA runtime unavailable at inference time. Falling back to CPU.")
                self.device = torch.device("cpu")
                self._model = self._load_model_with_cpu_fallback()
                loaded_with_cpu_fallback = True
        self._model.eval()

        self._dataset_parameters = self._extract_dataset_parameters(self._model)
        self.target_name = str(self._dataset_parameters.get("target", self.target_name))
        self.max_encoder_length = int(
            self._dataset_parameters.get("max_encoder_length", self.max_encoder_length)
        )
        self.max_prediction_length = int(
            self._dataset_parameters.get("max_prediction_length", self.max_prediction_length)
        )
        self.known_reals = list(
            self._dataset_parameters.get("time_varying_known_reals", self.known_reals)
        )
        self.unknown_reals = list(
            self._dataset_parameters.get("time_varying_unknown_reals", self.unknown_reals)
        )

        encoder_banks = self._extract_encoder_banks(self._dataset_parameters)
        if encoder_banks:
            self.supported_banks = encoder_banks

        self.meta["encoder_len"] = self.max_encoder_length
        self.meta["pred_len"] = self.max_prediction_length
        self.meta["known_reals"] = self.known_reals
        self.meta["unknown_reals"] = self.unknown_reals
        self.meta["banks"] = self.supported_banks
        self.meta["model_features"] = self._combine_feature_columns(self.meta)
        self.meta["n_features"] = len(self.meta["model_features"])

    def _load_model_with_cpu_fallback(self) -> Any:
        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        hyper_parameters = checkpoint.get("hyper_parameters", {})
        state_dict = checkpoint.get("state_dict", {})

        model = TemporalFusionTransformer(**hyper_parameters)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Manual CPU checkpoint load produced unexpected state_dict differences: "
                f"missing={missing}, unexpected={unexpected}"
            )

        dataset_parameters = checkpoint.get("dataset_parameters")
        if dataset_parameters is not None:
            model.dataset_parameters = dataset_parameters

        self._reset_metric_devices_to_cpu(model)

        return model

    def _should_use_cpu_checkpoint_fallback(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "no cuda gpus are available" in message
            or "torch not compiled with cuda enabled" in message
            or "found no nvidia driver" in message
            or "cuda driver version is insufficient" in message
        )

    def _reset_metric_devices_to_cpu(self, model: Any) -> None:
        cpu_device = torch.device("cpu")
        for module in model.modules():
            if hasattr(module, "_device"):
                try:
                    module._device = cpu_device
                except Exception:
                    pass

    def _extract_dataset_parameters(self, model: Any) -> dict[str, Any]:
        dataset_parameters = getattr(model, "dataset_parameters", None)
        if dataset_parameters is None:
            hparams = getattr(model, "hparams", None)
            if hparams is not None and hasattr(hparams, "dataset_parameters"):
                dataset_parameters = hparams.dataset_parameters
            elif isinstance(hparams, dict):
                dataset_parameters = hparams.get("dataset_parameters")

        if dataset_parameters is None:
            raise RuntimeError("TFT checkpoint does not expose dataset_parameters.")

        return dataset_parameters

    def _extract_encoder_banks(self, dataset_parameters: dict[str, Any]) -> list[str]:
        categorical_encoders = dataset_parameters.get("categorical_encoders", {})
        for key in ("bank", "__group_id__bank"):
            encoder = categorical_encoders.get(key)
            if encoder is None:
                continue
            classes = getattr(encoder, "classes_", None)
            if isinstance(classes, dict):
                return [str(label).strip().upper() for label in classes.keys()]
        return []

    def _ensure_runtime_dependencies(self) -> None:
        if (
            _IMPORT_ERROR is not None
            or TimeSeriesDataSet is None
            or TemporalFusionTransformer is None
            or GroupNormalizer is None
            or MAE is None
        ):
            raise RuntimeError(
                "PyTorch Forecasting runtime dependencies are missing. "
                "Install torch, lightning, and pytorch-forecasting before running inference."
            ) from _IMPORT_ERROR

    def _trainer_kwargs(self) -> dict[str, Any]:
        accelerator = "gpu" if str(self.device).startswith("cuda") else "cpu"
        return {
            "accelerator": accelerator,
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
        }

    def _resolve_device(self) -> Any:
        if torch is None:
            return "cpu"

        requested = os.getenv("TFT_INFERENCE_DEVICE", "cpu").strip().lower()
        if requested in {"cuda", "gpu"}:
            try:
                if torch.cuda.is_available():
                    torch.zeros(1, device="cuda")
                    return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")

    def _resolve_checkpoint_path(self, checkpoint_path: str | None) -> Path:
        configured_path = os.getenv("AUTOTFT_MODEL_PATH")
        if configured_path:
            return Path(configured_path).resolve()
        if checkpoint_path:
            return Path(checkpoint_path).resolve()

        search_roots = [self.artifact_dir, DEFAULT_ARTIFACT_DIR]
        for root in search_roots:
            if not root.exists():
                continue

            preferred_state = root / DEFAULT_STATE_DICT_NAME
            if preferred_state.exists():
                return preferred_state.resolve()

            preferred_ckpt = root / DEFAULT_CKPT_NAME
            if preferred_ckpt.exists():
                return preferred_ckpt.resolve()

            state_matches = sorted(root.glob("*.pth"))
            if state_matches:
                return state_matches[0].resolve()

            ckpt_matches = sorted(root.glob("*.ckpt"))
            if ckpt_matches:
                return ckpt_matches[0].resolve()

        recursive_state = sorted(Path.cwd().rglob("*.pth"))
        if recursive_state:
            return recursive_state[0].resolve()

        recursive_ckpt = sorted(Path.cwd().rglob("*.ckpt"))
        if recursive_ckpt:
            return recursive_ckpt[0].resolve()

        raise FileNotFoundError("No TFT artifact (.pth or .ckpt) could be found in the project.")

    def _load_meta(self) -> dict[str, Any]:
        candidate_files = [
            self.artifact_dir / "autotft_meta.json",
            self.artifact_dir / "tft_meta.json",
            Path("models/autotft_meta.json"),
            Path("models/tft_meta.json"),
            DEFAULT_ARTIFACT_DIR / "tft_meta.json",
        ]
        for candidate in candidate_files:
            if not candidate.exists():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except json.JSONDecodeError:
                continue
            if isinstance(meta, dict):
                return meta

        if self.checkpoint_path.suffix.lower() == ".pth":
            return {
                "best_checkpoint": str(self.checkpoint_path),
                "target": NOTEBOOK_TARGET,
                "encoder_len": NOTEBOOK_ENCODER_LENGTH,
                "pred_len": NOTEBOOK_PREDICTION_LENGTH,
                "known_reals": list(NOTEBOOK_KNOWN_REALS),
                "unknown_reals": list(NOTEBOOK_UNKNOWN_REALS),
                "banks": [],
                "metrics": {},
            }

        return {
            "best_checkpoint": str(self.checkpoint_path),
            "known_reals": [],
            "unknown_reals": [],
            "banks": [],
            "metrics": {},
        }

    def _combine_feature_columns(self, meta: dict[str, Any]) -> list[str]:
        known = [str(column) for column in meta.get("known_reals", [])]
        unknown = [str(column) for column in meta.get("unknown_reals", [])]
        combined = []
        for column in [*known, *unknown]:
            if column not in combined:
                combined.append(column)
        return combined

    def _infer_bank_symbol(self, ohlcv_df: pd.DataFrame, csv_path: str) -> str:
        if "bank" in ohlcv_df.columns:
            bank_values = ohlcv_df["bank"].dropna().astype(str).str.strip().str.upper().unique()
            if len(bank_values) == 1:
                return str(bank_values[0])

        return Path(csv_path).stem.strip().upper() or "ASSET"

    def _confidence_from_return(self, forecast_return: float) -> float:
        scaled = float(np.clip(forecast_return / 0.02, -50.0, 50.0))
        return float(1.0 / (1.0 + math.exp(-scaled)))

    @staticmethod
    def _confidence_label(prob_up: float) -> str:
        margin = abs(prob_up - 0.5)
        if margin >= 0.20:
            return "HIGH"
        if margin >= 0.10:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _signal_strength_label(return_value: float) -> str:
        absolute_return = abs(return_value)
        if absolute_return >= 0.05:
            return "STRONG"
        if absolute_return >= 0.02:
            return "MODERATE"
        return "WEAK"

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


TFTInferencePipeline = NEPSEInferencePipeline


def run_inference(
    csv_path: str,
    checkpoint_path: str | None = None,
    bank: str | None = None,
    nepse_csv_path: str | None = None,
    model_dir: str = str(DEFAULT_ARTIFACT_DIR),
    car: float = DEFAULT_CAR,
    npl: float = DEFAULT_NPL,
    policy_rate: float = POLICY_RATE_FALLBACK,
    batch_size: int = 64,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Run end-to-end TFT inference from a raw OHLCV CSV.

    Returns a forecast DataFrame with:
    - horizon_step
    - forecast_date
    - predicted_target
    - predicted_return
    - predicted_close
    - cumulative_return
    """

    pipeline = NEPSEInferencePipeline(
        model_dir=model_dir,
        verbose=False,
        checkpoint_path=checkpoint_path,
    )
    return pipeline.predict_from_csv(
        csv_path=csv_path,
        checkpoint_path=checkpoint_path,
        bank=bank,
        nepse_csv_path=nepse_csv_path,
        car=car,
        npl=npl,
        policy_rate=policy_rate,
        batch_size=batch_size,
        plot=plot,
    )
