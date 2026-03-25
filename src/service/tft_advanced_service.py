from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from src.db.schema.tft_advanced import (
    AdvancedConfidenceInterval,
    AdvancedForecastResponse,
    AdvancedHistoryResponse,
    AdvancedPredictionResponse,
)

try:
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only failure
    torch = None
    TimeSeriesDataSet = None
    TemporalFusionTransformer = None
    _IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _IMPORT_ERROR = None

DEFAULT_PRICE_VOLUME_URL_TEMPLATE = "http://localhost:8000/PriceVolumeHistory?symbol={symbol}"
DEFAULT_CHECKPOINT_PATH = "models/tft_best_model.ckpt"
DEFAULT_BUSINESS_WEEKMASK = "Sun Mon Tue Wed Thu"
DEFAULT_POLICY_RATE = 4.5


class AdvancedTFTError(Exception):
    pass


class DataFetchError(AdvancedTFTError):
    pass


class SymbolNotFoundError(AdvancedTFTError):
    pass


class InsufficientDataError(AdvancedTFTError):
    pass


class DataIntegrityError(AdvancedTFTError):
    pass


class InferenceError(AdvancedTFTError):
    pass


class AdvancedTFTInferenceService:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        price_volume_url_template: str = DEFAULT_PRICE_VOLUME_URL_TEMPLATE,
        min_records: int = 45,
        business_weekmask: str = DEFAULT_BUSINESS_WEEKMASK,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.price_volume_url_template = price_volume_url_template
        self.min_records = int(min_records)
        self.business_weekmask = business_weekmask
        self.timeout_seconds = float(timeout_seconds)

        self.max_encoder_length = 30
        self.max_prediction_length = 5
        self.target_name = "close"

        self._model: Any | None = None
        self._dataset_parameters: dict[str, Any] | None = None
        self._supported_symbols: set[str] = set()
        self._quantile_levels: list[float] = []
        self._load_lock = threading.Lock()

    def predict(self, symbol: str) -> AdvancedPredictionResponse:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise SymbolNotFoundError("Symbol is required.")

        self._ensure_model_loaded()
        if self._supported_symbols and normalized_symbol not in self._supported_symbols:
            supported = ", ".join(sorted(self._supported_symbols))
            raise SymbolNotFoundError(
                f"Symbol '{normalized_symbol}' is not supported by this TFT model. Supported symbols: {supported}"
            )

        raw_rows = self._fetch_symbol_rows(normalized_symbol)
        history = self.transform_rows_to_features(normalized_symbol, raw_rows)
        if len(history) < self.max_encoder_length:
            raise InsufficientDataError(
                f"Need at least {self.max_encoder_length} rows for encoder history, got {len(history)}."
            )

        inference_frame, future_dates = self.build_inference_frame(history, normalized_symbol)
        point_forecast, quantile_forecast, quantile_levels = self._predict_arrays(inference_frame)

        return self._build_response(
            symbol=normalized_symbol,
            history=history,
            future_dates=future_dates,
            point_forecast=point_forecast,
            quantile_forecast=quantile_forecast,
            quantile_levels=quantile_levels,
        )

    def transform_rows_to_features(
        self,
        symbol: str,
        raw_rows: list[dict[str, Any]],
    ) -> pd.DataFrame:
        mapped_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            mapped_rows.append(
                {
                    "date": row.get("businessDate"),
                    "open": row.get("openPrice"),
                    "high": row.get("highPrice"),
                    "low": row.get("lowPrice"),
                    "close": row.get("closePrice"),
                    "volume": row.get("totalTradedQuantity"),
                    "symbol": symbol,
                }
            )

        df = pd.DataFrame(mapped_rows)
        if df.empty:
            raise SymbolNotFoundError(f"No rows were returned for symbol '{symbol}'.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False).dt.tz_localize(None)
        for column in ("open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["date", "close"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if df.empty:
            raise SymbolNotFoundError(f"No usable rows were returned for symbol '{symbol}'.")
        if len(df) < self.min_records:
            raise InsufficientDataError(
                f"Need at least {self.min_records} records for inference, got {len(df)}."
            )

        if (df["close"] <= 0).any():
            raise DataIntegrityError("Close price must be strictly positive for log-return features.")
        if df["volume"].eq(0).any():
            zero_dates = [pd.Timestamp(dt).date().isoformat() for dt in df.loc[df["volume"].eq(0), "date"]]
            raise DataIntegrityError(
                "totalTradedQuantity contains zero values on dates: " + ", ".join(zero_dates)
            )

        self._validate_business_day_gaps(df["date"])

        previous_close = df["close"].shift(1)
        df["open"] = df["open"].fillna(previous_close).fillna(df["close"])
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        df["policy_rate"] = self._policy_rate()

        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["momentum_20d"] = (df["close"] / df["close"].shift(20)) - 1.0
        df["volatility_20d"] = log_returns.rolling(20, min_periods=20).std()
        volume_mean = df["volume"].rolling(20, min_periods=20).mean()
        volume_std = df["volume"].rolling(20, min_periods=20).std().replace(0.0, np.nan)
        df["volume_z"] = (df["volume"] - volume_mean) / volume_std
        df["rolling_mean_7"] = df["close"].rolling(7, min_periods=7).mean()
        df["rolling_std_7"] = df["close"].rolling(7, min_periods=7).std()
        df["day_of_week"] = df["date"].dt.dayofweek.astype(str)
        df["month"] = df["date"].dt.month.astype(str)

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df["time_idx"] = np.arange(len(df), dtype=np.int64)
        df["symbol"] = symbol

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "momentum_20d",
            "volatility_20d",
            "volume_z",
            "rolling_mean_7",
            "rolling_std_7",
            "day_of_week",
            "month",
            "time_idx",
            "policy_rate",
            "symbol",
        ]
        missing_after_fill = [column for column in required if df[column].isna().any()]
        if missing_after_fill:
            raise DataIntegrityError(
                "Unable to fully impute feature set. Remaining null columns: "
                + ", ".join(missing_after_fill)
            )

        return df.reset_index(drop=True)

    def build_inference_frame(
        self,
        history: pd.DataFrame,
        symbol: str,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        last_row = history.iloc[-1]
        future_dates = self._future_dates(
            last_date=pd.Timestamp(last_row["date"]),
            steps=self.max_prediction_length,
        )

        future = pd.DataFrame({"date": future_dates})
        future["symbol"] = symbol
        future["open"] = float(last_row["close"])
        future["high"] = float(last_row["close"])
        future["low"] = float(last_row["close"])
        future["close"] = float(last_row["close"])
        future["volume"] = float(last_row["volume"])
        future["momentum_20d"] = float(last_row["momentum_20d"])
        future["volatility_20d"] = float(last_row["volatility_20d"])
        future["volume_z"] = float(last_row["volume_z"])
        future["rolling_mean_7"] = float(last_row["rolling_mean_7"])
        future["rolling_std_7"] = float(last_row["rolling_std_7"])
        future["policy_rate"] = float(last_row["policy_rate"])
        future["day_of_week"] = future["date"].dt.dayofweek.astype(str)
        future["month"] = future["date"].dt.month.astype(str)
        future["time_idx"] = np.arange(
            int(last_row["time_idx"]) + 1,
            int(last_row["time_idx"]) + 1 + self.max_prediction_length,
            dtype=np.int64,
        )

        inference = pd.concat([history, future], ignore_index=True, sort=False)
        if self.target_name not in inference.columns:
            inference[self.target_name] = inference["close"]

        for column in self._categorical_columns():
            if column not in inference.columns:
                inference[column] = symbol if column in {"symbol"} else "0"
            inference[column] = inference[column].astype(str)

        for column in self._real_columns():
            if column not in inference.columns:
                inference[column] = 0.0
            inference[column] = pd.to_numeric(inference[column], errors="coerce")

        inference = inference.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return inference.reset_index(drop=True), future_dates

    def _predict_arrays(
        self,
        inference_frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        if self._dataset_parameters is None or self._model is None:
            raise InferenceError("Model is not loaded.")

        dataset = TimeSeriesDataSet.from_parameters(
            self._dataset_parameters,
            inference_frame,
            predict=True,
            stop_randomization=True,
        )
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        prediction_output = self._model.predict(
            dataloader,
            return_x=True,
            mode="prediction",
            trainer_kwargs=self._trainer_kwargs(),
        )
        quantile_output = self._model.predict(
            dataloader,
            return_x=True,
            mode="quantiles",
            trainer_kwargs=self._trainer_kwargs(),
        )

        point_forecast = self._to_numpy(getattr(prediction_output, "output", prediction_output))
        quantile_forecast = self._to_numpy(getattr(quantile_output, "output", quantile_output))
        return point_forecast, quantile_forecast, list(self._quantile_levels)

    def _build_response(
        self,
        symbol: str,
        history: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
        point_forecast: np.ndarray,
        quantile_forecast: np.ndarray,
        quantile_levels: list[float],
    ) -> AdvancedPredictionResponse:
        horizon_index = self.max_prediction_length - 1
        point = self._normalize_point_forecast(point_forecast)
        quantiles = self._normalize_quantile_forecast(quantile_forecast)

        if point.shape[1] < self.max_prediction_length:
            raise InferenceError(
                f"Model returned only {point.shape[1]} forecast steps; expected {self.max_prediction_length}."
            )
        if quantiles.shape[1] < self.max_prediction_length:
            raise InferenceError(
                f"Model quantile output has only {quantiles.shape[1]} steps; expected {self.max_prediction_length}."
            )

        predicted_magnitude = float(point[0, horizon_index])
        low_bound = self._extract_quantile(
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            quantile=0.1,
            horizon_index=horizon_index,
        )
        high_bound = self._extract_quantile(
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            quantile=0.9,
            horizon_index=horizon_index,
        )

        as_of_date = pd.Timestamp(history["date"].iloc[-1]).date()
        target_date = pd.Timestamp(future_dates[horizon_index]).date()
        last_close = float(history["close"].iloc[-1])
        expected_direction = "UP" if predicted_magnitude >= last_close else "DOWN"
        expected_change_pct = (
            ((predicted_magnitude - last_close) / last_close) * 100.0 if last_close != 0 else 0.0
        )

        return AdvancedPredictionResponse(
            symbol=symbol,
            as_of_date=as_of_date,
            history=AdvancedHistoryResponse(
                last_7_days=[float(value) for value in history["close"].tail(7).tolist()]
            ),
            forecast=AdvancedForecastResponse(
                target_date=target_date,
                predicted_magnitude=predicted_magnitude,
                confidence_interval=AdvancedConfidenceInterval(low=low_bound, high=high_bound),
                expected_direction=expected_direction,
                expected_change_pct=float(expected_change_pct),
            ),
        )

    def _extract_quantile(
        self,
        quantiles: np.ndarray,
        quantile_levels: list[float],
        quantile: float,
        horizon_index: int,
    ) -> float:
        index = self._quantile_index(quantile_levels, quantile)
        if index >= quantiles.shape[2]:
            raise InferenceError(
                f"Quantile index {index} out of bounds for quantile output with shape {quantiles.shape}."
            )
        return float(quantiles[0, horizon_index, index])

    def _quantile_index(self, quantile_levels: list[float], target: float) -> int:
        for idx, level in enumerate(quantile_levels):
            if abs(float(level) - float(target)) < 1e-8:
                return idx
        raise InferenceError(f"Quantile {target} is not available in model output levels {quantile_levels}.")

    def _normalize_point_forecast(self, forecast: np.ndarray) -> np.ndarray:
        array = np.asarray(forecast, dtype=float)
        if array.ndim == 0:
            raise InferenceError("Point forecast output is scalar; expected horizon vector.")
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 2:
            return array

        squeezed = array[0]
        while squeezed.ndim > 1:
            squeezed = squeezed[..., 0]
        return np.asarray(squeezed, dtype=float).reshape(1, -1)

    def _normalize_quantile_forecast(self, forecast: np.ndarray) -> np.ndarray:
        array = np.asarray(forecast, dtype=float)
        if array.ndim == 0:
            raise InferenceError("Quantile forecast output is scalar; expected horizon matrix.")
        if array.ndim == 1:
            return array.reshape(1, -1, 1)
        if array.ndim == 2:
            if array.shape[0] == self.max_prediction_length:
                return array.reshape(1, array.shape[0], array.shape[1])
            return array.reshape(array.shape[0], array.shape[1], 1)
        if array.ndim == 3:
            return array

        trimmed = array[0]
        while trimmed.ndim > 3:
            trimmed = trimmed[0]
        if trimmed.ndim == 2:
            return trimmed.reshape(1, trimmed.shape[0], trimmed.shape[1])
        if trimmed.ndim == 1:
            return trimmed.reshape(1, -1, 1)
        return np.asarray(trimmed, dtype=float)

    def _fetch_symbol_rows(self, symbol: str) -> list[dict[str, Any]]:
        template = os.getenv("NEPSE_PRICE_VOLUME_URL_TEMPLATE", self.price_volume_url_template).strip()
        if not template:
            raise DataFetchError("NEPSE_PRICE_VOLUME_URL_TEMPLATE is empty.")

        url = self._render_symbol_url(template=template, symbol=symbol)
        timeout = self._env_float("NEPSE_API_TIMEOUT_SECONDS", self.timeout_seconds)

        request = Request(
            url=url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ml-ete-advanced-tft/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise SymbolNotFoundError(f"Symbol '{symbol}' was not found by data source.") from exc
            raise DataFetchError(f"Data source returned HTTP {exc.code} for symbol '{symbol}'.") from exc
        except URLError as exc:
            raise DataFetchError(f"Unable to reach data source for symbol '{symbol}': {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DataFetchError("Data source returned invalid JSON.") from exc

        rows = self._extract_list_payload(payload)
        if rows is None:
            raise DataFetchError("Unexpected data payload shape; expected a JSON array.")
        if not rows:
            raise SymbolNotFoundError(f"No history rows found for symbol '{symbol}'.")
        if len(rows) < self.min_records:
            raise InsufficientDataError(
                f"Need at least {self.min_records} rows from source, got {len(rows)}."
            )
        return rows

    def _extract_list_payload(self, payload: Any) -> list[dict[str, Any]] | None:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "result", "items", "history"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return None

    def _validate_business_day_gaps(self, dates: pd.Series) -> None:
        ordered = pd.to_datetime(dates, errors="coerce").dropna().sort_values().reset_index(drop=True)
        if ordered.empty:
            return

        offset = CustomBusinessDay(weekmask=self.business_weekmask)
        for previous, current in zip(ordered[:-1], ordered[1:]):
            business_gap = len(pd.date_range(start=previous, end=current, freq=offset)) - 1
            if business_gap > 3:
                raise DataIntegrityError(
                    "Date sequence contains a gap larger than 3 business days "
                    f"between {pd.Timestamp(previous).date()} and {pd.Timestamp(current).date()}."
                )

    def _future_dates(self, last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
        offset = CustomBusinessDay(weekmask=self.business_weekmask)
        return pd.date_range(start=last_date + offset, periods=steps, freq=offset)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._dataset_parameters is not None:
            return

        with self._load_lock:
            if self._model is not None and self._dataset_parameters is not None:
                return

            self._ensure_runtime_dependencies()
            if not self.checkpoint_path.exists():
                raise InferenceError(f"Checkpoint not found: {self.checkpoint_path}")

            try:
                model = TemporalFusionTransformer.load_from_checkpoint(
                    str(self.checkpoint_path),
                    map_location="cpu",
                )
            except RuntimeError as exc:
                if not self._should_use_cpu_checkpoint_fallback(exc):
                    raise
                model = self._load_model_with_cpu_fallback()

            model.eval()
            self._model = model
            self._dataset_parameters = self._extract_dataset_parameters(model)
            self.target_name = str(self._dataset_parameters.get("target", self.target_name))
            self.max_encoder_length = int(
                self._dataset_parameters.get("max_encoder_length", self.max_encoder_length)
            )
            self.max_prediction_length = int(
                self._dataset_parameters.get("max_prediction_length", self.max_prediction_length)
            )
            self._supported_symbols = self._extract_supported_symbols(self._dataset_parameters)
            self._quantile_levels = self._extract_quantile_levels(model)

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
            raise InferenceError(
                "Manual CPU checkpoint load produced state mismatch: "
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
            raise InferenceError("TFT checkpoint does not expose dataset_parameters.")
        return dataset_parameters

    def _extract_supported_symbols(self, dataset_parameters: dict[str, Any]) -> set[str]:
        categorical_encoders = dataset_parameters.get("categorical_encoders", {})
        for key in ("symbol", "__group_id__symbol"):
            encoder = categorical_encoders.get(key)
            if encoder is None:
                continue
            classes = getattr(encoder, "classes_", None)
            if isinstance(classes, dict):
                return {str(item).strip().upper() for item in classes.keys()}
        return set()

    def _extract_quantile_levels(self, model: Any) -> list[float]:
        loss = getattr(model, "loss", None)
        quantiles = getattr(loss, "quantiles", None)
        if quantiles is None:
            return [0.1, 0.5, 0.9]
        return [float(value) for value in quantiles]

    def _categorical_columns(self) -> list[str]:
        if self._dataset_parameters is None:
            return ["symbol", "day_of_week", "month"]

        columns: list[str] = []
        for key in ("group_ids", "static_categoricals", "time_varying_known_categoricals"):
            values = self._dataset_parameters.get(key, [])
            for value in values:
                if value not in columns:
                    columns.append(value)
        return columns

    def _real_columns(self) -> list[str]:
        if self._dataset_parameters is None:
            return [
                "time_idx",
                "policy_rate",
                "open",
                "high",
                "low",
                "volume",
                "momentum_20d",
                "volatility_20d",
                "volume_z",
                "rolling_mean_7",
                "rolling_std_7",
                "close",
            ]

        columns: list[str] = [self.target_name]
        for key in (
            "time_varying_known_reals",
            "time_varying_unknown_reals",
            "static_reals",
        ):
            values = self._dataset_parameters.get(key, [])
            for value in values:
                if value not in columns:
                    columns.append(value)
        return columns

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "output"):
            return self._to_numpy(value.output)
        if isinstance(value, dict):
            for key in ("prediction", "predictions", "output"):
                if key in value:
                    return self._to_numpy(value[key])
        if isinstance(value, (list, tuple)):
            if not value:
                raise InferenceError("Model returned an empty prediction container.")
            return self._to_numpy(value[0])
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _trainer_kwargs(self) -> dict[str, Any]:
        return {
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
        }

    def _policy_rate(self) -> float:
        return self._env_float("NEPSE_POLICY_RATE", DEFAULT_POLICY_RATE)

    def _env_float(self, key: str, fallback: float) -> float:
        raw = os.getenv(key)
        if raw is None or not str(raw).strip():
            return float(fallback)
        try:
            return float(raw)
        except ValueError:
            return float(fallback)

    def _render_symbol_url(self, template: str, symbol: str) -> str:
        encoded_symbol = quote(symbol, safe="")
        if "{symbol}" in template:
            return template.replace("{symbol}", encoded_symbol)

        parsed = urlparse(template)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["symbol"] = symbol
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _ensure_runtime_dependencies(self) -> None:
        if _IMPORT_ERROR is not None:
            raise InferenceError(
                "PyTorch Forecasting runtime dependencies are missing. "
                "Install torch, lightning, and pytorch-forecasting."
            ) from _IMPORT_ERROR


advanced_tft_service = AdvancedTFTInferenceService()
