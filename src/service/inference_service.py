import importlib
import importlib.util
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models.company import Company
from src.db.models.inference import ModelVersion, Prediction
from src.db.schema.inference import (
    ForecastPoint,
    FundamentalInput,
    InferenceRequest,
    InferenceResponse,
    InferenceSignal,
    ModelVersionCreate,
    PastOHLCVPoint,
    PredictionOutcomeUpdate,
    PredictionRecordResponse,
    TimelinePoint,
)


def _load_inference_pipeline() -> tuple[type, float]:
    try:
        from notebooks.inference_pipeline import NEPSEInferencePipeline, POLICY_RATE_FALLBACK

        return NEPSEInferencePipeline, POLICY_RATE_FALLBACK
    except ModuleNotFoundError as exc:
        pipeline_file = Path(__file__).resolve().parents[2] / "notebooks" / "inference_pipeline.py"
        if not pipeline_file.exists():
            raise exc

        spec = importlib.util.spec_from_file_location("notebooks.inference_pipeline", pipeline_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load inference pipeline from: {pipeline_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        pipeline_cls = getattr(module, "NEPSEInferencePipeline", None)
        if pipeline_cls is None:
            raise RuntimeError(
                f"NEPSEInferencePipeline class is missing in {pipeline_file}"
            )

        fallback_policy_rate = float(getattr(module, "POLICY_RATE_FALLBACK", 4.5))
        return pipeline_cls, fallback_policy_rate


NEPSEInferencePipeline, POLICY_RATE_FALLBACK = _load_inference_pipeline()


@lru_cache(maxsize=4)
def get_pipeline(model_dir: str, verbose: bool) -> NEPSEInferencePipeline:
    return NEPSEInferencePipeline(model_dir=model_dir, verbose=verbose)


@dataclass(frozen=True)
class EnsembleArtifacts:
    meta: dict[str, Any]
    feature_cols: list[str]
    threshold: float
    direction_classifier: Any
    regressor: Any
    momentum_classifier: Any | None
    scaler: Any | None
    label_encoder: Any | None
    fundamentals_lookup: dict[str, dict[str, float | None]]
    direction_model_path: str


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in '{path}'.")
    return payload


def _read_fundamentals_lookup(path: str | None) -> dict[str, dict[str, float | None]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if df.empty or "bank" not in df.columns:
        return {}

    records: dict[str, dict[str, float | None]] = {}
    for _, row in df.iterrows():
        bank = str(row.get("bank", "")).strip().upper()
        if not bank:
            continue
        records[bank] = {
            "car": _to_optional_float_value(row.get("car")),
            "npl": _to_optional_float_value(row.get("npl")),
        }
    return records


def _to_optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def get_ensemble_artifacts(
    meta_path: str,
    direction_model_path: str,
    regressor_model_path: str,
    momentum_model_path: str | None,
    scaler_model_path: str | None,
    label_encoder_path: str | None,
    fundamentals_lookup_path: str | None,
) -> EnsembleArtifacts:
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError("joblib is required for ensemble inference artifacts.") from exc

    meta = _read_json(meta_path)
    feature_cols = meta.get("feature_cols") or meta.get("model_features")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise RuntimeError(
            f"Ensemble metadata '{meta_path}' is missing a non-empty feature list."
        )
    feature_cols = [str(item) for item in feature_cols]
    threshold = float(meta.get("threshold", 0.5))

    direction_classifier = joblib.load(direction_model_path)
    regressor = joblib.load(regressor_model_path)

    if hasattr(direction_classifier, "feature_names_in_"):
        model_features = [str(item) for item in list(direction_classifier.feature_names_in_)]
        if model_features:
            feature_cols = model_features

    if hasattr(regressor, "feature_names_in_"):
        regressor_features = [str(item) for item in list(regressor.feature_names_in_)]
        if regressor_features != feature_cols:
            raise RuntimeError(
                "Ensemble regressor feature set does not match classifier feature set."
            )

    momentum_classifier = None
    if momentum_model_path:
        loaded = joblib.load(momentum_model_path)
        if hasattr(loaded, "feature_names_in_"):
            momentum_features = [str(item) for item in list(loaded.feature_names_in_)]
            if momentum_features == feature_cols:
                momentum_classifier = loaded
        else:
            momentum_classifier = loaded

    scaler = None
    if scaler_model_path:
        scaler = joblib.load(scaler_model_path)

    label_encoder = None
    if label_encoder_path:
        label_encoder = joblib.load(label_encoder_path)

    fundamentals_lookup = _read_fundamentals_lookup(fundamentals_lookup_path)
    return EnsembleArtifacts(
        meta=meta,
        feature_cols=feature_cols,
        threshold=threshold,
        direction_classifier=direction_classifier,
        regressor=regressor,
        momentum_classifier=momentum_classifier,
        scaler=scaler,
        label_encoder=label_encoder,
        fundamentals_lookup=fundamentals_lookup,
        direction_model_path=direction_model_path,
    )


class InferenceService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.autotft_model_dir = os.getenv("MODEL_DIR", "src/tft_artifacts")
        self.ensemble_model_dir = os.getenv("ENSEMBLE_MODEL_DIR", "models")
        self.scraper_module = os.getenv("NEPSE_SCRAPER_MODULE", "src.scripts.nepse_scraper")
        self.scraper_function = os.getenv("NEPSE_SCRAPER_FUNCTION", "scrape_market_data")
        self.autotft_model_type = os.getenv("AUTOTFT_MODEL_TYPE", os.getenv("MODEL_TYPE", "autotft"))
        self.autotft_model_target = os.getenv(
            "AUTOTFT_MODEL_TARGET",
            os.getenv("MODEL_TARGET", "multi_horizon"),
        )
        self.ensemble_model_type = os.getenv("ENSEMBLE_MODEL_TYPE", "ensemble")
        self.ensemble_model_target = os.getenv("ENSEMBLE_MODEL_TARGET", "single_step")
        self.pipeline_verbose = os.getenv("INFERENCE_PIPELINE_VERBOSE", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def predict(self, payload: InferenceRequest) -> InferenceResponse:
        # Backward compatibility: default service predict path is AutoTFT.
        return self.predict_autotft(payload)

    def predict_autotft(self, payload: InferenceRequest) -> InferenceResponse:
        symbol = payload.symbol.strip().upper()
        company = self._get_company_by_symbol(symbol)
        prediction_date = payload.prediction_date or datetime.utcnow().date()
        model_version = self._get_active_model_version(
            model_type=self.autotft_model_type,
            target=self.autotft_model_target,
            strategy="autotft",
        )

        cached = self._get_prediction(
            company_id=company.company_id,
            model_version_id=model_version.id,
            prediction_date=prediction_date,
        )
        scraped = self._run_scraper(payload)
        ohlcv_df, nepse_df = self._build_frames(scraped)
        fundamentals = self._build_fundamentals(
            payload.fundamentals,
            scraped.get("fundamentals", {}),
        )

        if payload.policy_rate is not None:
            policy_rate = payload.policy_rate
        else:
            policy_rate = float(scraped.get("policy_rate", POLICY_RATE_FALLBACK))

        pipeline = get_pipeline(self.autotft_model_dir, self.pipeline_verbose)
        try:
            signals = pipeline.predict(
                ohlcv_df=ohlcv_df,
                nepse_df=nepse_df,
                fundamentals=fundamentals,
                policy_rate=policy_rate,
                verbose=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        selected = signals[signals["bank"] == symbol]
        if selected.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No signal generated for symbol '{symbol}'.",
            )
        selected_signal = self._row_to_signal(selected.iloc[0])
        if cached:
            stored_prediction = cached
        else:
            stored_prediction = self._save_prediction(
                company_id=company.company_id,
                model_version_id=model_version.id,
                prediction_date=prediction_date,
                signal=selected_signal,
            )

        source_details = scraped.get("source") if isinstance(scraped.get("source"), dict) else None
        model_checkpoint = getattr(pipeline, "checkpoint_path", None)
        if model_checkpoint is not None:
            model_checkpoint = str(model_checkpoint)

        return InferenceResponse(
            prediction_id=int(stored_prediction.id),
            model_version_id=model_version.id,
            model_type=self.autotft_model_type,
            model_target=self.autotft_model_target,
            model_checkpoint=model_checkpoint,
            data_source=self._resolve_data_source(source_details),
            data_source_details=source_details,
            prediction_date=prediction_date,
            from_cache=cached is not None,
            symbol=symbol,
            timeframe=payload.timeframe,
            lookback_days=payload.lookback_days,
            generated_at=datetime.utcnow(),
            rows_ohlcv=len(ohlcv_df),
            rows_nepse=len(nepse_df),
            selected_signal=selected_signal,
            all_signals=[self._row_to_signal(row) for _, row in signals.iterrows()],
            past_5_days=self._extract_past_5_days(ohlcv_df=ohlcv_df, symbol=symbol),
        )

    def predict_ensemble(self, payload: InferenceRequest) -> InferenceResponse:
        symbol = payload.symbol.strip().upper()
        company = self._get_company_by_symbol(symbol)
        prediction_date = payload.prediction_date or datetime.utcnow().date()
        model_version = self._get_active_model_version(
            model_type=self.ensemble_model_type,
            target=self.ensemble_model_target,
            strategy="ensemble",
        )

        cached = self._get_prediction(
            company_id=company.company_id,
            model_version_id=model_version.id,
            prediction_date=prediction_date,
        )
        scraped = self._run_scraper(payload)
        ohlcv_df, nepse_df = self._build_frames(scraped)
        fundamentals = self._build_fundamentals(
            payload.fundamentals,
            scraped.get("fundamentals", {}),
        )
        artifacts = self._load_ensemble_artifacts()
        feature_df = self._build_ensemble_feature_frame(
            ohlcv_df=ohlcv_df,
            nepse_df=nepse_df,
            fundamentals=fundamentals,
            artifacts=artifacts,
        )
        selected_row = self._build_ensemble_signal_row(
            feature_df=feature_df,
            symbol=symbol,
            artifacts=artifacts,
        )
        selected_signal = self._row_to_signal(pd.Series(selected_row))

        if cached:
            stored_prediction = cached
        else:
            stored_prediction = self._save_prediction(
                company_id=company.company_id,
                model_version_id=model_version.id,
                prediction_date=prediction_date,
                signal=selected_signal,
            )

        source_details = scraped.get("source") if isinstance(scraped.get("source"), dict) else None
        return InferenceResponse(
            prediction_id=int(stored_prediction.id),
            model_version_id=model_version.id,
            model_type=self.ensemble_model_type,
            model_target=self.ensemble_model_target,
            model_checkpoint=artifacts.direction_model_path,
            data_source=self._resolve_data_source(source_details),
            data_source_details=source_details,
            prediction_date=prediction_date,
            from_cache=cached is not None,
            symbol=symbol,
            timeframe=payload.timeframe,
            lookback_days=payload.lookback_days,
            generated_at=datetime.utcnow(),
            rows_ohlcv=len(ohlcv_df),
            rows_nepse=len(nepse_df),
            selected_signal=selected_signal,
            all_signals=[selected_signal],
            past_5_days=self._extract_past_5_days(ohlcv_df=ohlcv_df, symbol=symbol),
        )

    def create_model_version(self, payload: ModelVersionCreate) -> ModelVersion:
        if payload.is_active:
            self._deactivate_versions(model_type=payload.model_type, target=payload.target)

        model_version = ModelVersion(**payload.model_dump())
        self.session.add(model_version)
        self.session.commit()
        self.session.refresh(model_version)
        return model_version

    def list_model_versions(self) -> list[ModelVersion]:
        return (
            self.session.query(ModelVersion)
            .order_by(ModelVersion.is_active.desc(), ModelVersion.id.desc())
            .all()
        )

    def list_supported_symbols(self) -> list[str]:
        from_db = (
            self.session.query(Company.symbol)
            .filter(Company.is_active.is_(True))
            .order_by(Company.symbol.asc())
            .all()
        )
        db_symbols = [str(row[0]).strip().upper() for row in from_db if row[0]]

        try:
            module = importlib.import_module(self.scraper_module)
        except ModuleNotFoundError:
            return sorted(set(db_symbols))

        get_symbols = getattr(module, "list_supported_symbols", None)
        if not callable(get_symbols):
            return sorted(set(db_symbols))

        try:
            scraper_symbols = [str(item).strip().upper() for item in get_symbols() if str(item).strip()]
        except Exception:
            scraper_symbols = []

        return sorted(set(db_symbols) | set(scraper_symbols))

    def activate_model_version(self, model_version_id: int) -> ModelVersion:
        model_version = self.session.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if not model_version:
            raise HTTPException(status_code=404, detail="Model version not found.")

        self._deactivate_versions(model_type=model_version.model_type, target=model_version.target)
        model_version.is_active = True
        self.session.commit()
        self.session.refresh(model_version)
        return model_version

    def list_predictions(
        self,
        symbol: str | None = None,
        prediction_date: date | None = None,
        model_version_id: int | None = None,
    ) -> list[PredictionRecordResponse]:
        query = self.session.query(Prediction)

        if symbol:
            company = self._get_company_by_symbol(symbol.strip().upper())
            query = query.filter(Prediction.company_id == company.company_id)
        if prediction_date:
            query = query.filter(Prediction.prediction_date == prediction_date)
        if model_version_id:
            query = query.filter(Prediction.model_version_id == model_version_id)

        rows = (
            query.order_by(Prediction.prediction_date.desc(), Prediction.id.desc())
            .limit(200)
            .all()
        )
        return [PredictionRecordResponse.model_validate(row) for row in rows]

    def update_prediction_outcome(
        self,
        prediction_id: int,
        payload: PredictionOutcomeUpdate,
    ) -> PredictionRecordResponse:
        prediction = self.session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found.")

        updates = payload.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one field to update.",
            )

        for key, value in updates.items():
            setattr(prediction, key, value)

        self.session.commit()
        self.session.refresh(prediction)
        return PredictionRecordResponse.model_validate(prediction)

    def _run_scraper(self, payload: InferenceRequest) -> dict[str, Any]:
        symbol = payload.symbol.strip().upper()

        try:
            module = importlib.import_module(self.scraper_module)
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Scraper module '{self.scraper_module}' was not found. "
                    "Create it in src/scripts and expose a scrape function."
                ),
            ) from exc

        scraper = getattr(module, self.scraper_function, None)
        if not callable(scraper):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Scraper function '{self.scraper_function}' was not found in "
                    f"module '{self.scraper_module}'."
                ),
            )

        try:
            result = scraper(
                symbol=symbol,
                timeframe=payload.timeframe,
                lookback_days=payload.lookback_days,
            )
        except TypeError:
            result = scraper(symbol, payload.timeframe, payload.lookback_days)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scraper failed: {exc}") from exc

        if not isinstance(result, dict):
            raise HTTPException(
                status_code=500,
                detail="Scraper must return a dict with keys: ohlcv, nepse, fundamentals, policy_rate.",
            )

        return result

    def _build_frames(self, scraped: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        ohlcv = scraped.get("ohlcv")
        nepse = scraped.get("nepse", scraped.get("nepse_index"))

        if ohlcv is None or nepse is None:
            raise HTTPException(
                status_code=500,
                detail="Scraper output must include 'ohlcv' and 'nepse' (or 'nepse_index').",
            )

        ohlcv_df = pd.DataFrame(ohlcv)
        nepse_df = pd.DataFrame(nepse)

        if ohlcv_df.empty:
            raise HTTPException(status_code=400, detail="Scraped OHLCV data is empty.")
        if nepse_df.empty:
            raise HTTPException(status_code=400, detail="Scraped NEPSE data is empty.")

        return ohlcv_df, nepse_df

    def _build_fundamentals(
        self,
        request_fundamentals: dict[str, FundamentalInput] | None,
        scraper_fundamentals: dict[str, Any],
    ) -> dict[str, dict[str, float | None]]:
        merged: dict[str, dict[str, float | None]] = {}

        if isinstance(scraper_fundamentals, dict):
            for bank, vals in scraper_fundamentals.items():
                if not isinstance(vals, dict):
                    continue
                merged[bank.strip().upper()] = {
                    "car": vals.get("car"),
                    "npl": vals.get("npl"),
                }

        if request_fundamentals:
            for bank, vals in request_fundamentals.items():
                merged[bank.strip().upper()] = vals.model_dump()

        return merged

    def _artifact_candidates(self, env_var: str, defaults: list[str]) -> list[Path]:
        candidates: list[Path] = []
        configured = os.getenv(env_var)
        if configured:
            candidates.append(Path(configured))

        for name in defaults:
            candidates.append(Path(self.ensemble_model_dir) / name)
            candidates.append(Path(name))

        deduped: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    def _resolve_ensemble_path(
        self,
        env_var: str,
        defaults: list[str],
        required: bool = True,
    ) -> str | None:
        for path in self._artifact_candidates(env_var, defaults):
            if path.exists():
                return str(path.resolve())

        if required:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Missing ensemble artifact for {env_var}. "
                    f"Checked candidates: {[str(item) for item in self._artifact_candidates(env_var, defaults)]}"
                ),
            )
        return None

    def _load_ensemble_artifacts(self) -> EnsembleArtifacts:
        meta_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_META_PATH",
            defaults=["model_meta.json"],
        )
        direction_model_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_DIRECTION_MODEL_PATH",
            defaults=["direction_classifier.pkl", "model_clf_dir.pkl"],
        )
        regressor_model_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_REGRESSOR_MODEL_PATH",
            defaults=["return_regressor.pkl", "model_reg_mag.pkl"],
        )
        momentum_model_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_MOMENTUM_MODEL_PATH",
            defaults=["model_clf_mom.pkl"],
            required=False,
        )
        scaler_model_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_SCALER_MODEL_PATH",
            defaults=["model_scaler.pkl"],
            required=False,
        )
        label_encoder_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_LABEL_ENCODER_PATH",
            defaults=["label_encoder.pkl"],
            required=False,
        )
        fundamentals_lookup_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_FUNDAMENTALS_LOOKUP_PATH",
            defaults=["fundamental_lookup.csv"],
            required=False,
        )
        try:
            return get_ensemble_artifacts(
                meta_path=meta_path,
                direction_model_path=direction_model_path,
                regressor_model_path=regressor_model_path,
                momentum_model_path=momentum_model_path,
                scaler_model_path=scaler_model_path,
                label_encoder_path=label_encoder_path,
                fundamentals_lookup_path=fundamentals_lookup_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load ensemble artifacts: {exc}") from exc

    def _prepare_ohlcv_for_ensemble(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "bank", "close", "high", "low", "volume"}
        missing = [item for item in required if item not in ohlcv_df.columns]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Scraper OHLCV is missing required columns for ensemble inference: {missing}",
            )

        df = ohlcv_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["bank"] = df["bank"].astype(str).str.strip().str.upper()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if "open" not in df.columns:
            df["open"] = df["close"]

        df = df.dropna(subset=["date", "bank", "close", "high", "low", "volume"])
        df = (
            df.sort_values(["bank", "date"])
            .drop_duplicates(subset=["bank", "date"], keep="last")
            .reset_index(drop=True)
        )
        if df.empty:
            raise HTTPException(status_code=400, detail="No usable OHLCV rows for ensemble inference.")
        return df

    def _prepare_nepse_for_ensemble(self, nepse_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        if nepse_df.empty:
            np_df = pd.DataFrame()
        else:
            np_df = nepse_df.copy()

        if not np_df.empty and "date" in np_df.columns:
            np_df["date"] = pd.to_datetime(np_df["date"], errors="coerce")
        else:
            np_df = pd.DataFrame(columns=["date", "nepse_close"])

        if "nepse_close" not in np_df.columns:
            close_column = "close" if "close" in np_df.columns else None
            if close_column:
                np_df["nepse_close"] = pd.to_numeric(np_df[close_column], errors="coerce")

        np_df = np_df.dropna(subset=["date", "nepse_close"]) if not np_df.empty else np_df
        np_df = (
            np_df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        if np_df.empty:
            fallback = (
                ohlcv_df.groupby("date", as_index=False)["close"]
                .mean()
                .rename(columns={"close": "nepse_close"})
            )
            np_df = fallback

        np_df["nepse_ret_1d"] = np_df["nepse_close"].pct_change()
        np_df["nepse_ma_50"] = np_df["nepse_close"].rolling(50, min_periods=5).mean()
        np_df["nepse_bull"] = (np_df["nepse_close"] >= np_df["nepse_ma_50"]).astype(float)
        np_df["nepse_ret_1d"] = np_df["nepse_ret_1d"].replace([np.inf, -np.inf], np.nan)
        return np_df[["date", "nepse_ret_1d", "nepse_bull"]]

    def _encode_bank_value(self, bank: str, artifacts: EnsembleArtifacts) -> float:
        if artifacts.label_encoder is not None and hasattr(artifacts.label_encoder, "classes_"):
            classes = [str(item).strip().upper() for item in list(artifacts.label_encoder.classes_)]
            if bank in classes:
                return float(artifacts.label_encoder.transform([bank])[0])

        meta_classes = artifacts.meta.get("bank_classes")
        if isinstance(meta_classes, list):
            normalized = [str(item).strip().upper() for item in meta_classes]
            if bank in normalized:
                return float(normalized.index(bank))

        return float("nan")

    def _build_ensemble_feature_frame(
        self,
        ohlcv_df: pd.DataFrame,
        nepse_df: pd.DataFrame,
        fundamentals: dict[str, dict[str, float | None]],
        artifacts: EnsembleArtifacts,
    ) -> pd.DataFrame:
        df = self._prepare_ohlcv_for_ensemble(ohlcv_df)
        nepse = self._prepare_nepse_for_ensemble(nepse_df, df)
        df = df.merge(nepse, on="date", how="left")
        df["nepse_ret_1d"] = df["nepse_ret_1d"].ffill().bfill().fillna(0.0)
        df["nepse_bull"] = df["nepse_bull"].ffill().bfill().fillna(0.0)

        default_car = float(os.getenv("NEPSE_DEFAULT_CAR", 12.0))
        default_npl = float(os.getenv("NEPSE_DEFAULT_NPL", 2.0))

        def car_for(bank: str) -> float:
            scoped = fundamentals.get(bank) if isinstance(fundamentals, dict) else None
            if isinstance(scoped, dict):
                value = _to_optional_float_value(scoped.get("car"))
                if value is not None:
                    return value
            lookup = artifacts.fundamentals_lookup.get(bank)
            if isinstance(lookup, dict):
                value = _to_optional_float_value(lookup.get("car"))
                if value is not None:
                    return value
            return default_car

        def npl_for(bank: str) -> float:
            scoped = fundamentals.get(bank) if isinstance(fundamentals, dict) else None
            if isinstance(scoped, dict):
                value = _to_optional_float_value(scoped.get("npl"))
                if value is not None:
                    return value
            lookup = artifacts.fundamentals_lookup.get(bank)
            if isinstance(lookup, dict):
                value = _to_optional_float_value(lookup.get("npl"))
                if value is not None:
                    return value
            return default_npl

        df["car"] = df["bank"].map(car_for).astype(float)
        df["npl"] = df["bank"].map(npl_for).astype(float)

        df["ret_raw"] = df.groupby("bank")["close"].pct_change()
        sector_ret = (
            df.groupby("date", as_index=False)["ret_raw"]
            .mean()
            .rename(columns={"ret_raw": "sector_ret"})
        )
        df = df.merge(sector_ret, on="date", how="left")

        groups: list[pd.DataFrame] = []
        for bank_name, group in df.groupby("bank", sort=False):
            g = group.sort_values("date").copy()
            c = g["close"]
            ret = c.pct_change()

            g["ret_lag1"] = ret.shift(1)
            g["ret_3d_lag1"] = ret.rolling(3, min_periods=3).sum().shift(1)

            ma20 = c.rolling(20, min_periods=20).mean()
            std20 = c.rolling(20, min_periods=20).std()
            g["bb_pct_lag1"] = ((c - (ma20 - (2 * std20))) / ((4 * std20) + 1e-9)).shift(1)

            lo14 = g["low"].rolling(14, min_periods=14).min()
            hi14 = g["high"].rolling(14, min_periods=14).max()
            g["stoch_k_lag1"] = ((c - lo14) / ((hi14 - lo14) + 1e-9) * 100.0).shift(1)

            tr = pd.concat(
                [
                    g["high"] - g["low"],
                    (g["high"] - c.shift(1)).abs(),
                    (g["low"] - c.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            g["atr_ratio_lag1"] = (tr.rolling(14, min_periods=14).mean() / c.replace(0, np.nan)).shift(1)

            vol_5 = ret.rolling(5, min_periods=5).std()
            vol_21 = ret.rolling(21, min_periods=21).std()
            g["vol_ratio_5_21"] = vol_5 / (vol_21 + 1e-9)

            ma21 = c.rolling(21, min_periods=21).mean()
            g["close_to_ma21"] = (c / ma21) - 1.0

            lo252 = c.rolling(252, min_periods=60).min()
            g["dist_52w_low"] = (c / lo252) - 1.0

            vol_ma21 = g["volume"].rolling(21, min_periods=5).mean()
            g["vol_surge_lag1"] = (g["volume"] / vol_ma21.replace(0, np.nan)).shift(1)

            g["sector_ret_lag1"] = g["sector_ret"].shift(1)
            g["nepse_ret_lag1"] = g["nepse_ret_1d"].shift(1)
            g["month"] = g["date"].dt.month.astype(float)
            g["bank_enc"] = self._encode_bank_value(bank_name, artifacts)

            groups.append(g)

        engineered = pd.concat(groups, ignore_index=True)
        engineered = engineered.replace([np.inf, -np.inf], np.nan)
        return engineered.sort_values(["bank", "date"]).reset_index(drop=True)

    def _predict_probability_from_classifier(self, model: Any, features: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            if probabilities.ndim != 2 or probabilities.shape[0] == 0:
                raise HTTPException(status_code=500, detail="Classifier returned invalid probability shape.")
            if probabilities.shape[1] == 1:
                return float(np.clip(probabilities[0][0], 0.0, 1.0))

            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    index = classes.index(1)
                    return float(np.clip(probabilities[0][index], 0.0, 1.0))
            return float(np.clip(probabilities[0][-1], 0.0, 1.0))

        if hasattr(model, "decision_function"):
            logits = model.decision_function(features)
            value = float(np.ravel(logits)[0])
            return float(1.0 / (1.0 + np.exp(-value)))

        raise HTTPException(status_code=500, detail="Classifier does not expose predict_proba/decision_function.")

    def _build_ensemble_signal_row(
        self,
        feature_df: pd.DataFrame,
        symbol: str,
        artifacts: EnsembleArtifacts,
    ) -> dict[str, Any]:
        bank_rows = feature_df[feature_df["bank"] == symbol].sort_values("date")
        if bank_rows.empty:
            raise HTTPException(status_code=404, detail=f"No feature rows available for symbol '{symbol}'.")

        for column in artifacts.feature_cols:
            if column not in bank_rows.columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ensemble feature column '{column}' is missing from engineered frame.",
                )

        usable = bank_rows.dropna(subset=artifacts.feature_cols)
        if usable.empty:
            latest = bank_rows.iloc[-1]
            missing = [col for col in artifacts.feature_cols if pd.isna(latest.get(col))]
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Not enough history to build ensemble features for '{symbol}'. "
                    f"Missing in latest row: {missing}"
                ),
            )

        latest = usable.iloc[-1]
        features = usable.tail(1)[artifacts.feature_cols]

        prob_direction = self._predict_probability_from_classifier(artifacts.direction_classifier, features)
        predicted_return_raw = float(np.ravel(artifacts.regressor.predict(features))[0])
        if abs(predicted_return_raw) > 1.0:
            return_magnitude = predicted_return_raw / 100.0
        else:
            return_magnitude = predicted_return_raw
        predicted_mag = return_magnitude * 100.0

        if artifacts.momentum_classifier is not None:
            prob_momentum = self._predict_probability_from_classifier(artifacts.momentum_classifier, features)
        else:
            prob_momentum = prob_direction

        if artifacts.scaler is not None:
            try:
                mag_norm = float(np.ravel(artifacts.scaler.transform([[predicted_return_raw]]))[0])
                model_score = (0.6 * prob_direction) + (0.4 * mag_norm)
            except Exception:
                model_score = (0.6 * prob_direction) + (0.4 * prob_momentum)
        else:
            model_score = (0.6 * prob_direction) + (0.4 * prob_momentum)
        model_score = float(np.clip(model_score, 0.0, 1.0))

        threshold = artifacts.threshold
        direction = self._normalize_signal(
            raw_signal=None,
            prob_up=prob_direction,
            predicted_mag=predicted_mag,
            threshold=threshold,
        )

        confidence = self._confidence_label(prob_direction)
        signal_strength = self._signal_strength_label(return_magnitude)
        return {
            "bank": symbol,
            "date": pd.to_datetime(latest["date"]).to_pydatetime(),
            "close": float(latest["close"]),
            "prob_direction": float(prob_direction),
            "prob_momentum": float(prob_momentum),
            "predicted_mag": float(predicted_mag),
            "model_score": float(model_score),
            "signal": direction,
            "direction": direction,
            "prob_up": float(prob_direction),
            "prob_down": float(1.0 - prob_direction),
            "return_magnitude": float(return_magnitude),
            "return_magnitude_pct": f"{return_magnitude * 100:+.3f}%",
            "confidence": confidence,
            "signal_strength": signal_strength,
            "threshold_used": float(threshold),
            "nan_features": None,
            "car": _to_optional_float_value(latest.get("car")),
            "npl": _to_optional_float_value(latest.get("npl")),
            "forecast_next_5d": None,
            "timeline_10d": None,
        }

    def _extract_past_5_days(self, ohlcv_df: pd.DataFrame, symbol: str) -> list[PastOHLCVPoint] | None:
        if "bank" not in ohlcv_df.columns or "date" not in ohlcv_df.columns:
            return None

        frame = ohlcv_df.copy()
        frame["bank"] = frame["bank"].astype(str).str.strip().str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["bank"] == symbol].sort_values("date").tail(5)
        if frame.empty:
            return None

        points: list[PastOHLCVPoint] = []
        for _, row in frame.iterrows():
            close = _to_optional_float_value(row.get("close"))
            if close is None:
                continue
            open_value = _to_optional_float_value(row.get("open"))
            high_value = _to_optional_float_value(row.get("high"))
            low_value = _to_optional_float_value(row.get("low"))
            points.append(
                PastOHLCVPoint(
                    date=pd.to_datetime(row["date"]).to_pydatetime(),
                    open=close if open_value is None else open_value,
                    high=close if high_value is None else high_value,
                    low=close if low_value is None else low_value,
                    close=close,
                    volume=_to_optional_float_value(row.get("volume")),
                    amount=_to_optional_float_value(row.get("amount")),
                )
            )
        return points or None

    def _get_company_by_symbol(self, symbol: str) -> Company:
        company = (
            self.session.query(Company)
            .filter(func.upper(Company.symbol) == symbol.upper())
            .order_by(Company.company_id.asc())
            .first()
        )
        if company:
            return company

        auto_create_company = os.getenv("INFERENCE_AUTO_CREATE_COMPANY", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not auto_create_company:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Company with symbol '{symbol}' not found. "
                    "Create it first via POST /company."
                ),
            )

        created = Company(
            symbol=symbol,
            company_name=symbol,
            sector="Commercial Banks",
            listed_shares=0,
            is_active=True,
        )
        self.session.add(created)
        try:
            self.session.commit()
            self.session.refresh(created)
            return created
        except IntegrityError:
            self.session.rollback()
            company = (
                self.session.query(Company)
                .filter(func.upper(Company.symbol) == symbol.upper())
                .order_by(Company.company_id.asc())
                .first()
            )
            if company:
                return company
            raise HTTPException(
                status_code=500,
                detail=f"Failed to auto-create company for symbol '{symbol}'.",
            )

    def _get_active_model_version(
        self,
        model_type: str,
        target: str,
        strategy: str,
    ) -> ModelVersion:
        active = (
            self.session.query(ModelVersion)
            .filter(
                ModelVersion.model_type == model_type,
                ModelVersion.target == target,
                ModelVersion.is_active.is_(True),
            )
            .order_by(ModelVersion.id.desc())
            .first()
        )
        if active:
            return active

        if strategy == "ensemble":
            return self._create_ensemble_model_version_from_meta(
                model_type=model_type,
                target=target,
            )
        return self._create_autotft_model_version_from_meta(
            model_type=model_type,
            target=target,
        )

    def _create_autotft_model_version_from_meta(self, model_type: str, target: str) -> ModelVersion:
        meta = get_pipeline(self.autotft_model_dir, self.pipeline_verbose).meta
        metrics = meta.get("metrics", {})
        features = (
            meta.get("model_features")
            or meta.get("feature_cols")
            or [*meta.get("known_reals", []), *meta.get("unknown_reals", [])]
        )
        trained_at = self._parse_datetime(meta.get("trained_at")) or datetime.utcnow()
        train_end_value = meta.get("train_end") or meta.get("train_cutoff") or meta.get("train_end_date")
        train_end_date = self._parse_date(train_end_value) or datetime.utcnow().date()

        train_metrics = metrics.get("train", {}) if isinstance(metrics.get("train"), dict) else {}
        val_metrics = metrics.get("val", {}) if isinstance(metrics.get("val"), dict) else {}
        test_metrics = metrics.get("test", {}) if isinstance(metrics.get("test"), dict) else {}

        train_auc = self._to_optional_float(metrics.get("tr_auc_ens"))
        if train_auc is None:
            train_auc = self._to_optional_float(train_metrics.get("auc"))

        test_auc = self._to_optional_float(metrics.get("te_auc_ens"))
        if test_auc is None:
            test_auc = self._to_optional_float(test_metrics.get("auc"))
        if test_auc is None:
            test_auc = self._to_optional_float(val_metrics.get("auc"))
        if test_auc is None:
            test_auc = self._to_optional_float(meta.get("cv_mean_auc"))

        train_r2 = self._to_optional_float(metrics.get("tr_r2"))
        if train_r2 is None:
            train_r2 = self._to_optional_float(train_metrics.get("r2"))

        test_r2 = self._to_optional_float(metrics.get("te_r2"))
        if test_r2 is None:
            test_r2 = self._to_optional_float(test_metrics.get("r2"))
        if test_r2 is None:
            test_r2 = self._to_optional_float(val_metrics.get("r2"))

        self._deactivate_versions(model_type=model_type, target=target)
        model_version = ModelVersion(
            model_type=model_type,
            target=target,
            trained_at=trained_at,
            train_end_date=train_end_date,
            n_features=int(meta.get("n_features", len(features))),
            train_auc=train_auc,
            test_auc=test_auc,
            train_r2=train_r2,
            test_r2=test_r2,
            feature_list=json.dumps(features),
            notes="Auto-created from TFT artifact metadata.",
            is_active=True,
        )
        self.session.add(model_version)
        self.session.commit()
        self.session.refresh(model_version)
        return model_version

    def _create_ensemble_model_version_from_meta(self, model_type: str, target: str) -> ModelVersion:
        artifacts = self._load_ensemble_artifacts()
        meta = artifacts.meta
        metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
        features = artifacts.feature_cols

        trained_at = self._parse_datetime(meta.get("trained_at")) or datetime.utcnow()
        train_end_value = meta.get("train_end") or meta.get("train_cutoff") or meta.get("split_date")
        train_end_date = self._parse_date(train_end_value) or datetime.utcnow().date()

        train_auc = self._to_optional_float(meta.get("tr_auc_ens"))
        if train_auc is None:
            train_auc = self._to_optional_float(metrics.get("tr_auc_ens"))
        if train_auc is None:
            train_auc = self._to_optional_float(metrics.get("tr_auc_dir"))

        test_auc = self._to_optional_float(meta.get("te_auc_ens"))
        if test_auc is None:
            test_auc = self._to_optional_float(metrics.get("te_auc_ens"))
        if test_auc is None:
            test_auc = self._to_optional_float(meta.get("cv_mean_auc"))
        if test_auc is None:
            test_auc = self._to_optional_float(metrics.get("te_auc_dir"))

        train_r2 = self._to_optional_float(metrics.get("tr_r2"))
        test_r2 = self._to_optional_float(metrics.get("te_r2"))

        self._deactivate_versions(model_type=model_type, target=target)
        model_version = ModelVersion(
            model_type=model_type,
            target=target,
            trained_at=trained_at,
            train_end_date=train_end_date,
            n_features=len(features),
            train_auc=train_auc,
            test_auc=test_auc,
            train_r2=train_r2,
            test_r2=test_r2,
            feature_list=json.dumps(features),
            notes="Auto-created from ensemble artifact metadata.",
            is_active=True,
        )
        self.session.add(model_version)
        self.session.commit()
        self.session.refresh(model_version)
        return model_version

    def _deactivate_versions(self, model_type: str, target: str) -> None:
        (
            self.session.query(ModelVersion)
            .filter(
                ModelVersion.model_type == model_type,
                ModelVersion.target == target,
                ModelVersion.is_active.is_(True),
            )
            .update({ModelVersion.is_active: False}, synchronize_session=False)
        )

    def _get_prediction(
        self,
        company_id: int,
        model_version_id: int,
        prediction_date: date,
    ) -> Prediction | None:
        return (
            self.session.query(Prediction)
            .filter(
                Prediction.company_id == company_id,
                Prediction.model_version_id == model_version_id,
                Prediction.prediction_date == prediction_date,
            )
            .first()
        )

    def _save_prediction(
        self,
        company_id: int,
        model_version_id: int,
        prediction_date: date,
        signal: InferenceSignal,
    ) -> Prediction:
        predicted_price = self._predicted_price_5(signal)
        prediction = Prediction(
            company_id=company_id,
            model_version_id=model_version_id,
            prediction_date=prediction_date,
            predicted_price_5=predicted_price,
            prob_direction_up=signal.prob_direction,
            prob_momentum_5d=signal.prob_momentum,
            predicted_magnitude=signal.predicted_mag,
            ensemble_score=signal.model_score,
            signal=signal.signal,
            close_at_signal=signal.close,
            actual_close_21d=None,
            actual_return_21d=None,
            was_correct=None,
        )
        self.session.add(prediction)
        try:
            self.session.commit()
            self.session.refresh(prediction)
            return prediction
        except IntegrityError:
            self.session.rollback()
            existing = self._get_prediction(company_id, model_version_id, prediction_date)
            if existing:
                return existing
            raise

    def _prediction_to_signal(self, symbol: str, prediction: Prediction) -> InferenceSignal:
        prob_up = float(prediction.prob_direction_up)
        prob_down = 1.0 - prob_up
        threshold = self._current_threshold()
        signal_value = str(prediction.signal) if prediction.signal is not None else None
        return_mag = float(prediction.predicted_magnitude) / 100.0
        predicted_mag = float(prediction.predicted_magnitude)
        direction = self._normalize_signal(
            raw_signal=signal_value,
            prob_up=prob_up,
            predicted_mag=predicted_mag,
            threshold=threshold,
        )
        return_mag_pct = f"{return_mag * 100:+.3f}%"
        confidence = self._confidence_label(prob_up)
        signal_strength = self._signal_strength_label(return_mag)

        return InferenceSignal(
            bank=symbol,
            date=datetime.combine(prediction.prediction_date, time.min),
            close=prediction.close_at_signal,
            prob_direction=prob_up,
            prob_momentum=prediction.prob_momentum_5d,
            predicted_mag=predicted_mag,
            model_score=prediction.ensemble_score,
            signal=direction,
            direction=direction,
            prob_up=prob_up,
            prob_down=prob_down,
            return_magnitude=return_mag,
            return_magnitude_pct=return_mag_pct,
            confidence=confidence,
            signal_strength=signal_strength,
            threshold_used=threshold,
            nan_features=None,
            car=None,
            npl=None,
            forecast_next_5d=None,
            timeline_10d=None,
        )

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _parse_date(self, value: Any) -> date | None:
        if not value:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _row_to_signal(self, row: pd.Series) -> InferenceSignal:
        prob_direction = float(row["prob_direction"])
        prob_momentum = self._to_optional_float(row.get("prob_momentum"))
        if prob_momentum is None:
            prob_momentum = prob_direction

        model_score = self._to_optional_float(row.get("model_score"))
        if model_score is None:
            model_score = self._to_optional_float(row.get("ensemble_score"))
        if model_score is None:
            model_score = prob_direction

        predicted_mag = float(row["predicted_mag"])
        prob_up = self._to_optional_float(row.get("prob_up"))
        if prob_up is None:
            prob_up = prob_direction
        prob_down = self._to_optional_float(row.get("prob_down"))
        if prob_down is None and prob_up is not None:
            prob_down = 1.0 - prob_up

        return_mag = self._to_optional_float(row.get("return_magnitude"))
        if return_mag is None:
            return_mag = predicted_mag / 100.0

        threshold = self._to_optional_float(row.get("threshold_used"))
        if threshold is None:
            threshold = self._current_threshold()

        raw_signal = row.get("signal")
        if raw_signal is None or (isinstance(raw_signal, float) and pd.isna(raw_signal)):
            raw_signal = row.get("direction")
        direction = self._normalize_signal(
            raw_signal=raw_signal,
            prob_up=prob_up,
            predicted_mag=predicted_mag,
            threshold=threshold,
        )

        confidence = row.get("confidence")
        if not confidence or (isinstance(confidence, float) and pd.isna(confidence)):
            confidence = self._confidence_label(prob_up)

        signal_strength = row.get("signal_strength")
        if not signal_strength or (isinstance(signal_strength, float) and pd.isna(signal_strength)):
            signal_strength = self._signal_strength_label(return_mag)

        return_mag_pct = row.get("return_magnitude_pct")
        if not return_mag_pct or (isinstance(return_mag_pct, float) and pd.isna(return_mag_pct)):
            return_mag_pct = f"{return_mag * 100:+.3f}%"

        nan_features = row.get("nan_features")
        if not isinstance(nan_features, list):
            nan_features = None

        forecast_next_5d = self._normalize_forecast_points(row.get("forecast_next_5d"))
        timeline_10d = self._normalize_timeline_points(row.get("timeline_10d"))

        return InferenceSignal(
            bank=str(row["bank"]),
            date=pd.to_datetime(row["date"]).to_pydatetime(),
            close=float(row["close"]),
            prob_direction=prob_direction,
            prob_momentum=prob_momentum,
            predicted_mag=predicted_mag,
            model_score=model_score,
            signal=direction,
            direction=str(direction),
            prob_up=prob_up,
            prob_down=prob_down,
            return_magnitude=return_mag,
            return_magnitude_pct=str(return_mag_pct),
            confidence=str(confidence),
            signal_strength=str(signal_strength),
            threshold_used=threshold,
            nan_features=nan_features,
            car=self._to_optional_float(row.get("car")),
            npl=self._to_optional_float(row.get("npl")),
            forecast_next_5d=forecast_next_5d,
            timeline_10d=timeline_10d,
        )

    def _resolve_data_source(self, source_details: dict[str, Any] | None) -> str | None:
        if not source_details:
            return None
        target_source = source_details.get("target_source")
        if not isinstance(target_source, dict):
            return None
        source_name = target_source.get("source")
        if not source_name:
            return None
        return str(source_name)

    def _normalize_signal(
        self,
        raw_signal: Any,
        prob_up: float,
        predicted_mag: float,
        threshold: float,
    ) -> str:
        if raw_signal is not None:
            normalized = str(raw_signal).strip().upper()
            if normalized in {"UP", "DOWN"}:
                return normalized
        if predicted_mag > 0:
            return "UP"
        if predicted_mag < 0:
            return "DOWN"
        return "UP" if prob_up >= threshold else "DOWN"

    def _predicted_price_5(self, signal: InferenceSignal) -> float:
        if signal.forecast_next_5d:
            return float(signal.forecast_next_5d[-1].predicted_close)
        return float(signal.close * (1 + (signal.predicted_mag / 100.0)))

    def _normalize_forecast_points(self, value: Any) -> list[ForecastPoint] | None:
        if not isinstance(value, list):
            return None

        points: list[ForecastPoint] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            forecast_date = item.get("forecast_date")
            if forecast_date is None:
                continue
            points.append(
                ForecastPoint(
                    horizon_day=int(item["horizon_day"]),
                    forecast_date=pd.to_datetime(forecast_date).to_pydatetime(),
                    predicted_close=float(item["predicted_close"]),
                    predicted_return=float(item["predicted_return"]),
                    cumulative_return=float(item["cumulative_return"]),
                )
            )
        return points or None

    def _normalize_timeline_points(self, value: Any) -> list[TimelinePoint] | None:
        if not isinstance(value, list):
            return None

        points: list[TimelinePoint] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            date_value = item.get("date")
            close_value = item.get("close")
            if date_value is None or close_value is None:
                continue
            points.append(
                TimelinePoint(
                    date=pd.to_datetime(date_value).to_pydatetime(),
                    point_type=str(item.get("point_type", "")).strip().lower() or "history",
                    horizon_day=(
                        None if item.get("horizon_day") is None else int(item.get("horizon_day"))
                    ),
                    open=self._to_optional_float(item.get("open")),
                    high=self._to_optional_float(item.get("high")),
                    low=self._to_optional_float(item.get("low")),
                    close=float(close_value),
                    volume=self._to_optional_float(item.get("volume")),
                    predicted_return=self._to_optional_float(item.get("predicted_return")),
                    cumulative_return=self._to_optional_float(item.get("cumulative_return")),
                )
            )
        return points or None

    def _current_threshold(self) -> float:
        try:
            pipeline = get_pipeline(self.autotft_model_dir, self.pipeline_verbose)
            threshold = float(getattr(pipeline, "threshold", 0.5))
        except Exception:
            threshold = 0.5
        return threshold

    def _confidence_label(self, prob_up: float) -> str:
        margin = abs(prob_up - 0.5)
        if margin >= 0.12:
            return "HIGH"
        if margin >= 0.06:
            return "MEDIUM"
        return "LOW"

    def _signal_strength_label(self, return_mag: float) -> str:
        abs_ret = abs(return_mag)
        if abs_ret >= 0.02:
            return "STRONG"
        if abs_ret >= 0.01:
            return "MODERATE"
        return "WEAK"

    def _to_optional_float(self, value: Any) -> float | None:
        return _to_optional_float_value(value)
