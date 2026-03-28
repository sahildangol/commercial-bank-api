import importlib
import importlib.util
import json
import os
import pickle
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
    model_bundle: dict[str, Any]
    selected_features: list[str]
    classifier: Any
    signal_label_encoder: Any | None
    bank_encoder: Any | None
    cluster_map: dict[str, int]
    forward_days: int
    cv_auc_mean: float | None
    trained_on: str | None
    model_path: str


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
def get_ensemble_artifacts(model_path: str) -> EnsembleArtifacts:
    model_file = Path(model_path)
    with model_file.open("rb") as handle:
        model_bundle = pickle.load(handle)

    if not isinstance(model_bundle, dict):
        raise RuntimeError("Expected a dict artifact in NEPSE model pickle.")

    classifier = model_bundle.get("model")
    if classifier is None or not hasattr(classifier, "predict") or not hasattr(classifier, "predict_proba"):
        raise RuntimeError("NEPSE model artifact is missing a classifier with predict/predict_proba.")

    selected_features_raw = model_bundle.get("selected_features")
    if not isinstance(selected_features_raw, list) or not selected_features_raw:
        raise RuntimeError("NEPSE model artifact must include non-empty 'selected_features'.")
    selected_features = [str(item) for item in selected_features_raw]

    cluster_map: dict[str, int] = {}
    cluster_raw = model_bundle.get("cluster_map")
    if isinstance(cluster_raw, dict):
        for key, value in cluster_raw.items():
            bank = str(key).strip().upper()
            if not bank:
                continue
            try:
                cluster_map[bank] = int(value)
            except (TypeError, ValueError):
                continue

    forward_days = 5
    try:
        forward_days = max(int(model_bundle.get("forward_days", 5)), 1)
    except (TypeError, ValueError):
        forward_days = 5

    return EnsembleArtifacts(
        model_bundle=model_bundle,
        selected_features=selected_features,
        classifier=classifier,
        signal_label_encoder=model_bundle.get("label_encoder"),
        bank_encoder=model_bundle.get("bank_encoder"),
        cluster_map=cluster_map,
        forward_days=forward_days,
        cv_auc_mean=_to_optional_float_value(model_bundle.get("cv_auc_mean")),
        trained_on=str(model_bundle.get("trained_on")) if model_bundle.get("trained_on") else None,
        model_path=str(model_file.resolve()),
    )


class InferenceService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.autotft_model_dir = os.getenv("MODEL_DIR", "models")
        self.ensemble_model_dir = os.getenv("ENSEMBLE_MODEL_DIR", "models")
        self.scraper_module = os.getenv("NEPSE_SCRAPER_MODULE", "src.scripts.nepse_scraper")
        self.scraper_function = os.getenv("NEPSE_SCRAPER_FUNCTION", "scrape_market_data")
        self.autotft_model_type = os.getenv("AUTOTFT_MODEL_TYPE", os.getenv("MODEL_TYPE", "autotft"))
        self.autotft_model_target = os.getenv(
            "AUTOTFT_MODEL_TARGET",
            os.getenv("MODEL_TARGET", "multi_horizon"),
        )
        self.autotft_include_active_context = os.getenv(
            "AUTOTFT_INCLUDE_ACTIVE_CONTEXT",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
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
        scraped = self._run_scraper(
            payload,
            include_active_context=self.autotft_include_active_context,
        )
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
        minimum_lookback = max(int(os.getenv("ENSEMBLE_MIN_LOOKBACK_DAYS", "520")), 1)
        if payload.lookback_days < minimum_lookback:
            if hasattr(payload, "model_copy"):
                request_payload = payload.model_copy(update={"lookback_days": minimum_lookback})
            else:
                request_payload = InferenceRequest(
                    **{**payload.model_dump(), "lookback_days": minimum_lookback}
                )
        else:
            request_payload = payload
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
        scraped = self._run_scraper(request_payload)
        ohlcv_df, nepse_df = self._build_frames(scraped)
        fundamentals = self._build_fundamentals(
            request_payload.fundamentals,
            scraped.get("fundamentals", {}),
        )
        if request_payload.policy_rate is not None:
            policy_rate = float(request_payload.policy_rate)
        else:
            policy_rate = float(scraped.get("policy_rate", POLICY_RATE_FALLBACK))
        artifacts = self._load_ensemble_artifacts()
        feature_df = self._build_ensemble_feature_frame(
            ohlcv_df=ohlcv_df,
            nepse_df=nepse_df,
            fundamentals=fundamentals,
            policy_rate=policy_rate,
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
            model_checkpoint=artifacts.model_path,
            data_source=self._resolve_data_source(source_details),
            data_source_details=source_details,
            prediction_date=prediction_date,
            from_cache=cached is not None,
            symbol=symbol,
            timeframe=request_payload.timeframe,
            lookback_days=request_payload.lookback_days,
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
        commercial_companies = self.list_commercial_bank_companies()
        commercial_symbols = [
            str(item.get("symbol", "")).strip().upper()
            for item in commercial_companies
            if str(item.get("symbol", "")).strip()
        ]

        try:
            module = importlib.import_module(self.scraper_module)
        except ModuleNotFoundError:
            return sorted(set(db_symbols) | set(commercial_symbols))

        get_symbols = getattr(module, "list_supported_symbols", None)
        if not callable(get_symbols):
            return sorted(set(db_symbols) | set(commercial_symbols))

        try:
            scraper_symbols = [str(item).strip().upper() for item in get_symbols() if str(item).strip()]
        except Exception:
            scraper_symbols = []

        return sorted(set(db_symbols) | set(scraper_symbols) | set(commercial_symbols))

    def list_commercial_bank_companies(self) -> list[dict[str, Any]]:
        try:
            module = importlib.import_module(self.scraper_module)
        except ModuleNotFoundError:
            module = None

        if module is not None:
            fetch_companies = getattr(module, "list_commercial_bank_companies", None)
            if callable(fetch_companies):
                try:
                    rows = fetch_companies()
                    if isinstance(rows, list):
                        return [item for item in rows if isinstance(item, dict)]
                except Exception:
                    pass

        # Fallback for dropdown continuity when remote API is unavailable.
        companies = (
            self.session.query(Company)
            .filter(
                Company.is_active.is_(True),
                func.lower(Company.sector) == "commercial banks",
            )
            .order_by(Company.symbol.asc())
            .all()
        )
        return [
            {
                "id": int(company.company_id),
                "companyName": company.company_name,
                "symbol": company.symbol,
                "securityName": company.company_name,
                "status": "A" if bool(company.is_active) else "I",
                "companyEmail": None,
                "website": None,
                "sectorName": company.sector,
                "regulatoryBody": None,
                "instrumentType": "Equity",
            }
            for company in companies
            if company.symbol
        ]

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

    def _run_scraper(
        self,
        payload: InferenceRequest,
        include_active_context: bool = False,
    ) -> dict[str, Any]:
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
                include_active_context=include_active_context,
            )
        except TypeError:
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
        model_path = self._resolve_ensemble_path(
            env_var="ENSEMBLE_MODEL_PATH",
            defaults=["nepse_model.pkl"],
        )
        try:
            return get_ensemble_artifacts(model_path=model_path)
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
            np_df = pd.DataFrame(columns=["date", "nepse_close"])
        else:
            np_df = nepse_df.copy()
            if "date" in np_df.columns:
                np_df["date"] = pd.to_datetime(np_df["date"], errors="coerce")
            else:
                np_df["date"] = pd.NaT

            if "nepse_close" in np_df.columns:
                np_df["nepse_close"] = pd.to_numeric(np_df["nepse_close"], errors="coerce")
            elif "close" in np_df.columns:
                np_df["nepse_close"] = pd.to_numeric(np_df["close"], errors="coerce")
            else:
                np_df["nepse_close"] = np.nan

            np_df = np_df[["date", "nepse_close"]]

        np_df = np_df.dropna(subset=["date", "nepse_close"])
        np_df = np_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        if np_df.empty:
            np_df = (
                ohlcv_df.groupby("date", as_index=False)["close"]
                .mean()
                .rename(columns={"close": "nepse_close"})
            )

        np_df = np_df.sort_values("date").reset_index(drop=True)
        np_df["nepse_close"] = np_df["nepse_close"].ffill().bfill()
        close = np_df["nepse_close"]
        np_df["nepse_ret_1d"] = np.log(close / close.shift(1))
        np_df["nepse_ret_5d"] = np.log(close / close.shift(5))
        np_df["nepse_ret_21d"] = np.log(close / close.shift(21))
        np_df = np_df.replace([np.inf, -np.inf], np.nan)
        return np_df[["date", "nepse_close", "nepse_ret_1d", "nepse_ret_5d", "nepse_ret_21d"]]

    def _encode_bank_feature(self, bank: str, artifacts: EnsembleArtifacts) -> int:
        encoder = artifacts.bank_encoder
        if encoder is None or not hasattr(encoder, "classes_") or not hasattr(encoder, "transform"):
            return -1

        classes = {str(item).strip().upper() for item in list(encoder.classes_)}
        if bank not in classes:
            return -1
        try:
            return int(encoder.transform([bank])[0])
        except Exception:
            return -1

    def _build_ensemble_feature_frame(
        self,
        ohlcv_df: pd.DataFrame,
        nepse_df: pd.DataFrame,
        fundamentals: dict[str, dict[str, float | None]],
        policy_rate: float,
        artifacts: EnsembleArtifacts,
    ) -> pd.DataFrame:
        df = self._prepare_ohlcv_for_ensemble(ohlcv_df)
        nepse = self._prepare_nepse_for_ensemble(nepse_df, df)
        df = df.merge(nepse, on="date", how="left")
        df["nepse_close"] = df["nepse_close"].ffill().bfill()
        df["nepse_ret_1d"] = df["nepse_ret_1d"].ffill().bfill()
        df["nepse_ret_5d"] = df["nepse_ret_5d"].ffill().bfill()
        df["nepse_ret_21d"] = df["nepse_ret_21d"].ffill().bfill()
        df["policy_rate"] = float(policy_rate)

        default_car = float(os.getenv("NEPSE_DEFAULT_CAR", 12.0))
        default_npl = float(os.getenv("NEPSE_DEFAULT_NPL", 2.0))

        def car_for(bank: str) -> float:
            scoped = fundamentals.get(bank) if isinstance(fundamentals, dict) else None
            if isinstance(scoped, dict):
                value = _to_optional_float_value(scoped.get("car"))
                if value is not None:
                    return value
            return default_car

        def npl_for(bank: str) -> float:
            scoped = fundamentals.get(bank) if isinstance(fundamentals, dict) else None
            if isinstance(scoped, dict):
                value = _to_optional_float_value(scoped.get("npl"))
                if value is not None:
                    return value
            return default_npl

        df["car"] = df["bank"].map(car_for).astype(float)
        df["npl"] = df["bank"].map(npl_for).astype(float)

        groups: list[pd.DataFrame] = []
        for bank_name, group in df.groupby("bank", sort=False):
            g = group.sort_values("date").copy()
            c = g["close"]
            h = g["high"]
            lo = g["low"]
            v = g["volume"]
            o = g["open"]
            nc = g["nepse_close"]

            g["log_ret_1d"] = np.log(c / c.shift(1))
            g["log_ret_3d"] = np.log(c / c.shift(3))
            g["log_ret_5d"] = np.log(c / c.shift(5))
            g["log_ret_10d"] = np.log(c / c.shift(10))
            g["log_ret_21d"] = np.log(c / c.shift(21))

            g["sma_5"] = c.rolling(5).mean()
            g["sma_21"] = c.rolling(21).mean()
            g["sma_63"] = c.rolling(63).mean()
            g["ema_9"] = c.ewm(span=9, adjust=False).mean()
            g["ema_21"] = c.ewm(span=21, adjust=False).mean()

            g["price_to_sma5"] = c / g["sma_5"] - 1
            g["price_to_sma21"] = c / g["sma_21"] - 1
            g["price_to_sma63"] = c / g["sma_63"] - 1
            g["sma5_to_sma21"] = g["sma_5"] / g["sma_21"] - 1
            g["sma21_to_sma63"] = g["sma_21"] / g["sma_63"] - 1

            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            g["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

            gain2 = delta.clip(lower=0).rolling(28).mean()
            loss2 = (-delta.clip(upper=0)).rolling(28).mean()
            g["rsi_28"] = 100 - (100 / (1 + gain2 / (loss2 + 1e-9)))

            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            g["macd_norm"] = macd / (c + 1e-9)
            g["macd_hist_norm"] = (macd - macd_signal) / (c + 1e-9)

            g["roc_5"] = c.pct_change(5)
            g["roc_21"] = c.pct_change(21)

            g["vol_21"] = g["log_ret_1d"].rolling(21).std()
            g["vol_63"] = g["log_ret_1d"].rolling(63).std()
            g["vol_ratio"] = g["vol_21"] / (g["vol_63"] + 1e-9)

            tr = pd.concat(
                [
                    h - lo,
                    (h - c.shift(1)).abs(),
                    (lo - c.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            g["atr_14_norm"] = tr.rolling(14).mean() / (c + 1e-9)

            bb_mid = c.rolling(20).mean()
            bb_std = c.rolling(20).std()
            g["bb_width"] = (2 * bb_std) / (bb_mid + 1e-9)
            g["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

            g["hl_ratio"] = (h - lo) / (c + 1e-9)
            g["co_ratio"] = (c - o) / (h - lo + 1e-9)
            g["upper_shadow"] = (h - np.maximum(o, c)) / (h - lo + 1e-9)
            g["lower_shadow"] = (np.minimum(o, c) - lo) / (h - lo + 1e-9)

            v_ma20 = v.rolling(20).mean()
            g["vol_ratio_20"] = v / (v_ma20 + 1e-9)
            g["vol_ma5_ma20"] = v.rolling(5).mean() / (v_ma20 + 1e-9)

            obv = (np.sign(g["log_ret_1d"]) * v).cumsum()
            obv_std = obv.rolling(21).std()
            g["obv_momentum"] = obv.diff(5) / (obv_std + 1e-9)
            g["pv_divergence"] = g["log_ret_5d"] * (1 - g["vol_ratio_20"])

            g["pct_from_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1
            g["pct_from_52w_low"] = c / c.rolling(252, min_periods=60).min() - 1

            nepse_ret_1d = np.log(nc / nc.shift(1))
            g["alpha_1d"] = g["log_ret_1d"] - nepse_ret_1d
            g["alpha_5d"] = g["log_ret_5d"] - g["nepse_ret_5d"]
            g["alpha_21d"] = g["log_ret_21d"] - g["nepse_ret_21d"]
            g["nepse_bull_derived"] = (nc.rolling(21).mean() > nc.rolling(63).mean()).astype(int)

            g["month"] = g["date"].dt.month
            g["quarter"] = g["date"].dt.quarter
            g["day_of_week"] = g["date"].dt.dayofweek
            g["fiscal_q"] = ((g["date"].dt.month - 7) % 12 // 3 + 1)
            g["covid_regime"] = (
                (g["date"] >= "2020-02-01")
                & (g["date"] <= "2021-12-31")
            ).astype(int)
            g["high_rate_regime"] = (g["policy_rate"] >= 5.5).astype(int)

            g["close_zscore_63"] = (c - c.rolling(63).mean()) / (c.rolling(63).std() + 1e-9)
            g["vol_zscore_63"] = (v - v.rolling(63).mean()) / (v.rolling(63).std() + 1e-9)

            g["policy_rate_chg"] = g["policy_rate"].diff().fillna(0.0)
            g["car_chg"] = g["car"].diff().fillna(0.0)
            g["npl_chg"] = g["npl"].diff().fillna(0.0)

            g["bank_enc"] = self._encode_bank_feature(bank_name, artifacts)
            g["bank_cluster"] = int(artifacts.cluster_map.get(bank_name, -1))

            groups.append(g)

        engineered = pd.concat(groups, ignore_index=True)
        engineered = engineered.replace([np.inf, -np.inf], np.nan)
        return engineered.sort_values(["bank", "date"]).reset_index(drop=True)

    def _prob_for_class(
        self,
        labels: list[str],
        probabilities: np.ndarray,
        target_class: str,
    ) -> float:
        normalized = [item.strip().lower() for item in labels]
        target = target_class.strip().lower()
        if target not in normalized:
            return 0.0
        idx = normalized.index(target)
        return float(np.clip(probabilities[idx], 0.0, 1.0))

    def _estimate_return_from_class_probs(
        self,
        history_close: pd.Series,
        prob_buy: float,
        prob_hold: float,
        prob_sell: float,
        forward_days: int,
    ) -> float:
        trailing = np.log(history_close / history_close.shift(forward_days)).dropna()
        if trailing.empty:
            return (prob_buy - prob_sell) * 0.01

        buy_thr = float(np.percentile(trailing, 70))
        sell_thr = float(np.percentile(trailing, 30))
        hold_level = float(np.median(trailing))
        expected_log_ret = (prob_buy * buy_thr) + (prob_hold * hold_level) + (prob_sell * sell_thr)
        return float(np.expm1(expected_log_ret))

    def _build_ensemble_signal_row(
        self,
        feature_df: pd.DataFrame,
        symbol: str,
        artifacts: EnsembleArtifacts,
    ) -> dict[str, Any]:
        bank_rows = feature_df[feature_df["bank"] == symbol].sort_values("date")
        if bank_rows.empty:
            raise HTTPException(status_code=404, detail=f"No feature rows available for symbol '{symbol}'.")

        for column in artifacts.selected_features:
            if column not in bank_rows.columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ensemble feature column '{column}' is missing from engineered frame.",
                )

        usable = bank_rows.dropna(subset=artifacts.selected_features)
        nan_features: list[str] | None = None
        if usable.empty:
            latest = bank_rows.iloc[-1].copy()
            feature_values = latest[artifacts.selected_features].copy()
            nan_features = [
                column for column in artifacts.selected_features if pd.isna(feature_values.get(column))
            ]
            for column in nan_features:
                bank_series = bank_rows[column].dropna()
                if not bank_series.empty:
                    feature_values[column] = bank_series.iloc[-1]
                    continue

                global_series = feature_df[column].dropna()
                if not global_series.empty:
                    feature_values[column] = float(global_series.median())
                else:
                    feature_values[column] = 0.0
            features = pd.DataFrame([feature_values], columns=artifacts.selected_features)
        else:
            latest = usable.iloc[-1]
            features = usable.tail(1)[artifacts.selected_features]
        probabilities = np.asarray(artifacts.classifier.predict_proba(features))[0]

        if artifacts.signal_label_encoder is not None and hasattr(
            artifacts.signal_label_encoder, "classes_"
        ):
            labels = [str(item) for item in list(artifacts.signal_label_encoder.classes_)]
            predicted_label = str(
                artifacts.signal_label_encoder.inverse_transform(
                    np.asarray(artifacts.classifier.predict(features))
                )[0]
            )
        elif hasattr(artifacts.classifier, "classes_"):
            labels = [str(item) for item in list(artifacts.classifier.classes_)]
            predicted_label = str(labels[int(np.argmax(probabilities))])
        else:
            labels = ["Buy", "Hold", "Sell"]
            predicted_label = labels[int(np.argmax(probabilities))]

        prob_buy = self._prob_for_class(labels, probabilities, "Buy")
        prob_hold = self._prob_for_class(labels, probabilities, "Hold")
        prob_sell = self._prob_for_class(labels, probabilities, "Sell")

        prob_direction = float(np.clip(prob_buy, 0.0, 1.0))
        prob_momentum = prob_direction
        model_score = float(np.clip(float(np.max(probabilities)), 0.0, 1.0))
        return_magnitude = self._estimate_return_from_class_probs(
            history_close=bank_rows["close"],
            prob_buy=prob_buy,
            prob_hold=prob_hold,
            prob_sell=prob_sell,
            forward_days=artifacts.forward_days,
        )
        predicted_mag = float(return_magnitude * 100.0)

        threshold = float(os.getenv("ENSEMBLE_SIGNAL_THRESHOLD", "0.5"))
        raw_signal = "UP" if predicted_label.strip().lower() == "buy" else "DOWN"
        if predicted_label.strip().lower() == "hold":
            raw_signal = "UP" if prob_buy >= prob_sell else "DOWN"
        direction = self._normalize_signal(
            raw_signal=raw_signal,
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
            "nan_features": nan_features,
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
        features = artifacts.selected_features

        trained_at = self._parse_datetime(artifacts.trained_on)
        if trained_at is None:
            trained_date = self._parse_date(artifacts.trained_on)
            if trained_date is not None:
                trained_at = datetime.combine(trained_date, time.min)
        if trained_at is None:
            trained_at = datetime.utcnow()

        train_end_date = self._parse_date(artifacts.trained_on) or datetime.utcnow().date()
        test_auc = artifacts.cv_auc_mean

        self._deactivate_versions(model_type=model_type, target=target)
        model_version = ModelVersion(
            model_type=model_type,
            target=target,
            trained_at=trained_at,
            train_end_date=train_end_date,
            n_features=len(features),
            train_auc=None,
            test_auc=test_auc,
            train_r2=None,
            test_r2=None,
            feature_list=json.dumps(features),
            notes="Auto-created from NEPSE HistGradient artifact metadata.",
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
            threshold = float(os.getenv("ENSEMBLE_SIGNAL_THRESHOLD", "0.5"))
        except ValueError:
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
