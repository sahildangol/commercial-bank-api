import importlib
import json
import os
from datetime import date, datetime, time
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from notebooks.inference_pipeline import NEPSEInferencePipeline, POLICY_RATE_FALLBACK
from src.db.models.company import Company
from src.db.models.inference import ModelVersion, Prediction
from src.db.schema.inference import (
    FundamentalInput,
    InferenceRequest,
    InferenceResponse,
    InferenceSignal,
    ModelVersionCreate,
    PredictionOutcomeUpdate,
    PredictionRecordResponse,
)


@lru_cache(maxsize=4)
def get_pipeline(model_dir: str) -> NEPSEInferencePipeline:
    return NEPSEInferencePipeline(model_dir=model_dir)


class InferenceService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.model_dir = os.getenv("MODEL_DIR", "models")
        self.scraper_module = os.getenv("NEPSE_SCRAPER_MODULE", "src.scripts.nepse_scraper")
        self.scraper_function = os.getenv("NEPSE_SCRAPER_FUNCTION", "scrape_market_data")
        self.model_type = os.getenv("MODEL_TYPE", "ensemble")
        self.model_target = os.getenv("MODEL_TARGET", "signal_21d")

    def predict(self, payload: InferenceRequest) -> InferenceResponse:
        symbol = payload.symbol.strip().upper()
        company = self._get_company_by_symbol(symbol)
        prediction_date = payload.prediction_date or datetime.utcnow().date()
        model_version = self._get_active_model_version()

        cached = self._get_prediction(
            company_id=company.company_id,
            model_version_id=model_version.id,
            prediction_date=prediction_date,
        )
        if cached:
            return InferenceResponse(
                prediction_id=int(cached.id),
                model_version_id=model_version.id,
                prediction_date=prediction_date,
                from_cache=True,
                symbol=symbol,
                timeframe=payload.timeframe,
                lookback_days=payload.lookback_days,
                generated_at=datetime.utcnow(),
                rows_ohlcv=0,
                rows_nepse=0,
                selected_signal=self._prediction_to_signal(symbol=symbol, prediction=cached),
                all_signals=None,
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

        try:
            signals = get_pipeline(self.model_dir).predict(
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
        stored_prediction = self._save_prediction(
            company_id=company.company_id,
            model_version_id=model_version.id,
            prediction_date=prediction_date,
            signal=selected_signal,
        )

        return InferenceResponse(
            prediction_id=int(stored_prediction.id),
            model_version_id=model_version.id,
            prediction_date=prediction_date,
            from_cache=False,
            symbol=symbol,
            timeframe=payload.timeframe,
            lookback_days=payload.lookback_days,
            generated_at=datetime.utcnow(),
            rows_ohlcv=len(ohlcv_df),
            rows_nepse=len(nepse_df),
            selected_signal=selected_signal,
            all_signals=[self._row_to_signal(row) for _, row in signals.iterrows()],
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

    def _get_company_by_symbol(self, symbol: str) -> Company:
        company = (
            self.session.query(Company)
            .filter(func.upper(Company.symbol) == symbol.upper())
            .order_by(Company.company_id.asc())
            .first()
        )
        if not company:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Company with symbol '{symbol}' not found. "
                    "Create it first via POST /company."
                ),
            )
        return company

    def _get_active_model_version(self) -> ModelVersion:
        active = (
            self.session.query(ModelVersion)
            .filter(
                ModelVersion.model_type == self.model_type,
                ModelVersion.target == self.model_target,
                ModelVersion.is_active.is_(True),
            )
            .order_by(ModelVersion.id.desc())
            .first()
        )
        if active:
            return active
        return self._create_model_version_from_meta()

    def _create_model_version_from_meta(self) -> ModelVersion:
        meta = get_pipeline(self.model_dir).meta
        metrics = meta.get("metrics", {})
        features = meta.get("model_features", [])
        trained_at = self._parse_datetime(meta.get("trained_at")) or datetime.utcnow()
        train_end_date = self._parse_date(meta.get("train_end")) or datetime.utcnow().date()

        self._deactivate_versions(model_type=self.model_type, target=self.model_target)
        model_version = ModelVersion(
            model_type=self.model_type,
            target=self.model_target,
            trained_at=trained_at,
            train_end_date=train_end_date,
            n_features=int(meta.get("n_features", len(features))),
            train_auc=self._to_optional_float(metrics.get("tr_auc_ens")),
            test_auc=self._to_optional_float(metrics.get("te_auc_ens")),
            train_r2=self._to_optional_float(metrics.get("tr_r2")),
            test_r2=self._to_optional_float(metrics.get("te_r2")),
            feature_list=json.dumps(features),
            notes="Auto-created from model_meta.json.",
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
        predicted_price = int(round(signal.close * (1 + (signal.predicted_mag / 100.0))))
        prediction = Prediction(
            company_id=company_id,
            model_version_id=model_version_id,
            prediction_date=prediction_date,
            predicted_price_5=predicted_price,
            prob_direction_up=signal.prob_direction,
            prob_momentum_5d=signal.prob_momentum,
            predicted_magnitude=signal.predicted_mag,
            ensemble_score=signal.ensemble_score,
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
        return InferenceSignal(
            bank=symbol,
            date=datetime.combine(prediction.prediction_date, time.min),
            close=prediction.close_at_signal,
            prob_direction=prediction.prob_direction_up,
            prob_momentum=prediction.prob_momentum_5d,
            predicted_mag=prediction.predicted_magnitude,
            ensemble_score=prediction.ensemble_score,
            signal=prediction.signal,
            car=None,
            npl=None,
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
        return InferenceSignal(
            bank=str(row["bank"]),
            date=pd.to_datetime(row["date"]).to_pydatetime(),
            close=float(row["close"]),
            prob_direction=float(row["prob_direction"]),
            prob_momentum=float(row["prob_momentum"]),
            predicted_mag=float(row["predicted_mag"]),
            ensemble_score=float(row["ensemble_score"]),
            signal=str(row["signal"]),
            car=self._to_optional_float(row.get("car")),
            npl=self._to_optional_float(row.get("npl")),
        )

    def _to_optional_float(self, value: Any) -> float | None:
        if pd.isna(value):
            return None
        return float(value)
