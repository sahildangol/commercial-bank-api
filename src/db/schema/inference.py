from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FundamentalInput(BaseModel):
    car: float | None = None
    npl: float | None = None


class ForecastPoint(BaseModel):
    horizon_day: int
    forecast_date: datetime
    predicted_close: float
    predicted_return: float
    cumulative_return: float


class TimelinePoint(BaseModel):
    date: datetime
    point_type: str
    horizon_day: int | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None
    predicted_return: float | None = None
    cumulative_return: float | None = None


class PastOHLCVPoint(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    amount: float | None = None


class InferenceSimpleRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "NABIL",
            }
        }
    )

    symbol: str = Field(
        min_length=1,
        description="Bank symbol, e.g. NABIL. Uses the active ensemble model.",
    )


class InferenceRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "NABIL",
            }
        }
    )

    symbol: str = Field(
        min_length=1,
        description="Bank symbol, e.g. NABIL. Uses the active AutoTFT model in advanced prediction.",
    )
    timeframe: str = Field(default="1d", min_length=1)
    lookback_days: int = Field(default=320, ge=250, le=2000)
    prediction_date: date | None = Field(
        default=None,
        description="Date key for prediction caching. Defaults to current UTC date.",
    )
    policy_rate: float | None = None
    fundamentals: dict[str, FundamentalInput] | None = None


class InferenceSignal(BaseModel):
    bank: str
    date: datetime
    close: float
    prob_direction: float
    prob_momentum: float
    predicted_mag: float
    model_score: float
    signal: str
    direction: str | None = None
    prob_up: float | None = None
    prob_down: float | None = None
    return_magnitude: float | None = None
    return_magnitude_pct: str | None = None
    confidence: str | None = None
    signal_strength: str | None = None
    threshold_used: float | None = None
    nan_features: list[str] | None = None
    car: float | None = None
    npl: float | None = None
    forecast_next_5d: list[ForecastPoint] | None = None
    timeline_10d: list[TimelinePoint] | None = None


class InferenceResponse(BaseModel):
    prediction_id: int
    model_version_id: int = Field(
        description="Resolved active model version used for this prediction."
    )
    model_type: str
    model_target: str
    model_checkpoint: str | None = None
    data_source: str | None = None
    data_source_details: dict[str, Any] | None = None
    prediction_date: date
    from_cache: bool
    symbol: str
    timeframe: str
    lookback_days: int
    generated_at: datetime
    rows_ohlcv: int
    rows_nepse: int
    selected_signal: InferenceSignal
    all_signals: list[InferenceSignal] | None = None
    past_5_days: list[PastOHLCVPoint] | None = None


class ModelVersionCreate(BaseModel):
    model_type: str = Field(min_length=1)
    target: str = Field(min_length=1)
    trained_at: datetime
    train_end_date: date
    n_features: int = Field(gt=0)
    train_auc: float | None = None
    test_auc: float | None = None
    train_r2: float | None = None
    test_r2: float | None = None
    feature_list: str | None = None
    notes: str | None = None
    is_active: bool = False


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_type: str
    target: str
    trained_at: datetime
    train_end_date: date
    n_features: int
    train_auc: float | None
    test_auc: float | None
    train_r2: float | None
    test_r2: float | None
    feature_list: str | None
    notes: str | None
    is_active: bool
    created_at: datetime


class PredictionRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    company_id: int
    model_version_id: int
    prediction_date: date
    predicted_price_5: float | None
    prob_direction_up: float
    prob_momentum_5d: float
    predicted_magnitude: float
    ensemble_score: float
    signal: str
    close_at_signal: float
    actual_close_21d: float | None
    actual_return_21d: float | None
    was_correct: bool | None
    created_at: datetime


class PredictionOutcomeUpdate(BaseModel):
    actual_close_21d: float | None = None
    actual_return_21d: float | None = None
    was_correct: bool | None = None
