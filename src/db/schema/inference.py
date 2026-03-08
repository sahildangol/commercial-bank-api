from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class FundamentalInput(BaseModel):
    car: float | None = None
    npl: float | None = None


class InferenceSimpleRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "NABIL",
            }
        }
    )

    symbol: str = Field(min_length=1, description="Bank symbol, e.g. NABIL")


class InferenceRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "NABIL",
                "timeframe": "1d",
                "lookback_days": 320,
            }
        }
    )

    symbol: str = Field(min_length=1, description="Bank symbol, e.g. NABIL")
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
    ensemble_score: float
    signal: str
    car: float | None = None
    npl: float | None = None


class InferenceResponse(BaseModel):
    prediction_id: int
    model_version_id: int
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
