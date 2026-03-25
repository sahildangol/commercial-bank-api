from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class AdvancedHistoryResponse(BaseModel):
    last_7_days: list[float] = Field(
        default_factory=list,
        description="Last 7 observed close prices used from encoder history.",
    )


class AdvancedConfidenceInterval(BaseModel):
    low: float
    high: float


class AdvancedForecastResponse(BaseModel):
    target_date: date
    predicted_magnitude: float
    confidence_interval: AdvancedConfidenceInterval
    expected_direction: str
    expected_change_pct: float


class AdvancedPredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "NABIL",
                "as_of_date": "2026-03-24",
                "history": {"last_7_days": [542.0, 544.6, 546.2, 547.1, 548.0, 551.5, 553.1]},
                "forecast": {
                    "target_date": "2026-03-29",
                    "predicted_magnitude": 558.2,
                    "confidence_interval": {"low": 548.1, "high": 565.4},
                    "expected_direction": "UP",
                    "expected_change_pct": 2.49,
                },
            }
        }
    )

    symbol: str
    as_of_date: date
    history: AdvancedHistoryResponse
    forecast: AdvancedForecastResponse
