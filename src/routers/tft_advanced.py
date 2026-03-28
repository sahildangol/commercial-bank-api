from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.db.schema.inference import InferenceRequest, InferenceResponse
from src.db.schema.tft_advanced import (
    AdvancedConfidenceInterval,
    AdvancedForecastResponse,
    AdvancedHistoryResponse,
    AdvancedPredictionResponse,
)
from src.service.inference_service import InferenceService

tftAdvancedRouter = APIRouter()


def _to_advanced_response(result: InferenceResponse) -> AdvancedPredictionResponse:
    history_points = result.past_5_days or []
    history = [float(point.close) for point in history_points]

    forecast_points = result.selected_signal.forecast_next_5d or []
    if forecast_points:
        target_point = forecast_points[-1]
        predicted_close = float(target_point.predicted_close)
        target_date = target_point.forecast_date.date()
        low_bound = min(float(point.predicted_close) for point in forecast_points)
        high_bound = max(float(point.predicted_close) for point in forecast_points)
    else:
        predicted_close = float(result.selected_signal.close)
        target_date = result.selected_signal.date.date()
        low_bound = predicted_close
        high_bound = predicted_close

    base_close = float(result.selected_signal.close)
    expected_change_pct = ((predicted_close - base_close) / base_close) * 100.0 if base_close else 0.0

    return AdvancedPredictionResponse(
        symbol=result.symbol,
        as_of_date=result.selected_signal.date.date(),
        history=AdvancedHistoryResponse(last_7_days=history),
        forecast=AdvancedForecastResponse(
            target_date=target_date,
            predicted_magnitude=predicted_close,
            confidence_interval=AdvancedConfidenceInterval(low=low_bound, high=high_bound),
            expected_direction=result.selected_signal.signal,
            expected_change_pct=float(expected_change_pct),
        ),
    )


@tftAdvancedRouter.post(
    "/predict/advanced/{symbol}",
    status_code=200,
    response_model=AdvancedPredictionResponse,
)
def predict_advanced(symbol: str, session: Session = Depends(get_db)):
    try:
        inference_result = InferenceService(session=session).predict_autotft(
            InferenceRequest(symbol=symbol)
        )
        return _to_advanced_response(inference_result)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected inference failure: {exc}") from exc
