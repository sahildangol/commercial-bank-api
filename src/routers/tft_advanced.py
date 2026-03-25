from fastapi import APIRouter, HTTPException

from src.db.schema.tft_advanced import AdvancedPredictionResponse
from src.service.tft_advanced_service import (
    DataFetchError,
    DataIntegrityError,
    InferenceError,
    InsufficientDataError,
    SymbolNotFoundError,
    advanced_tft_service,
)

tftAdvancedRouter = APIRouter()


@tftAdvancedRouter.post(
    "/predict/advanced/{symbol}",
    status_code=200,
    response_model=AdvancedPredictionResponse,
)
def predict_advanced(symbol: str):
    try:
        return advanced_tft_service.predict(symbol)
    except SymbolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataIntegrityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DataFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected inference failure: {exc}") from exc
