from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.db.schema.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceSimpleRequest,
    ModelVersionCreate,
    ModelVersionResponse,
    PredictionOutcomeUpdate,
    PredictionRecordResponse,
)
from src.service.inference_service import InferenceService

inferenceRouter = APIRouter()


@inferenceRouter.post("/predict", status_code=200, response_model=InferenceResponse)
def predict_symbol_signal(payload: InferenceSimpleRequest, session: Session = Depends(get_db)):
    request = InferenceRequest(symbol=payload.symbol)
    result = InferenceService(session=session).predict_ensemble(request)
    result.all_signals = None
    return result


@inferenceRouter.post("/predict/advanced", status_code=200, response_model=InferenceResponse)
def predict_symbol_signal_advanced(payload: InferenceRequest, session: Session = Depends(get_db)):
    return InferenceService(session=session).predict_autotft(payload)


@inferenceRouter.get("/supported-symbols", status_code=200, response_model=list[str])
def list_supported_symbols(session: Session = Depends(get_db)):
    return InferenceService(session=session).list_supported_symbols()


@inferenceRouter.post("/model-versions", status_code=201, response_model=ModelVersionResponse)
def create_model_version(payload: ModelVersionCreate, session: Session = Depends(get_db)):
    model_version = InferenceService(session=session).create_model_version(payload)
    return ModelVersionResponse.model_validate(model_version)


@inferenceRouter.get("/model-versions", status_code=200, response_model=list[ModelVersionResponse])
def list_model_versions(session: Session = Depends(get_db)):
    model_versions = InferenceService(session=session).list_model_versions()
    return [ModelVersionResponse.model_validate(row) for row in model_versions]


@inferenceRouter.patch(
    "/model-versions/{model_version_id}/activate",
    status_code=200,
    response_model=ModelVersionResponse,
)
def activate_model_version(model_version_id: int, session: Session = Depends(get_db)):
    model_version = InferenceService(session=session).activate_model_version(model_version_id)
    return ModelVersionResponse.model_validate(model_version)


@inferenceRouter.get("/predictions", status_code=200, response_model=list[PredictionRecordResponse])
def list_predictions(
    symbol: str | None = None,
    prediction_date: date | None = None,
    model_version_id: int | None = None,
    session: Session = Depends(get_db),
):
    return InferenceService(session=session).list_predictions(
        symbol=symbol,
        prediction_date=prediction_date,
        model_version_id=model_version_id,
    )


@inferenceRouter.patch(
    "/predictions/{prediction_id}",
    status_code=200,
    response_model=PredictionRecordResponse,
)
def update_prediction_outcome(
    prediction_id: int,
    payload: PredictionOutcomeUpdate,
    session: Session = Depends(get_db),
):
    return InferenceService(session=session).update_prediction_outcome(
        prediction_id=prediction_id,
        payload=payload,
    )
