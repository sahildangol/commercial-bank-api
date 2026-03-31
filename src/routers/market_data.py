from datetime import date

#for docker comm
import httpx
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.db.models.company import Company
from src.db.models.market_data import FinancialHistory, NepseIndex, TechnicalHistory
from src.db.schema.market_data import (
    FinancialHistoryCreate,
    FinancialHistoryResponse,
    FinancialHistoryUpdate,
    NepseIndexCreate,
    NepseIndexResponse,
    NepseIndexUpdate,
    TechnicalHistoryCreate,
    TechnicalHistoryResponse,
    TechnicalHistoryUpdate,
)

marketDataRouter = APIRouter()


def _ensure_company_exists(session: Session, company_id: int) -> None:
    exists = session.query(Company).filter(Company.company_id == company_id).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Company not found.")


@marketDataRouter.post("/technical-history", status_code=201, response_model=TechnicalHistoryResponse)
def create_technical_history(payload: TechnicalHistoryCreate, session: Session = Depends(get_db)):
    _ensure_company_exists(session, payload.company_id)
    row = TechnicalHistory(**payload.model_dump())
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="Technical history already exists for this company and date.",
        )
    session.refresh(row)
    return TechnicalHistoryResponse.model_validate(row)


@marketDataRouter.get("/technical-history", status_code=200, response_model=list[TechnicalHistoryResponse])
def list_technical_history(
    company_id: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    query = session.query(TechnicalHistory)
    if company_id is not None:
        query = query.filter(TechnicalHistory.company_id == company_id)
    if date_from is not None:
        query = query.filter(TechnicalHistory.date >= date_from)
    if date_to is not None:
        query = query.filter(TechnicalHistory.date <= date_to)
    rows = (
        query.order_by(TechnicalHistory.date.desc(), TechnicalHistory.id.desc())
        .limit(limit)
        .all()
    )
    return [TechnicalHistoryResponse.model_validate(row) for row in rows]


@marketDataRouter.get("/technical-history/{history_id}", status_code=200, response_model=TechnicalHistoryResponse)
def get_technical_history(history_id: int, session: Session = Depends(get_db)):
    row = session.query(TechnicalHistory).filter(TechnicalHistory.id == history_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Technical history not found.")
    return TechnicalHistoryResponse.model_validate(row)


@marketDataRouter.patch("/technical-history/{history_id}", status_code=200, response_model=TechnicalHistoryResponse)
def update_technical_history(
    history_id: int,
    payload: TechnicalHistoryUpdate,
    session: Session = Depends(get_db),
):
    row = session.query(TechnicalHistory).filter(TechnicalHistory.id == history_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Technical history not found.")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    for key, value in updates.items():
        setattr(row, key, value)

    session.commit()
    session.refresh(row)
    return TechnicalHistoryResponse.model_validate(row)


@marketDataRouter.delete("/technical-history/{history_id}", status_code=200)
def delete_technical_history(history_id: int, session: Session = Depends(get_db)):
    row = session.query(TechnicalHistory).filter(TechnicalHistory.id == history_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Technical history not found.")
    session.delete(row)
    session.commit()
    return {"message": "Technical history deleted successfully."}


@marketDataRouter.post("/financial-history", status_code=201, response_model=FinancialHistoryResponse)
def create_financial_history(payload: FinancialHistoryCreate, session: Session = Depends(get_db)):
    _ensure_company_exists(session, payload.company_id)
    row = FinancialHistory(**payload.model_dump())
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="Financial history already exists for this company and fiscal year.",
        )
    session.refresh(row)
    return FinancialHistoryResponse.model_validate(row)


@marketDataRouter.get("/financial-history", status_code=200, response_model=list[FinancialHistoryResponse])
def list_financial_history(
    company_id: int | None = None,
    fiscal_year: int | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    query = session.query(FinancialHistory)
    if company_id is not None:
        query = query.filter(FinancialHistory.company_id == company_id)
    if fiscal_year is not None:
        query = query.filter(FinancialHistory.fiscal_year == fiscal_year)
    rows = (
        query.order_by(FinancialHistory.fiscal_year.desc(), FinancialHistory.id.desc())
        .limit(limit)
        .all()
    )
    return [FinancialHistoryResponse.model_validate(row) for row in rows]


@marketDataRouter.get("/financial-history/{history_id}", status_code=200, response_model=FinancialHistoryResponse)
def get_financial_history(history_id: int, session: Session = Depends(get_db)):
    row = session.query(FinancialHistory).filter(FinancialHistory.id == history_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Financial history not found.")
    return FinancialHistoryResponse.model_validate(row)


@marketDataRouter.patch("/financial-history/{history_id}", status_code=200, response_model=FinancialHistoryResponse)
def update_financial_history(
    history_id: int,
    payload: FinancialHistoryUpdate,
    session: Session = Depends(get_db),
):
    row = session.query(FinancialHistory).filter(FinancialHistory.id == history_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Financial history not found.")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    for key, value in updates.items():
        setattr(row, key, value)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="Financial history with this fiscal year already exists for the company.",
        )
    session.refresh(row)
    return FinancialHistoryResponse.model_validate(row)


@marketDataRouter.delete("/financial-history/{history_id}", status_code=200)
def delete_financial_history(history_id: int, session: Session = Depends(get_db)):
    row = session.query(FinancialHistory).filter(FinancialHistory.id == history_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Financial history not found.")
    session.delete(row)
    session.commit()
    return {"message": "Financial history deleted successfully."}


@marketDataRouter.post("/nepse-index", status_code=201, response_model=NepseIndexResponse)
def create_nepse_index(payload: NepseIndexCreate, session: Session = Depends(get_db)):
    row = NepseIndex(**payload.model_dump())
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="NEPSE index already exists for this date.")
    session.refresh(row)
    return NepseIndexResponse.model_validate(row)


@marketDataRouter.get("/nepse-index", status_code=200, response_model=list[NepseIndexResponse])
def list_nepse_index(
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    query = session.query(NepseIndex)
    if date_from is not None:
        query = query.filter(NepseIndex.date >= date_from)
    if date_to is not None:
        query = query.filter(NepseIndex.date <= date_to)
    rows = query.order_by(NepseIndex.date.desc(), NepseIndex.id.desc()).limit(limit).all()
    return [NepseIndexResponse.model_validate(row) for row in rows]


@marketDataRouter.get("/nepse-index/{nepse_id}", status_code=200, response_model=NepseIndexResponse)
def get_nepse_index(nepse_id: int, session: Session = Depends(get_db)):
    row = session.query(NepseIndex).filter(NepseIndex.id == nepse_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="NEPSE index row not found.")
    return NepseIndexResponse.model_validate(row)


@marketDataRouter.patch("/nepse-index/{nepse_id}", status_code=200, response_model=NepseIndexResponse)
def update_nepse_index(nepse_id: int, payload: NepseIndexUpdate, session: Session = Depends(get_db)):
    row = session.query(NepseIndex).filter(NepseIndex.id == nepse_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="NEPSE index row not found.")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    for key, value in updates.items():
        setattr(row, key, value)

    session.commit()
    session.refresh(row)
    return NepseIndexResponse.model_validate(row)


@marketDataRouter.delete("/nepse-index/{nepse_id}", status_code=200)
def delete_nepse_index(nepse_id: int, session: Session = Depends(get_db)):
    row = session.query(NepseIndex).filter(NepseIndex.id == nepse_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="NEPSE index row not found.")
    session.delete(row)
    session.commit()
    return {"message": "NEPSE index row deleted successfully."}

#tempory docker comm for front
NEPSE_DOCKER_URL = "http://nepseapi:8000"


@marketDataRouter.get("/live-overview", status_code=200)
async def get_live_market_overview():
    async with httpx.AsyncClient() as client:
        try:
            index_task = client.get(f"{NEPSE_DOCKER_URL}/NepseIndex", timeout=5.0)
            gainers_task = client.get(f"{NEPSE_DOCKER_URL}/TopGainers", timeout=5.0)
            losers_task = client.get(f"{NEPSE_DOCKER_URL}/TopLosers", timeout=5.0)

            responses = await asyncio.gather(index_task, gainers_task, losers_task)

            return {
                "index": responses[0].json(),
                "TopGainers": responses[1].json(),
                "TopLosers": responses[2].json()
            }
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="NEPSE service unavailable")


@marketDataRouter.get("/live-full", status_code=200)
async def get_live_full_market():
    async with httpx.AsyncClient() as client:
        try:
            # Fetching from the 8000 container internally
            response = await client.get(f"{NEPSE_DOCKER_URL}/LiveMarket", timeout=10.0)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Error from NEPSE service")
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="NEPSE Docker service unreachable on port 8000")


@marketDataRouter.get("/summary", status_code=200)
async def get_market_summary():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{NEPSE_DOCKER_URL}/Summary", timeout=5.0)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Error from NEPSE service")
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="NEPSE Docker service unreachable on port 8000")


@marketDataRouter.get("/is-nepse-open", status_code=200)
async def get_is_nepse_open():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{NEPSE_DOCKER_URL}/IsNepseOpen", timeout=5.0)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Error from NEPSE service")
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="NEPSE Docker service unreachable on port 8000")
