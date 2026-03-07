from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class TechnicalHistoryCreate(BaseModel):
    company_id: int
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = None
    per_change: float | None = None


class TechnicalHistoryUpdate(BaseModel):
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    per_change: float | None = None


class TechnicalHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    company_id: int
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None
    per_change: float | None


class FinancialHistoryCreate(BaseModel):
    company_id: int
    fiscal_year: int
    report_date: date
    car: float | None = None
    npl: float | None = None
    roe: float | None = None
    pb: float | None = None
    eps: float | None = None


class FinancialHistoryUpdate(BaseModel):
    fiscal_year: int | None = None
    report_date: date | None = None
    car: float | None = None
    npl: float | None = None
    roe: float | None = None
    pb: float | None = None
    eps: float | None = None


class FinancialHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    company_id: int
    fiscal_year: int
    report_date: date
    car: float | None
    npl: float | None
    roe: float | None
    pb: float | None
    eps: float | None
    created_at: datetime


class NepseIndexCreate(BaseModel):
    date: date
    close: float
    volume: float | None = None
    turnover: float | None = None


class NepseIndexUpdate(BaseModel):
    close: float | None = None
    volume: float | None = None
    turnover: float | None = None


class NepseIndexResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    close: float
    volume: float | None
    turnover: float | None
