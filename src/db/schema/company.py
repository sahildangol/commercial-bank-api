from datetime import datetime
from pydantic import BaseModel,ConfigDict


class CompanyCreate(BaseModel):
    symbol: str
    company_name: str
    sector: str
    listed_shares: float
    is_active: bool = False


class CompanyUpdate(BaseModel):
    symbol: str | None = None
    company_name: str | None = None
    sector: str | None = None
    listed_shares: float | None = None
    is_active: bool | None = None


class CompanyResponse(BaseModel):
    model_config=ConfigDict(from_attributes=True)

    company_id: int
    symbol: str
    company_name: str
    sector: str
    listed_shares: float
    is_active: bool
    created_at: datetime
    updated_at: datetime


class CompanyDeleteResponse(BaseModel):
    message: str
