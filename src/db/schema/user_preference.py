from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserWatchlistCreate(BaseModel):
    user_id: int
    company_id: int
    note: str | None = None
    is_active: bool = True


class UserWatchlistUpdate(BaseModel):
    note: str | None = None
    is_active: bool | None = None
    removed_at: datetime | None = None


class UserWatchlistResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    company_id: int
    note: str | None
    is_active: bool
    added_at: datetime
    removed_at: datetime | None


class AlertCreate(BaseModel):
    user_id: int
    company_id: int
    alert_type: str = Field(min_length=1)
    threshold: float
    is_active: bool = True


class AlertUpdate(BaseModel):
    alert_type: str | None = None
    threshold: float | None = None
    is_active: bool | None = None
    deleted_at: datetime | None = None


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    company_id: int
    alert_type: str
    threshold: float
    is_active: bool
    deleted_at: datetime | None
    created_at: datetime
