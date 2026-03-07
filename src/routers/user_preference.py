from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.db.models.company import Company
from src.db.models.user import User
from src.db.models.user_preference import Alert, UserWatchlist
from src.db.schema.user_preference import (
    AlertCreate,
    AlertResponse,
    AlertUpdate,
    UserWatchlistCreate,
    UserWatchlistResponse,
    UserWatchlistUpdate,
)

userPreferenceRouter = APIRouter()


def _ensure_user_exists(session: Session, user_id: int) -> None:
    exists = session.query(User).filter(User.id == user_id).first()
    if not exists:
        raise HTTPException(status_code=404, detail="User not found.")


def _ensure_company_exists(session: Session, company_id: int) -> None:
    exists = session.query(Company).filter(Company.company_id == company_id).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Company not found.")


@userPreferenceRouter.post("/watchlist", status_code=201, response_model=UserWatchlistResponse)
def create_watchlist(payload: UserWatchlistCreate, session: Session = Depends(get_db)):
    _ensure_user_exists(session, payload.user_id)
    _ensure_company_exists(session, payload.company_id)

    row = UserWatchlist(**payload.model_dump())
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Watchlist already exists for this user and company.")
    session.refresh(row)
    return UserWatchlistResponse.model_validate(row)


@userPreferenceRouter.get("/watchlist", status_code=200, response_model=list[UserWatchlistResponse])
def list_watchlist(
    user_id: int | None = None,
    company_id: int | None = None,
    is_active: bool | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    query = session.query(UserWatchlist)
    if user_id is not None:
        query = query.filter(UserWatchlist.user_id == user_id)
    if company_id is not None:
        query = query.filter(UserWatchlist.company_id == company_id)
    if is_active is not None:
        query = query.filter(UserWatchlist.is_active == is_active)

    rows = query.order_by(UserWatchlist.id.desc()).limit(limit).all()
    return [UserWatchlistResponse.model_validate(row) for row in rows]


@userPreferenceRouter.patch("/watchlist/{watchlist_id}", status_code=200, response_model=UserWatchlistResponse)
def update_watchlist(
    watchlist_id: int,
    payload: UserWatchlistUpdate,
    session: Session = Depends(get_db),
):
    row = session.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Watchlist not found.")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    for key, value in updates.items():
        setattr(row, key, value)

    if row.is_active and "removed_at" not in updates:
        row.removed_at = None

    session.commit()
    session.refresh(row)
    return UserWatchlistResponse.model_validate(row)


@userPreferenceRouter.delete("/watchlist/{watchlist_id}", status_code=200)
def remove_watchlist(watchlist_id: int, session: Session = Depends(get_db)):
    row = session.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Watchlist not found.")
    row.is_active = False
    row.removed_at = datetime.utcnow()
    session.commit()
    return {"message": "Watchlist removed successfully."}


@userPreferenceRouter.post("/alerts", status_code=201, response_model=AlertResponse)
def create_alert(payload: AlertCreate, session: Session = Depends(get_db)):
    _ensure_user_exists(session, payload.user_id)
    _ensure_company_exists(session, payload.company_id)

    row = Alert(**payload.model_dump())
    session.add(row)
    session.commit()
    session.refresh(row)
    return AlertResponse.model_validate(row)


@userPreferenceRouter.get("/alerts", status_code=200, response_model=list[AlertResponse])
def list_alerts(
    user_id: int | None = None,
    company_id: int | None = None,
    is_active: bool | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    query = session.query(Alert)
    if user_id is not None:
        query = query.filter(Alert.user_id == user_id)
    if company_id is not None:
        query = query.filter(Alert.company_id == company_id)
    if is_active is not None:
        query = query.filter(Alert.is_active == is_active)
    rows = query.order_by(Alert.id.desc()).limit(limit).all()
    return [AlertResponse.model_validate(row) for row in rows]


@userPreferenceRouter.patch("/alerts/{alert_id}", status_code=200, response_model=AlertResponse)
def update_alert(alert_id: int, payload: AlertUpdate, session: Session = Depends(get_db)):
    row = session.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found.")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    for key, value in updates.items():
        setattr(row, key, value)

    if row.is_active and "deleted_at" not in updates:
        row.deleted_at = None

    session.commit()
    session.refresh(row)
    return AlertResponse.model_validate(row)


@userPreferenceRouter.delete("/alerts/{alert_id}", status_code=200)
def delete_alert(alert_id: int, session: Session = Depends(get_db)):
    row = session.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found.")
    row.is_active = False
    row.deleted_at = datetime.utcnow()
    session.commit()
    return {"message": "Alert deleted successfully."}
