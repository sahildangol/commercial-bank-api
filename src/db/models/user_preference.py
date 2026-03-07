from src.core.database import Base
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)


class UserWatchlist(Base):
    __tablename__ = "UserWatchlist"
    __table_args__ = (
        UniqueConstraint("user_id", "company_id", name="uq_watchlist_user_company"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("Users.id"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("Company.company_id"), nullable=False, index=True)
    note = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, server_default=text("true"), nullable=False)
    added_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    removed_at = Column(DateTime(timezone=True), nullable=True)


class Alert(Base):
    __tablename__ = "Alert"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("Users.id"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("Company.company_id"), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    threshold = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True, server_default=text("true"), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
