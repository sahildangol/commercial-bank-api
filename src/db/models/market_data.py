from src.core.database import Base
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    UniqueConstraint,
    func,
)


class TechnicalHistory(Base):
    __tablename__ = "TechnicalHistory"
    __table_args__ = (
        UniqueConstraint("company_id", "date", name="uq_technicalhistory_company_date"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("Company.company_id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    amount = Column(Float, nullable=True)
    per_change = Column(Float, nullable=True)


class FinancialHistory(Base):
    __tablename__ = "FinancialHistory"
    __table_args__ = (
        UniqueConstraint("company_id", "fiscal_year", name="uq_financialhistory_company_fiscal_year"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("Company.company_id"), nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    report_date = Column(Date, nullable=False)
    car = Column(Float, nullable=True)
    npl = Column(Float, nullable=True)
    roe = Column(Float, nullable=True)
    pb = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class NepseIndex(Base):
    __tablename__ = "NepseIndex"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)
