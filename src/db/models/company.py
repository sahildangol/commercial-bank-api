from src.core.database import Base
from sqlalchemy import Column,Integer,String,Boolean,DateTime,BigInteger,func,text,DDL,event

class Company(Base):
    __tablename__="Company"
    company_id=Column(Integer,primary_key=True)
    symbol=Column(String(50),nullable=False)
    company_name=Column(String(50),nullable=False)
    sector=Column(String(50),nullable=False)
    listed_shares=Column(BigInteger,nullable=False)
    is_active=Column(Boolean,default=False,server_default=text("false"),nullable=False)
    created_at=Column(DateTime(timezone=True),server_default=func.now(),nullable=False)
    updated_at=Column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now(),nullable=False)

event.listen(
    Company.__table__,
    "after_create",
    DDL(
        """
        CREATE OR REPLACE FUNCTION set_company_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    ),
)

event.listen(
    Company.__table__,
    "after_create",
    DDL(
        """
        CREATE TRIGGER trg_company_set_updated_at
        BEFORE UPDATE ON "Company"
        FOR EACH ROW
        EXECUTE FUNCTION set_company_updated_at();
        """
    ),
)
