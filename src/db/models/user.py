from src.core.database import Base
from sqlalchemy import Column,Integer,String,Boolean,DateTime,func,text,DDL,event

class User(Base):
    __tablename__="Users"
    id=Column(Integer,primary_key=True)
    first_name=Column(String(50))
    last_name=Column(String(50))
    email=Column(String(50),unique=True)
    password=Column(String(250))
    is_verified=Column(Boolean,default=False,server_default=text("false"),nullable=False)
    is_deleted=Column(Boolean,default=False,server_default=text("false"),nullable=False)
    created_at=Column(DateTime(timezone=True),server_default=func.now(),nullable=False)
    updated_at=Column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now(),nullable=False)
    deleted_at=Column(DateTime(timezone=True),nullable=True)


event.listen(
    User.__table__,
    "after_create",
    DDL(
        """
        CREATE OR REPLACE FUNCTION set_users_updated_at()
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
    User.__table__,
    "after_create",
    DDL(
        """
        CREATE TRIGGER trg_users_set_updated_at
        BEFORE UPDATE ON "Users"
        FOR EACH ROW
        EXECUTE FUNCTION set_users_updated_at();
        """
    ),
)
