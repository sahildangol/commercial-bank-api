from sqlalchemy import create_engine
from decouple import config
from src.core.database import Base
from src.db.models.user import User
from src.db.models.company import Company
from src.db.models.inference import ModelVersion, Prediction
from src.db.models.market_data import TechnicalHistory, FinancialHistory, NepseIndex
from src.db.models.user_preference import UserWatchlist, Alert

def create_tables():
    admin_user     = config("DB_ADMIN_USER",     default=config("POSTGRES_USER"))
    admin_password = config("DB_ADMIN_PASSWORD", default=config("POSTGRES_PASSWORD"))
    db_host = config("POSTGRES_HOST", default="localhost")
    db_port = config("POSTGRES_PORT", default=5432, cast=int)
    db_name = config("POSTGRES_DB",   default="postgres")

    admin_url = (
        f"postgresql://{admin_user}:{admin_password}"
        f"@{db_host}:{db_port}/{db_name}"
    )
    admin_engine = create_engine(admin_url)
    Base.metadata.create_all(bind=admin_engine)
    admin_engine.dispose()
