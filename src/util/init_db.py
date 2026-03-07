from src.core.database import Base,engine
from src.db.models.user import User
from src.db.models.company import Company
from src.db.models.inference import ModelVersion, Prediction
from src.db.models.market_data import TechnicalHistory, FinancialHistory, NepseIndex
from src.db.models.user_preference import UserWatchlist, Alert

def create_tables():
    Base.metadata.create_all(bind=engine)
