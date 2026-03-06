from src.core.database import Base,engine
from src.db.models.user import User
from src.db.models.company import Company

def create_tables():
    Base.metadata.create_all(bind=engine)
