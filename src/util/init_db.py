from src.core.database import Base,engine
from src.db.models.user import User

def create_tables():
    Base.metadata.create_all(bind=engine)