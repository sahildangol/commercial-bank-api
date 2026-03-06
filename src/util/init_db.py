from src.core.database import Base,engine
from src.db.models import user

def create_tables():
    Base.metadata.create_all(bind=engine)