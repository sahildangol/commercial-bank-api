from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from decouple import config

POSTGRES_USER=config("POSTGRES_USER")
POSTGRES_PASSWORD=config("POSTGRES_PASSWORD")
POSTGRES_DB=config("POSTGRES_DB",default="postgres")
POSTGRES_HOST=config("POSTGRES_HOST",default="localhost")
POSTGRES_PORT=config("POSTGRES_PORT",default=5432,cast=int)

SQLALCHEMY_DATABASE_URL=(
    f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}'
    f'@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
)

engine=create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base=declarative_base()

def get_db():
    db=SessionLocal()
    try:
        yield db  #session for each request
    finally:
        db.close()
