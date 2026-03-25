from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from decouple import config
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db.schema.user import UserPublicResponse
from src.routers.auth import authRouter
from src.routers.company import companyRouter
from src.routers.inference import inferenceRouter
from src.routers.market_data import marketDataRouter
from src.routers.tft_advanced import tftAdvancedRouter
from src.routers.user_preference import userPreferenceRouter
from src.util.init_db import create_tables
from src.util.middleware import get_current_user

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    fail_hard = config("DB_INIT_FAIL_HARD", default=True, cast=bool)
    try:
        create_tables()
    except Exception:
        logger.exception("Database initialization failed during app startup.")
        if fail_hard:
            raise
        logger.warning(
            "Continuing startup because DB_INIT_FAIL_HARD=false. "
            "Database-backed endpoints may fail until the database is reachable."
        )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=authRouter, tags=["auth"], prefix="/auth")
app.include_router(router=companyRouter, tags=["company"], prefix="/company")
app.include_router(router=inferenceRouter, tags=["inference"], prefix="/inference")
app.include_router(router=marketDataRouter, tags=["market-data"], prefix="/market-data")
app.include_router(router=userPreferenceRouter, tags=["user-preference"], prefix="/user-preference")
app.include_router(router=tftAdvancedRouter, tags=["tft-advanced"])

@app.get("/health")
def health_check():
    return {"status": "Running..."}


@app.get("/protected")
def read_protected(user: UserPublicResponse = Depends(get_current_user)):
    return {"status": "Yup Protected..."}
