from fastapi import FastAPI,Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.util.init_db import create_tables
from src.routers.auth import authRouter
from src.routers.company import companyRouter
from src.routers.inference import inferenceRouter
from src.routers.market_data import marketDataRouter
from src.routers.user_preference import userPreferenceRouter
from src.util.middleware import get_current_user
from src.db.schema.user import UserPublicResponse
@asynccontextmanager
async def lifespan(app:FastAPI):
    # Initialize DB at Start
    create_tables()
    yield

app=FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=authRouter,tags=["auth"],prefix="/auth")
app.include_router(router=companyRouter,tags=["company"],prefix="/company")
app.include_router(router=inferenceRouter,tags=["inference"],prefix="/inference")
app.include_router(router=marketDataRouter,tags=["market-data"],prefix="/market-data")
app.include_router(router=userPreferenceRouter,tags=["user-preference"],prefix="/user-preference")

@app.get("/health")
def health_check():
    return{"status":"Running..."}


@app.get("/protected")
def read_protected(user:UserPublicResponse=Depends(get_current_user)):
    return{"status":"Yup Protected..."}
