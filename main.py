from fastapi import FastAPI,Depends
from contextlib import asynccontextmanager
from src.util.init_db import create_tables
from src.routers.auth import authRouter
from src.util.middleware import get_current_user
from src.db.schema.user import UserPublicResponse
from fastapi.middleware.cors import CORSMiddleware
@asynccontextmanager
async def lifespan(app:FastAPI):
    # Initialize DB at Start
    create_tables()
    yield

origins = ["http://localhost:5173",]

app=FastAPI(lifespan=lifespan)
app.include_router(router=authRouter,tags=["auth"],prefix="/auth")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return{"status":"Running..."}


@app.get("/protected")
def read_protected(user:UserPublicResponse=Depends(get_current_user)):
    return{"status":"Yup Protected..."}
