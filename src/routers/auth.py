from fastapi import APIRouter,Depends
from src.db.schema.user import UserCreate,UserLogin,UserCreateResponse,UserLoginResponse,UserUpdatePassword,UserUpdatePasswordResponse,UserUpdate
from src.core.database import get_db
from sqlalchemy.orm import Session
from src.controller.auth.userController import UserController

authRouter=APIRouter()

@authRouter.post("/login",status_code=200,response_model=UserLoginResponse)
def login(loginDetails:UserLogin,session:Session=Depends(get_db)):
    try:
        return UserController(session=session).login(login_details=loginDetails)
    except Exception as e:
        print(f"Exception Occured:{e}")
        raise e

@authRouter.post("/signup",status_code=201,response_model=UserCreateResponse)
def signup(signUpDetails:UserCreate,session:Session=Depends(get_db)):
    try:
        return UserController(session=session).signup(user_details=signUpDetails)
    except Exception as e:
        print(f"Exception Occured at {e}")
        raise e
    
@authRouter.post("/change-password",status_code=200,response_model=UserUpdatePasswordResponse)
def changePassword(change_password_details:UserUpdatePassword,session:Session=Depends(get_db)):
    try:
        return UserController(session=session).changePassword(change_password_details=change_password_details)
    except Exception as e:
        print(f"Exception Occured at {e}")
        raise e
    
@authRouter.patch("/update-user/{id}",status_code=200,response_model=UserUpdatePasswordResponse)
def upateUser(id:int,user_details:UserUpdate,session:Session=Depends(get_db)):
    try:
        return UserController(session=session).updateUser(user_details=user_details,user_id=id)
    except Exception as e:
        print(f"Exception Occured at {e}")
        raise e
