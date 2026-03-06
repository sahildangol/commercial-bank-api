from src.service.userService import UserService
from src.db.schema.user import UserCreate,UserCreateResponse,UserLogin,UserLoginResponse,UserUpdatePassword,UserUpdatePasswordResponse
from src.core.auth.hashHandler import HashHelper
from src.core.auth.authHandler import AuthHandler
from sqlalchemy.orm import Session
from fastapi import HTTPException

class UserController:
    def __init__(self,session:Session):
        self.__userService=UserService(session=session)
        
    def signup(self,user_details:UserCreate)->UserCreateResponse:
        if self.__userService.get_user_by_email(email=user_details.email):
            raise HTTPException(status_code=400,detail="Account Already Exist.Try Log in.")
        hashed_password=HashHelper.get_password_hash(password=user_details.password)
        user_details.password=hashed_password
        user=self.__userService.create_user(user_data=user_details)
        if not user:
            raise HTTPException(status_code=500,detail="Unable to create user")
        return UserCreateResponse(**user)
    
    def login(self,login_details:UserLogin)->UserLoginResponse:
        user=self.__userService.get_user_by_email(email=login_details.email)
        if not user:
            raise HTTPException(status_code=400,detail="Account Not Found. Try Sign In.")

        if HashHelper.verify_password(password=login_details.password,hashedPassword=user["password"]):
            token=AuthHandler.sign_jwt(user_id=user["id"])
            if token:
                return UserLoginResponse(token=token)
            raise HTTPException(status_code=500,detail="Internal Server Error")
        raise HTTPException(status_code=400,detail="Invalid Credentials")
    
    def get_user_by_id(self,user_id:int):
        user=self.__userService.get_user_by_id(user_id=user_id)
        if user:
            return user
        raise HTTPException(status_code=400,detail="User is Not Available")
    
     
    def changePassword(self,change_password_details:UserUpdatePassword)->UserUpdatePasswordResponse:
        user=self.__userService.get_user_by_email(email=change_password_details.email)
        if not user:
            raise HTTPException(status_code=400,detail="Account Not Found.")

        if HashHelper.verify_password(password=change_password_details.old_password,hashedPassword=user["password"]):
            hashed_password=HashHelper.get_password_hash(password=change_password_details.new_password)
            updated=self.__userService.update_password(user_id=user["id"],new_hashed_password=hashed_password)
            if not updated:
                raise HTTPException(status_code=404,detail="User is Not Available")
            return UserUpdatePasswordResponse(message="Password updated successfully")
        raise HTTPException(status_code=400,detail="Invalid Credentials")
