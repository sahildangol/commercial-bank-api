from pydantic import EmailStr,BaseModel
from typing import Union

class UserCreate(BaseModel):
    first_name:str
    last_name:str
    email:EmailStr
    password:str
    is_verified:bool
    is_deleted:bool
    created_at: str
    deleted_at: str
    updated_at: str

class UserCreateResponse(BaseModel):
    id:int
    first_name:str
    last_name:str
    email:EmailStr
    is_verified:bool

class UserPublicResponse(BaseModel):
    id:int
    first_name:str
    last_name:str
    email:EmailStr
    is_verified:bool

class UserUpdate(BaseModel):
    first_name:Union[str,None]=None
    last_name:Union[str,None]=None
    email:Union[EmailStr,None]=None

class UserUpdatePassword(BaseModel):
    email:EmailStr
    old_password:str
    new_password:str

class UserUpdatePasswordResponse(BaseModel):
    message:str

class UserLogin(BaseModel):
    email:EmailStr
    password:str

class UserLoginResponse(BaseModel):
    token:str
