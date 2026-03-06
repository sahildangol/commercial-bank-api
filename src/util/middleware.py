from fastapi import Depends,Header,HTTPException,status
from sqlalchemy.orm import Session
from typing import Annotated,Union
from src.core.auth.authHandler import AuthHandler
from src.service.userService import UserService
from src.core.database import get_db
from src.db.schema.user import UserPublicResponse

AUTH_PREFIX='Bearer '

def get_current_user(session:Session=Depends(get_db),
                     authorization:Annotated[Union[str,None],Header()]=None
                     )->UserPublicResponse:
    auth_exception=HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid Authentication Credentials"
    )
    if not authorization:
        raise auth_exception
    if not authorization.startswith(AUTH_PREFIX):
        raise auth_exception
    payload=AuthHandler.decode_jwt(token=authorization[len(AUTH_PREFIX):])
    if payload and payload["user_id"]:
        try:
            user=UserService(session=session).get_user_by_id(payload['user_id'])
            return UserPublicResponse(
                id=user["id"],
                first_name=user["first_name"],
                last_name=user["last_name"],
                email=user["email"],
                is_verified=user["is_verified"]
            )
        except Exception as e:
            print(f"Error:{e}")
            raise e
    raise auth_exception
    
    
