from .base import BaseService
from src.db.schema.user import UserCreate,UserUpdate
from src.db.models.user import User
from sqlalchemy import text
from typing import Optional

class UserService(BaseService):
    def create_user(self,user_data:UserCreate)->Optional[dict]:
        stmt=text(
            """
            INSERT INTO "Users" (first_name,last_name,email,password,is_verified)
            VALUES (:first_name,:last_name,:email,:password,false)
            RETURNING id,first_name,last_name,email,is_verified
            """
        )
        result=self.session.execute(
            stmt,
            user_data.model_dump(exclude_none=True)
        )
        row=result.mappings().one_or_none()
        if row is None:
            self.session.rollback()
            return None
        self.session.commit()
        return dict(row)
    
    def get_user_by_email(self,email:str)->Optional[dict]:
        stmt=text(
            """
            SELECT id,first_name,last_name,email,password,is_verified
            FROM "Users"
            WHERE email = :email AND is_deleted = false
            LIMIT 1
            """
        )
        row=self.session.execute(stmt,{"email":email}).mappings().one_or_none()
        return dict(row) if row else None

    def get_user_by_id(self,user_id:int)->Optional[dict]:
        stmt=text(
            """
            SELECT id,first_name,last_name,email,is_verified
            FROM "Users"
            WHERE id = :user_id AND is_deleted = false
            LIMIT 1
            """
        )
        row=self.session.execute(stmt,{"user_id":user_id}).mappings().one_or_none()
        return dict(row) if row else None

    def update_password(self,user_id:int,new_hashed_password:str)->bool:
        stmt=text(
            """
            UPDATE "Users"
            SET password = :password, updated_at = NOW()
            WHERE id = :user_id AND is_deleted = false
            RETURNING id
            """
        )
        result=self.session.execute(
            stmt,
            {"password":new_hashed_password,"user_id":user_id}
        )
        updated_id=result.scalar_one_or_none()
        if updated_id is None:
            self.session.rollback()
            return False
        self.session.commit()
        return True
        
    def updateUser(self,user_details:UserUpdate,user_id:int):
        updated_data=user_details.model_dump(exclude_unset=True)
        user=self.session.query(User).filter(User.id==user_id,User.is_deleted==False).first()
        if not user:
            return None
        for field,value in updated_data.items():
            setattr(user,field,value)
        self.session.commit()
        self.session.refresh(user)
        return user
