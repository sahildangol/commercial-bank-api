from typing import Optional

from src.db.models.company import Company
from src.db.schema.company import CompanyCreate,CompanyUpdate
from .base import BaseService


class CompanyService(BaseService):
    def create_company(self,company_data:CompanyCreate)->Optional[Company]:
        company=Company(**company_data.model_dump())
        self.session.add(company)
        self.session.commit()
        self.session.refresh(company)
        return company

    def get_company_by_id(self,company_id:int)->Optional[Company]:
        return self.session.query(Company).filter(Company.company_id==company_id).first()

    def get_all_companies(self)->list[Company]:
        return self.session.query(Company).order_by(Company.company_id.asc()).all()

    def update_company(self,company_id:int,company_data:CompanyUpdate)->Optional[Company]:
        company=self.get_company_by_id(company_id=company_id)
        if not company:
            return None

        updated_data=company_data.model_dump(exclude_unset=True)
        for field,value in updated_data.items():
            setattr(company,field,value)

        self.session.commit()
        self.session.refresh(company)
        return company

    def delete_company(self,company_id:int)->bool:
        company=self.get_company_by_id(company_id=company_id)
        if not company:
            return False

        self.session.delete(company)
        self.session.commit()
        return True
