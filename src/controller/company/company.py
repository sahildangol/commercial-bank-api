from src.service.companyService import CompanyService
from src.db.schema.company import CompanyCreate,CompanyUpdate,CompanyResponse,CompanyDeleteResponse
from sqlalchemy.orm import Session
from fastapi import HTTPException

class CompanyController:
    def __init__(self,session:Session):
        self.__companyService=CompanyService(session=session)

    def create_company(self,company_details:CompanyCreate)->CompanyResponse:
        company=self.__companyService.create_company(company_data=company_details)
        if not company:
            raise HTTPException(status_code=500,detail="Unable to create company")
        return CompanyResponse.model_validate(company)

    def get_company_by_id(self,company_id:int)->CompanyResponse:
        company=self.__companyService.get_company_by_id(company_id=company_id)
        if not company:
            raise HTTPException(status_code=404,detail="Company not found")
        return CompanyResponse.model_validate(company)

    def get_all_companies(self)->list[CompanyResponse]:
        companies=self.__companyService.get_all_companies()
        return [CompanyResponse.model_validate(company) for company in companies]

    def update_company(self,company_id:int,company_details:CompanyUpdate)->CompanyResponse:
        company=self.__companyService.update_company(company_id=company_id,company_data=company_details)
        if not company:
            raise HTTPException(status_code=404,detail="Company not found")
        return CompanyResponse.model_validate(company)

    def delete_company(self,company_id:int)->CompanyDeleteResponse:
        deleted=self.__companyService.delete_company(company_id=company_id)
        if not deleted:
            raise HTTPException(status_code=404,detail="Company not found")
        return CompanyDeleteResponse(message="Company deleted successfully")
