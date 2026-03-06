from fastapi import APIRouter,Depends
from src.db.schema.company import CompanyCreate,CompanyUpdate,CompanyResponse,CompanyDeleteResponse
from src.core.database import get_db
from sqlalchemy.orm import Session
from src.controller.company.company import CompanyController

companyRouter=APIRouter()

@companyRouter.post("/",status_code=201,response_model=CompanyResponse)
def create_company(company_details:CompanyCreate,session:Session=Depends(get_db)):
    try:
        return CompanyController(session=session).create_company(company_details=company_details)
    except Exception as e:
        print(f"Exception Occured:{e}")
        raise e


@companyRouter.get("/",status_code=200,response_model=list[CompanyResponse])
def get_all_companies(session:Session=Depends(get_db)):
    try:
        return CompanyController(session=session).get_all_companies()
    except Exception as e:
        print(f"Exception Occured:{e}")
        raise e


@companyRouter.get("/{company_id}",status_code=200,response_model=CompanyResponse)
def get_company_by_id(company_id:int,session:Session=Depends(get_db)):
    try:
        return CompanyController(session=session).get_company_by_id(company_id=company_id)
    except Exception as e:
        print(f"Exception Occured:{e}")
        raise e


@companyRouter.patch("/{company_id}",status_code=200,response_model=CompanyResponse)
def update_company(company_id:int,company_details:CompanyUpdate,session:Session=Depends(get_db)):
    try:
        return CompanyController(session=session).update_company(company_id=company_id,company_details=company_details)
    except Exception as e:
        print(f"Exception Occured:{e}")
        raise e


@companyRouter.delete("/{company_id}",status_code=200,response_model=CompanyDeleteResponse)
def delete_company(company_id:int,session:Session=Depends(get_db)):
    try:
        return CompanyController(session=session).delete_company(company_id=company_id)
    except Exception as e:
        print(f"Exception Occured:{e}")
        raise e
