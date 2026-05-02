from typing import Annotated, List
from ...main import VectKv
from ..response import HttpResponse
from .service import PagesAPIService
from ...schema import VectKvSearchResult
from .schemas.get_page_response_schema import GetPageResponseSchema
from fastapi import status as HttpStatus
from fastapi import APIRouter, Depends, Request
from .schemas.create_page_schema import CreatePageSchema
from .schemas.search_vectors_schema import SearchVectorsSchema
from .schemas.set_ef_search_schema import SetEfSearchSchema

router = APIRouter(prefix="/v1/pages", tags=["Pages"])


@router.post("/", response_model=HttpResponse[None], status_code=HttpStatus.HTTP_201_CREATED)
async def create_page(request: Request, create_page_schema: CreatePageSchema, pages_api_service: Annotated[PagesAPIService, Depends(PagesAPIService)]) -> HttpResponse[None]:
    db: VectKv = request.state.db
    await pages_api_service.create_page(db, create_page_schema)
    return HttpResponse(message="Page created", data=None, status_code=HttpStatus.HTTP_201_CREATED)


@router.get("/", response_model=HttpResponse[List[str]], status_code=HttpStatus.HTTP_200_OK)
async def get_all_pages(request: Request, pages_api_service: Annotated[PagesAPIService, Depends(PagesAPIService)]) -> HttpResponse[List[str]]:
    db: VectKv = request.state.db
    pages = await pages_api_service.get_all_pages(db)
    return HttpResponse(message="Pages retrieved", data=pages, status_code=HttpStatus.HTTP_200_OK)


@router.get("/{page_name}/", response_model=HttpResponse[GetPageResponseSchema], status_code=HttpStatus.HTTP_200_OK)
async def get_page(request: Request, page_name: str, pages_api_service: Annotated[PagesAPIService, Depends(PagesAPIService)]) -> HttpResponse[GetPageResponseSchema]:
    db: VectKv = request.state.db
    page = await pages_api_service.get_page(db, page_name)
    return HttpResponse(message="Page retrieved", data=page, status_code=HttpStatus.HTTP_200_OK)


@router.delete("/{page_name}/", response_model=HttpResponse[None], status_code=HttpStatus.HTTP_200_OK)
async def delete_page(request: Request, page_name: str, pages_api_service: Annotated[PagesAPIService, Depends(PagesAPIService)]) -> HttpResponse[None]:
    db: VectKv = request.state.db
    await pages_api_service.delete_page(db, page_name)
    return HttpResponse(message="Page deleted", data=None, status_code=HttpStatus.HTTP_200_OK)


@router.post("/{page_name}/search/", response_model=HttpResponse[List[VectKvSearchResult]], status_code=HttpStatus.HTTP_200_OK)
async def search_vectors(request: Request, page_name: str, search_vectors_schema: SearchVectorsSchema, pages_api_service: Annotated[PagesAPIService, Depends(PagesAPIService)]) -> HttpResponse[List[VectKvSearchResult]]:
    db: VectKv = request.state.db
    results = await pages_api_service.search_vectors(db, page_name, search_vectors_schema)
    return HttpResponse(message="Search completed", data=results, status_code=HttpStatus.HTTP_200_OK)


@router.patch("/{page_name}/ef-search/", response_model=HttpResponse[None], status_code=HttpStatus.HTTP_200_OK)
async def set_ef_search(request: Request, page_name: str, set_ef_search_schema: SetEfSearchSchema, pages_api_service: Annotated[PagesAPIService, Depends(PagesAPIService)]) -> HttpResponse[None]:
    db: VectKv = request.state.db
    await pages_api_service.set_ef_search(db, page_name, set_ef_search_schema)
    return HttpResponse(message="EF search updated", data=None, status_code=HttpStatus.HTTP_200_OK)
