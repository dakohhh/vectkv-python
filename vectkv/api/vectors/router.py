from typing import Annotated
from ...main import VectKv
from ..response import HttpResponse
from .service import VectorsAPIService
from fastapi import status as HttpStatus
from fastapi import APIRouter, Depends, Request
from .schemas.add_vector_schema import AddVectorSchema

router = APIRouter(prefix="/v1/vectors", tags=["Vectors"])


@router.post("/", response_model=HttpResponse[None], status_code=HttpStatus.HTTP_201_CREATED)
async def add_vector(request: Request, add_vector_schema: AddVectorSchema, vectors_api_service: Annotated[VectorsAPIService, Depends(VectorsAPIService)]) -> HttpResponse[None]:
    db: VectKv = request.state.db
    await vectors_api_service.add_vector(db, add_vector_schema)
    return HttpResponse(message="Vector added", data=None, status_code=HttpStatus.HTTP_201_CREATED)
