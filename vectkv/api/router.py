from enum import Enum
from fastapi import APIRouter, Request, HTTPException
from fastapi import status as HttpStatus
from typing import List, Optional, Union
from .schemas.create_page_schema import CreatePageSchema
from ..exceptions import VectKvException


class VersionRouter(APIRouter):
    def __init__(
        self,
        version: str,
        path: str,
        tags: Optional[List[Union[str, Enum]]] = None,
    ):
        self._validate_version(version)
        self.version = version
        self.prefix = f"/v{version}/{path}"
        super().__init__(prefix=self.prefix, tags=tags)

    def _validate_version(self, version: str) -> None:
        if not version.isdigit() or int(version) <= 0:
            raise ValueError(f"Version must be a string representing a positive integer, got '{version}'")


router = VersionRouter(path="pages", version="1", tags=["Pages"])


@router.post("/", status_code=HttpStatus.HTTP_201_CREATED)
async def create_page(request: Request, body: CreatePageSchema) -> None:
    db = request.state.db

    await db.create_page(
        page_name=body.page_name,
        vector_dim=body.vector_dim,
        metric=body.metric,
        allow_updates=body.allow_updates,
        HNSW_M=body.HNSW_M,
        HNSW_EF_CONSTRUCTION=body.HNSW_EF_CONSTRUCTION,
        HNSW_MAX_ELEMENT=body.HNSW_MAX_ELEMENT,
        HNSW_RESIZE_COUNT=body.HNSW_RESIZE_COUNT,
        HNSW_EF_SEARCH=body.HNSW_EF_SEARCH,
    )