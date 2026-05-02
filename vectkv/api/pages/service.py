from typing import List
from ...main import VectKv
from ...schema import VectKvSearchResult
from .schemas.get_page_response_schema import GetPageResponseSchema
from .schemas.create_page_schema import CreatePageSchema
from .schemas.search_vectors_schema import SearchVectorsSchema
from .schemas.set_ef_search_schema import SetEfSearchSchema


class PagesAPIService:

    async def create_page(self, db: VectKv, create_page_schema: CreatePageSchema) -> None:
        await db.create_page(
            page_name=create_page_schema.page_name,
            vector_dim=create_page_schema.vector_dim,
            metric=create_page_schema.metric,
            allow_updates=create_page_schema.allow_updates,
            HNSW_M=create_page_schema.HNSW_M,
            HNSW_EF_CONSTRUCTION=create_page_schema.HNSW_EF_CONSTRUCTION,
            HNSW_MAX_ELEMENT=create_page_schema.HNSW_MAX_ELEMENT,
            HNSW_RESIZE_COUNT=create_page_schema.HNSW_RESIZE_COUNT,
            HNSW_EF_SEARCH=create_page_schema.HNSW_EF_SEARCH,
        )

    async def get_page(self, db: VectKv, page_name: str) -> GetPageResponseSchema:
        page = await db.get_page(page_name=page_name)
        return GetPageResponseSchema(
            vector_dim=page.vector_dim,
            metric=page.metric,
            allow_updates=page.allow_updates,
            HNSW_M=page.HNSW_M,
            HNSW_EF_CONSTRUCTION=page.HNSW_EF_CONSTRUCTION,
            HNSW_MAX_ELEMENT=page.HNSW_MAX_ELEMENT,
            HNSW_RESIZE_COUNT=page.HNSW_RESIZE_COUNT,
            HNSW_EF_SEARCH=page.HNSW_EF_SEARCH,
        )

    async def delete_page(self, db: VectKv, page_name: str) -> None:
        await db.delete_page(page_name=page_name)

    async def get_all_pages(self, db: VectKv) -> List[str]:
        return await db.get_all_pages()

    async def search_vectors(self, db: VectKv, page_name: str, search_vectors_schema: SearchVectorsSchema) -> List[VectKvSearchResult]:
        return await db.search_vectors(
            page_name=page_name,
            query_vector=search_vectors_schema.query_vector,
            top_k=search_vectors_schema.top_k,
        )

    async def set_ef_search(self, db: VectKv, page_name: str, set_ef_search_schema: SetEfSearchSchema) -> None:
        await db.set_ef_search(page_name=page_name, ef_search=set_ef_search_schema.ef_search)
