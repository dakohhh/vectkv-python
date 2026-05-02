from ...main import VectKv
from .schemas.add_vector_schema import AddVectorSchema


class VectorsAPIService:

    async def add_vector(self, db: VectKv, add_vector_schema: AddVectorSchema) -> None:
        await db.add_vector(
            page_name=add_vector_schema.page_name,
            vector_id=add_vector_schema.vector_id,
            vector=add_vector_schema.vector,
            metadata=add_vector_schema.metadata,
        )
