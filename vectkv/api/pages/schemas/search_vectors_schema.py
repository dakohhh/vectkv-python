from typing import List
from pydantic import BaseModel, Field


class SearchVectorsSchema(BaseModel):
    query_vector: List[float]
    top_k: int = Field(default=10, gt=0)
