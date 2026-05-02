from typing import Optional
from pydantic import BaseModel
from ....types import Metric


class GetPageResponseSchema(BaseModel):
    vector_dim: int
    metric: Metric
    allow_updates: bool
    HNSW_M: int
    HNSW_EF_CONSTRUCTION: int
    HNSW_MAX_ELEMENT: int
    HNSW_RESIZE_COUNT: int
    HNSW_EF_SEARCH: Optional[int] = None
