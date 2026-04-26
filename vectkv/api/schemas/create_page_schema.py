from typing import Optional
from ...types import Metric
from pydantic import BaseModel, Field


class CreatePageSchema(BaseModel):
    page_name: str
    vector_dim: int = Field(gt=0)
    metric: Metric = "cosine"
    allow_updates: bool = True
    HNSW_M: Optional[int] = Field(default=16, gt=0)
    HNSW_EF_CONSTRUCTION: Optional[int] = Field(default=200, gt=0)
    HNSW_MAX_ELEMENT: Optional[int] = Field(default=10000, gt=0)
    HNSW_RESIZE_COUNT: Optional[int] = Field(default=10000, gt=0)
    HNSW_EF_SEARCH: Optional[int] = Field(default=None, gt=0)
