import asyncio
import hnswlib
from pydantic import BaseModel, Field, ConfigDict
from .types import JSON, Metric, VectKvId
from typing import Any, Dict, Literal, Optional, cast

_METRIC_TO_SPACE: Dict[str, str] = {"cosine": "cosine", "euclidean": "l2", "ip": "ip"}

class VectorStore(BaseModel):
    metadata: Optional[JSON] = None


class Page(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectors: Dict[VectKvId, VectorStore]
    vector_dim: int
    hnsw_id_to_vectkv_id_map: Dict[int, VectKvId]
    vectkv_id_to_hnsw_id_map: Dict[VectKvId, int]
    HNSW_M: int
    HNSW_EF_CONSTRUCTION: int
    HNSW_MAX_ELEMENT: int
    HNSW_RESIZE_COUNT: int
    HNSW_EF_SEARCH: Optional[int] = None
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, exclude=True)
    next_hnsw_id: int = 0
    metric: Metric = "cosine"
    allow_updates: bool = True
    hnsw_index: Optional[hnswlib.Index] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        space = cast(Literal["l2", "ip", "cosine"], _METRIC_TO_SPACE[self.metric])
        self.hnsw_index = hnswlib.Index(space=space, dim=self.vector_dim)
        self.hnsw_index.init_index(ef_construction=self.HNSW_EF_CONSTRUCTION, M=self.HNSW_M, max_elements=self.HNSW_MAX_ELEMENT)

        if self.HNSW_EF_SEARCH is not None:
            self.hnsw_index.set_ef(self.HNSW_EF_SEARCH)


class VectKvSearchResult(BaseModel):
    vectkv_id: VectKvId
    distance: float
    similarity_score: Optional[float] = None
    metadata: Optional[JSON] = None
