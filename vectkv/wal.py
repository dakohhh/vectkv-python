"""Write-Ahead Log (WAL) operation models for VectKV.

This module defines the data structures for WAL operations that ensure
durability and crash recovery for vector database operations.
"""

from pydantic import BaseModel
from typing import Optional, List
from .types import VectKvId, Metric, JSON, OperationType


class WALOperation(BaseModel):
    """Base class for all WAL operations.

    Attributes:
        op: The type of operation being performed.
        page_name: The name of the page this operation applies to.
    """

    op: OperationType
    page_name: str


class CreatePageWALOperation(WALOperation):
    """WAL operation for creating a new vector page.

    Attributes:
        vector_dim: The dimensionality of vectors stored in this page.
        metric: The distance metric used for similarity search (e.g., cosine, L2).
        allow_updates: Whether vectors can be updated after insertion.
        HNSW_M: The number of bi-directional links created for each element in HNSW.
        HNSW_EF_CONSTRUCTION: The size of the dynamic candidate list during construction.
        HNSW_MAX_ELEMENT: The maximum number of elements the index can hold.
        HNSW_RESIZE_COUNT: The number of elements to add when resizing the index.
        HNSW_EF_SEARCH: The size of the dynamic candidate list during search.
    """

    vector_dim: int
    metric: Metric

    allow_updates: bool = True

    HNSW_M: int

    HNSW_EF_CONSTRUCTION: int

    HNSW_MAX_ELEMENT: int

    HNSW_RESIZE_COUNT: int

    HNSW_EF_SEARCH: Optional[int] = None


class DeletePageWALOperation(WALOperation):
    """WAL operation for deleting an existing vector page."""


class AddVectorPageWALOperation(WALOperation):
    """WAL operation for adding a vector to a page.

    Attributes:
        vector_id: The unique identifier for the vector.
        vector: The vector data as a list of floats.
        metadata: Optional JSON metadata associated with the vector.
    """

    vector_id: VectKvId

    vector: List[float]

    metadata: Optional[JSON]




async def replay_wal() -> None:
    pass