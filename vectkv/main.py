import os
import json
import asyncio
import hnswlib
import aiofiles
import numpy as np
from pathlib import Path
from types import TracebackType
from  dataclasses import dataclass, field
from .types import JSON, Metric, VectKvId, OperationType
from typing import Dict, List, Literal, cast, Optional, Self
from .wal import CreatePageWALOperation, DeletePageWALOperation, AddVectorPageWALOperation
from .exceptions import VectKvException, PageVectorDimException, VectKvIdAlreadyExistException

@dataclass
class VectorStore:
    # vector: List[float]
    metadata: Optional[JSON] = None

_METRIC_TO_SPACE: Dict[str, str] = {"cosine": "cosine", "euclidean": "l2", "ip": "ip"}


BASE_DIR: Path = Path(__file__).resolve().parent.parent

@dataclass
class Page:
    vectors: Dict[VectKvId, VectorStore]

    vector_dim: int

    hnsw_id_to_vectkv_id_map: Dict[int, VectKvId]

    vectkv_id_to_hnsw_id_map: Dict[VectKvId, int]

    HNSW_M: int

    HNSW_EF_CONSTRUCTION: int

    HNSW_MAX_ELEMENT: int

    HNSW_RESIZE_COUNT: int

    HNSW_EF_SEARCH: Optional[int] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    next_hnsw_id: int = field(default=0)

    metric: Metric = "cosine"

    allow_updates: bool = True

    hnsw_index: hnswlib.Index = field(init=False)

    def __post_init__(self) -> None:
        space = cast(Literal["l2", "ip", "cosine"], _METRIC_TO_SPACE[self.metric])
        self.hnsw_index = hnswlib.Index(space=space, dim=self.vector_dim)
        self.hnsw_index.init_index(ef_construction=self.HNSW_EF_CONSTRUCTION, M=self.HNSW_M, max_elements=self.HNSW_MAX_ELEMENT)

        if self.HNSW_EF_SEARCH is not None:
            self.hnsw_index.set_ef(self.HNSW_EF_SEARCH)

@dataclass
class VectKvSearchResult:
    vectkv_id: VectKvId
    distance: float
    similarity_score: Optional[float]
    metadata: Optional[JSON] = None


DEFAULT_HNSW_M = 16

DEFAULT_HNSW_EF_CONSTRUCTION = 200

DEFAULT_HNSW_MAX_ELEMENT = 10000

DEFAULT_HNSW_RESIZE_COUNT = 10000


def distance_to_cosine_similarity(d: float) -> float:
    return 1 - d

def distance_to_euclidean_similarity(d: float) -> float:
    return 1 / (1 + d)

class VectKv:
    def __init__(self) -> None:
        self.global_store: Dict[VectKvId, Page] = {}
        self.global_lock = asyncio.Lock()
    
    async def create_page(
        self,
        page_name: VectKvId, 
        vector_dim: int, 
        metric: Metric = "cosine", 

        HNSW_M: Optional[int] = None,

        HNSW_MAX_ELEMENT: Optional[int] = None,

        HNSW_EF_CONSTRUCTION: Optional[int] = None,

        HNSW_RESIZE_COUNT: Optional[int]  = None,

        HNSW_EF_SEARCH: Optional[int] = None,

        allow_updates: bool = True,

        write_to_wal_log: bool = True

    ) -> None:
        async with self.global_lock:
            if page_name in self.global_store:
                raise VectKvException("Page name exist")
            
            if HNSW_EF_SEARCH:
                if HNSW_EF_SEARCH <= 0:
                    raise VectKvException("HNSW_EF_SEARCH must not be less than or equal 0")
                

            wal_operation = CreatePageWALOperation(
                op="CREATE_PAGE",
                page_name=page_name,
                vector_dim=vector_dim,
                metric=metric,
                allow_updates=allow_updates,
                HNSW_M=HNSW_M or DEFAULT_HNSW_M,
                HNSW_EF_CONSTRUCTION=HNSW_EF_CONSTRUCTION or DEFAULT_HNSW_EF_CONSTRUCTION,
                HNSW_MAX_ELEMENT=HNSW_MAX_ELEMENT or DEFAULT_HNSW_MAX_ELEMENT,
                HNSW_RESIZE_COUNT=HNSW_RESIZE_COUNT or DEFAULT_HNSW_RESIZE_COUNT,
                HNSW_EF_SEARCH=HNSW_EF_SEARCH,
            )

            if write_to_wal_log:
                await self.wal_file.write(wal_operation.model_dump_json() + "\n")

            self.global_store[page_name] = Page(
                vectors={}, 
                hnsw_id_to_vectkv_id_map={},
                vectkv_id_to_hnsw_id_map={},
                vector_dim=vector_dim, 
                metric=metric,
                allow_updates=allow_updates,
                HNSW_EF_CONSTRUCTION= HNSW_EF_CONSTRUCTION or DEFAULT_HNSW_EF_CONSTRUCTION,
                HNSW_RESIZE_COUNT= HNSW_RESIZE_COUNT or DEFAULT_HNSW_RESIZE_COUNT,
                HNSW_MAX_ELEMENT = HNSW_MAX_ELEMENT or DEFAULT_HNSW_MAX_ELEMENT,
                HNSW_M = HNSW_M or DEFAULT_HNSW_M,
                HNSW_EF_SEARCH=HNSW_EF_SEARCH
            )

    async def delete_page(self, page_name: VectKvId, write_to_wal_log: bool = True) -> None:
        async with self.global_lock:
            if page_name not in self.global_store:
                raise VectKvException("Page name does not exist")

            wal_operation = DeletePageWALOperation(
                op="DELETE_PAGE",
                page_name=page_name,
            )

            if write_to_wal_log:
                await self.wal_file.write(wal_operation.model_dump_json() + "\n")

            del self.global_store[page_name]


    async def get_all_pages(self) -> List[str]:
        async with self.global_lock:
            return list(self.global_store.keys())
        

    async def add_vector(self, page_name: VectKvId, vector_id: VectKvId, vector: List[float], metadata: Optional[JSON] = None, write_to_wal_log: bool = True) -> None:
        # Get the page if exists
        async with self.global_lock:
            if page_name not in self.global_store:
                raise VectKvException("Page name does not exist")
            
        page =  self.global_store[page_name]

        # Use the lock on the page
        async with page.lock:

            numpy_vector = np.array(vector)

            # We just verify if the shape is not the same as the page's vector dim
            if numpy_vector.shape[0] != page.vector_dim:
                raise PageVectorDimException("Supplied vector dim not the same as page vector dim")
            
            if not page.allow_updates:

                # Check if the id already exits
                existing_vector  = page.vectors.get(vector_id, None)

                if existing_vector:
                    raise VectKvIdAlreadyExistException("Supplied vector id already exists")
            
            wal_operation = AddVectorPageWALOperation(
                op="ADD_VECTOR",
                page_name=page_name,
                vector_id=vector_id,
                vector=vector,
                metadata=metadata,
            )

            if write_to_wal_log:
                await self.wal_file.write(wal_operation.model_dump_json() + "\n")

            existing_vector_hnsw_id: Optional[int] = page.vectkv_id_to_hnsw_id_map.get(vector_id, None)
            
            page.vectors[vector_id] = VectorStore(metadata)

            # Write to the index
            page.hnsw_index.add_items(numpy_vector.reshape(1, -1), [ existing_vector_hnsw_id if existing_vector_hnsw_id is not None else page.next_hnsw_id])

            page.hnsw_id_to_vectkv_id_map[existing_vector_hnsw_id if existing_vector_hnsw_id is not None else page.next_hnsw_id] = vector_id

            page.vectkv_id_to_hnsw_id_map[vector_id] = existing_vector_hnsw_id if existing_vector_hnsw_id is not None else page.next_hnsw_id

            if existing_vector_hnsw_id is None:
                page.next_hnsw_id += 1

            
    
    async def __aenter__(self) -> Self:
        WAL_LOG_PATH = "wal.log.ndjson"

        wal_log_path_exists = os.path.exists(os.path.join(BASE_DIR, WAL_LOG_PATH))

        if wal_log_path_exists:
            async with aiofiles.open(WAL_LOG_PATH, "r") as wal_file:
                async for line in wal_file:
                    wal_operation_json = json.loads(line)

                    operation: OperationType = wal_operation_json.get("op", None)

                    if not operation:
                        continue

                    if operation == "CREATE_PAGE":
                        create_wal_operation = CreatePageWALOperation.model_validate(wal_operation_json)



        # Check if the
        self.wal_file = await aiofiles.open("wal.log.ndjson", "a")

        return self
    
    async def __aexit__(self, exc_type: Optional[type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:

        await self.wal_file.flush()
        await self.wal_file.close()


    async def search_vectors(self, *, page_name: VectKvId, query_vector: List[float], top_k: int) -> List[VectKvSearchResult]:
        async with self.global_lock:
            if page_name not in self.global_store:
                raise VectKvException("Page name does not exist")
            
        page =  self.global_store[page_name]

        async with page.lock:

            if top_k < 1:
                raise VectKvException("Top k must be 1 or greater")

            numpy_vector = np.array(query_vector)

            if numpy_vector.shape[0] != page.vector_dim:
                raise PageVectorDimException("Supplied query vector dim not the same as page vector dim")

            # get the minimum top_k to prevent runtime error, (top_k cannot be more than the current number of vectors)
            minimum_top_k = min(top_k, page.hnsw_index.get_current_count()) 

            labels, distances = page.hnsw_index.knn_query(numpy_vector.reshape(1, -1), k=minimum_top_k)

            results: List[VectKvSearchResult] = []
            
            for label, distance in zip(labels[0], distances[0]):
                print(f"VectkvID: {page.hnsw_id_to_vectkv_id_map[label]}   Label: {label}   Distance {distance}")


                distance = float(distance)

                # Only compute similarity scores for cosine and euclidean not ip
                similarity_score: Optional[float] = None

                if page.metric == "cosine":
                    similarity_score = distance_to_cosine_similarity(distance)

                elif page.metric == "euclidean":
                    similarity_score = distance_to_euclidean_similarity(distance)
                

                search_result = VectKvSearchResult(
                    vectkv_id=page.hnsw_id_to_vectkv_id_map[label],
                    distance=float(distance),
                    similarity_score=similarity_score,
                    metadata=page.vectors[page.hnsw_id_to_vectkv_id_map[label]].metadata
                )

                results.append(search_result)

            return results

    async def set_ef_search(self, page_name: VectKvId, ef_search: int) -> None:
        async with self.global_lock:
            if page_name not in self.global_store:
                raise VectKvException("Page name does not exist")
            
        page =  self.global_store[page_name]

        async with page.lock:

            if ef_search <= 0:
                    raise VectKvException("HNSW_EF_SEARCH must not be less than or equal 0")
            page.hnsw_index.set_ef(ef_search)
    

async def main() -> None:
    async with VectKv() as db:

        await db.create_page(
            page_name="animals",
            vector_dim=3,
            allow_updates=False
        )


        await db.set_ef_search("animals", 50)

        # Add some vectors

        await db.add_vector("animals", "dog", [1.0, 2.0, 3.0])

        await db.add_vector("animals", "cat", [1.1, 2.2, 3.3])

        await db.add_vector("animals", "lion", [1.9, 1.9, 3.2])

        await db.add_vector("animals", "bird", [4.0, 6.0, 7.0])


        results = await db.search_vectors(page_name="animals", query_vector=[1.2, 2.3, 3.4], top_k=10)

        print(results)

asyncio.run(main())


    





