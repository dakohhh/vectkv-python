# VectKV

A vector database built from scratch in Python. This is not a wrapper around an existing vector DB — it implements the core storage, indexing, and durability layers directly.

Built as a learning project to understand how production vector databases like Chroma and Qdrant actually work under the hood.

> **Status:** Actively in development. APIs may change.

---

## What it does

VectKV stores vectors grouped into **pages** (namespaced collections). Each page maintains its own HNSW index for approximate nearest-neighbour search, with support for cosine, euclidean, and inner product distance metrics.

---

## Architecture & Design Decisions

### Pages
Vectors are organized into pages — isolated namespaces each with their own index, vector dimensionality, and HNSW configuration. This mirrors how production databases like Qdrant use "collections": you can tune each page independently rather than forcing all vectors into a single global index.

### HNSW Indexing
Vector search is powered by [hnswlib](https://github.com/nmslib/hnswlib), a C++ implementation of the Hierarchical Navigable Small World algorithm. This is the same algorithm used under the hood by Chroma, Qdrant, and others. It gives sub-linear approximate nearest-neighbour search with tunable accuracy/speed trade-offs via `HNSW_M`, `HNSW_EF_CONSTRUCTION`, and `HNSW_EF_SEARCH`.

### Write-Ahead Log (WAL)
Every mutation (`CREATE_PAGE`, `ADD_VECTOR`, `DELETE_PAGE`) is written to a WAL file (`wal.log.ndjson`) before it is applied to in-memory state. On startup, VectKV replays the WAL to reconstruct the full state. This gives crash recovery without a full on-disk persistence layer.

### Snapshots *(in progress)*
To avoid replaying an unbounded WAL on every startup, VectKV will support point-in-time snapshots. A snapshot serializes all page state (via Pydantic's `model_dump`) to `state.json` and saves each page's HNSW index as a separate binary file. On restore, the serializable state is reloaded via `model_validate` and the HNSW index is rehydrated with `load_index`.

### Async-First Concurrency
The entire engine is async. Concurrency is managed at two levels:
- **`global_lock`** — a single asyncio lock guards structural operations (creating and deleting pages)
- **`page.lock`** — each page has its own lock for vector-level writes and searches, so concurrent operations on different pages never block each other

### HTTP Layer
VectKV exposes a FastAPI HTTP server. Pydantic models are used for all request validation and response serialization. Errors from the core engine are mapped to appropriate HTTP status codes via FastAPI exception handlers.

---

## Project Structure

```
vectkv/
├── main.py          # VectKv engine — core logic, WAL replay, context manager lifecycle
├── schema.py        # Pydantic models: Page, VectorStore, VectKvSearchResult
├── wal.py           # WAL operation models
├── types.py         # Shared type aliases (VectKvId, Metric, etc.)
├── exceptions.py    # Exception hierarchy + error response models
├── handler.py       # FastAPI exception handlers
├── server.py        # App factory + lifespan
└── api/
    ├── router.py    # Route definitions
    └── schemas/     # Request body schemas
```

---

## Getting Started

**Requirements:** Python 3.12+, [Poetry](https://python-poetry.org/)

```bash
git clone https://github.com/dakohhh/VectKV-Python.git
cd VectKV-Python
poetry install
poetry run python -m vectkv.server
```

The server starts on port `5955`.

---

## API

### Create a page

```http
POST /v1/pages/
Content-Type: application/json

{
  "page_name": "my-embeddings",
  "vector_dim": 1536,
  "metric": "cosine"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `page_name` | string | required | Unique name for the page |
| `vector_dim` | int | required | Dimensionality of vectors stored in this page |
| `metric` | string | `"cosine"` | Distance metric: `cosine`, `euclidean`, or `ip` |
| `allow_updates` | bool | `true` | Whether existing vector IDs can be overwritten |
| `HNSW_M` | int | `16` | Number of bi-directional links per HNSW node |
| `HNSW_EF_CONSTRUCTION` | int | `200` | Candidate list size during index construction |
| `HNSW_MAX_ELEMENT` | int | `10000` | Maximum vectors the index can hold |
| `HNSW_RESIZE_COUNT` | int | `10000` | Growth increment when resizing |
| `HNSW_EF_SEARCH` | int | `null` | Candidate list size during search (higher = more accurate, slower) |

---

## Roadmap

- [x] In-memory page storage with HNSW indexing
- [x] Write-ahead log with crash recovery
- [x] Async concurrency with per-page locking
- [x] FastAPI HTTP layer
- [ ] Snapshots (WAL + binary index persistence)
- [ ] WAL compaction after snapshot
- [ ] Delete vector support
- [ ] API key authentication
- [ ] Rate limiting

---

## Contributing

This project is open to contributions. If you have ideas, find a bug, or want to improve something:

- **Issues** — open one to discuss a bug, ask a question, or propose a feature before building it
- **Pull requests** — welcome for anything on the roadmap or improvements you think are worthwhile. For larger changes, open an issue first so we can align before you invest the time

There's no formal contribution guide yet — keep code async-first, add Pydantic models for any new data shapes, and follow the existing exception hierarchy for errors.

---

## License

MIT
