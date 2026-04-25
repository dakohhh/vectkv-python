from typing import Dict, List, Literal

type JSON = Dict[str, "JSON"] | List["JSON"] | str | int | float | bool | None

type Metric = Literal["cosine", "euclidean", "ip"]

type VectKvId = str

type OperationType = Literal["ADD_VECTOR", "CREATE_PAGE", "DELETE_PAGE"]