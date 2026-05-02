from typing import List, Optional
from pydantic import BaseModel
from ....types import JSON, VectKvId


class AddVectorSchema(BaseModel):
    page_name: str
    vector_id: VectKvId
    vector: List[float]
    metadata: Optional[JSON] = None
