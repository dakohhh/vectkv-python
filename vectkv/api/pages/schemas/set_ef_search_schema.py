from pydantic import BaseModel, Field


class SetEfSearchSchema(BaseModel):
    ef_search: int = Field(gt=0)
