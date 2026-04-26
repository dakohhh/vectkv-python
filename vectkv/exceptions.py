from pydantic import BaseModel
from typing import Optional, Union, Any
from fastapi.exceptions import HTTPException
class VectKvException(Exception):
    """Base Exception class"""

class PageVectorDimException(VectKvException):
    """Supplied vector dim not the same as page vector dim"""

class VectKvIdAlreadyExistException(VectKvException):
    """Supplied vector id already exists"""


ErrorDataType = Optional[Union[str, dict[Any, Any], list[Any]]]

class ErrorResponse(BaseModel):
    message: str
    status_code: int
    data: ErrorDataType = None

class BaseHTTPException(HTTPException):
    def __init__(self, message: str, data: ErrorDataType = None):

        error = ErrorResponse(message=message, data=data, status_code=self.status_code)

        super().__init__(status_code=self.status_code, detail=error)
