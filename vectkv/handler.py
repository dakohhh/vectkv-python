from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status
from .exceptions import VectKvException, PageVectorDimException, VectKvIdAlreadyExistException
from fastapi.exceptions import HTTPException

def configure_error_middleware(app: FastAPI) -> None:

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        content = None

        if isinstance(exc.detail, str):
            content = {"message": exc.detail }

        return JSONResponse(content=content, status_code=exc.status_code)
    

    @app.exception_handler(VectKvIdAlreadyExistException)
    async def vectkv_id_already_exist_exception_handler(request: Request, exc: VectKvIdAlreadyExistException) -> JSONResponse:
        return JSONResponse(content={"message": str(exc)}, status_code=status.HTTP_409_CONFLICT)

    @app.exception_handler(PageVectorDimException)
    async def page_vector_dim_exception_handler(request: Request, exc: PageVectorDimException) -> JSONResponse:
        return JSONResponse(content={"message": str(exc)}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

    @app.exception_handler(VectKvException)
    async def vectkv_exception_handler(request: Request, exc: VectKvException) -> JSONResponse:
        return JSONResponse(content={"message": str(exc)}, status_code=status.HTTP_400_BAD_REQUEST)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Use Sentry to capture exception
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)                               