import uvicorn
from .main import VectKv
from fastapi import FastAPI
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from .handler import configure_error_middleware
from .api.pages.router import router as pages_router
from .api.vectors.router import router as vectors_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict, None]:
    async with VectKv() as db:
        yield {"db": db}


def register_routers(app: FastAPI) -> None:
    app.include_router(pages_router)
    app.include_router(vectors_router)


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    configure_error_middleware(app)

    register_routers(app)

    return app



app = create_app()

if __name__ == "__main__":
    uvicorn.run("vectkv.server:app", port=5955, reload=True)