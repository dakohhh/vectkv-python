import uvicorn
from .main import VectKv
from fastapi import FastAPI
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from .handler import configure_error_middleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with VectKv() as db:
        app.state.db = db
        yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    configure_error_middleware(app)

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), port=5955)