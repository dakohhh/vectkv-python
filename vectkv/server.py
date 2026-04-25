from .main import VectKv
from fastapi import FastAPI
from typing import AsyncGenerator
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with VectKv() as db:
        app.state.db = db
        yield


app = FastAPI(lifespan=lifespan)