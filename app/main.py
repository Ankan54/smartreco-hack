import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app import config, db
from app.routes import admin, pages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    yield


app = FastAPI(title="Reckon", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY, max_age=60 * 60 * 24 * 14)
app.mount("/static", StaticFiles(directory=config.ROOT / "app" / "static"), name="static")
app.include_router(pages.router)
app.include_router(admin.router)


@app.get("/healthz")
def healthz():
    return {"ok": True}
