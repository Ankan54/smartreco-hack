import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app import config, db, scheduler, vectors
from app.routes import admin, api, pages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    problem = vectors.check_index_model()
    if problem:
        logging.getLogger(__name__).error("STARTUP: %s", problem)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="Reckon", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY, max_age=60 * 60 * 24 * 14)
app.mount("/static", StaticFiles(directory=config.ROOT / "app" / "static"), name="static")
app.include_router(pages.router)
app.include_router(admin.router)
app.include_router(api.router)


@app.get("/healthz")
def healthz():
    return {"ok": True}
