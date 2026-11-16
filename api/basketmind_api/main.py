"""FastAPI application. Run with: uvicorn basketmind_api.main:app

The model and data are loaded once at startup (about half a minute). If they cannot be loaded the app still
starts: /health reports `degraded` and the other routes answer 503 with the reason.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from basketmind.inference import DataService
from basketmind.utils.config import project_root
from basketmind.utils.logging import get_logger
from basketmind_api.routes import router

log = get_logger("api")


def create_app(
    service: DataService | None = None, factory: Callable[[], DataService] | None = None
) -> FastAPI:
    """Pass a ready `service` (tests) or a `factory` that builds one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service, app.state.load_error = service, None
        if service is None and factory is not None:
            try:
                app.state.service = factory()
                log.info("model loaded")
            except Exception as e:  # keep the API up so /health can say what is wrong
                app.state.load_error = f"{type(e).__name__}: {e}"
                log.error("model could not be loaded: %s", app.state.load_error)
        yield

    app = FastAPI(
        title="BasketMind API",
        version="0.1.0",
        description="Next-basket recommendations from a research prototype (educational, not production).",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


def _load_live_model() -> DataService:
    from basketmind.inference import BasketMindService  # imported here: it needs torch, the demo does not

    return BasketMindService.from_artifacts(project_root())


app = create_app(factory=_load_live_model)
