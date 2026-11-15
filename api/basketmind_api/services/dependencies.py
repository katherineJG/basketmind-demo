"""FastAPI dependency: the loaded inference service, or a clear 503 if the model could not be loaded."""

from fastapi import HTTPException, Request

from basketmind.inference import BasketMindService


def get_service(request: Request) -> BasketMindService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        reason = getattr(request.app.state, "load_error", "the model has not been loaded")
        raise HTTPException(status_code=503, detail=f"Model not available: {reason}")
    return service
