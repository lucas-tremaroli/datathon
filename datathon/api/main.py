from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from datathon.api.routes.model import router as model_router
from datathon.api.routes.predict import router as predict_router
from datathon.api.util.logging import RequestLoggingMiddleware, setup_logging

setup_logging()

app = FastAPI(
    title="Datathon API",
    description="API for predicting lag worsening for students from Passos Mágicos.",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(model_router)
app.include_router(predict_router)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode(),
        media_type=CONTENT_TYPE_LATEST,
    )
