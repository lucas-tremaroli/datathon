from fastapi import FastAPI

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
