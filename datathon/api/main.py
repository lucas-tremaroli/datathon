from fastapi import FastAPI

from datathon.api.routes.predict import router as predict_router

app = FastAPI(
    title="Datathon API",
    description="API for predicting lag worsening for students from Passos Mágicos.",
    version="1.0.0",
)

app.include_router(predict_router)

@app.get("/")
async def root():
    return {"status": "ok"}
