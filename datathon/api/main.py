from fastapi import FastAPI

from datathon.api.routes.predict import router as predict_router

app = FastAPI()

app.include_router(predict_router)

@app.get("/")
async def root():
    return {"status": "ok"}
