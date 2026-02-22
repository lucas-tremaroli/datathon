from fastapi import APIRouter

router = APIRouter(
    prefix="/api",
    tags=["predict"]
)

@router.get("/predict")
async def predict():
    return {"message": "This is the predict endpoint"}
