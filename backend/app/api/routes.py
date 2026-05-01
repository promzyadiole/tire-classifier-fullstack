from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.predictor import predict_tire

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Tire classifier backend is running.",
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file upload.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    try:
        result = predict_tire(image_bytes)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )