from fastapi import APIRouter

from app.api.v1.endpoints import food_scan, health, risk_prediction, segmentation

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(
    risk_prediction.router, prefix="/risk-prediction", tags=["CU10 - Random Forest"]
)
api_router.include_router(
    segmentation.router, prefix="/segmentation", tags=["CU11 - K-means"]
)
api_router.include_router(
    food_scan.router, prefix="/food-scan", tags=["CU9 - OCR / Deep Learning"]
)
