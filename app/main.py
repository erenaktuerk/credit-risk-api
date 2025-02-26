from fastapi import FastAPI
from app.routes.predict import router as predict_router
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with additional metadata
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API for predicting credit risk for loan applications using a machine learning model.",
    version="1.0.0"
)

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Include the prediction router with prefix /api
app.include_router(predict_router, prefix="/api", tags=["Credit Risk"])