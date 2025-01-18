from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI()

# Include routes
app.include_router(predict_router, prefix="/api", tags=["Credit Risk"])