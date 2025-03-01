from fastapi import FastAPI
from app.routes.predict import router as predict_router
from app.evaluation.model_interpretation import ModelInterpreter
import logging
import pandas as pd
import joblib

# Set up basic logging configuration for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with additional metadata for better documentation
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API for predicting credit risk for loan applications using a machine learning model.",
    version="1.0.0"
)

# Middleware for logging requests — captures incoming requests and outgoing responses
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Include the prediction router with the /api prefix and tag for better organization
app.include_router(predict_router, prefix="/api", tags=["Credit Risk"])

# Model interpretation workflow for SHAP analysis
def interpret_model():
    """
    This function initializes and executes the model interpretation workflow.
    It loads the preprocessed dataset, ensures the input features match the model's expected columns,
    computes SHAP values, and generates a summary plot to interpret feature importance.
    """
    try:
        # Load preprocessed dataset
        df = pd.read_csv("data/cleaned_data/cleaned_data.csv")

        # Prepare input features by removing the target column if present
        if "loan_grade" in df.columns:
            X = df.drop(columns=["loan_grade"])
        else:
            X = df.copy()

        # Ensure input features match model's expected columns
        model_columns = joblib.load("app/models/model_columns.pkl")
        missing_cols = set(model_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0  # Add missing columns with default value 0
        X = X[model_columns]  # Reorder columns to match training data

        # Load the trained model (using the correct file name)
        model = joblib.load("app/models/model.pkl")

        # Initialize ModelInterpreter with the trained model and the feature data
        interpreter = ModelInterpreter(model=model, data=X)

        # Select a background data sample and initialize the SHAP explainer using the public method
        background = X.sample(n=100, random_state=42)
        interpreter.initialize_explainer(background)

        # Compute SHAP values for a sample input (first 10 rows)
        sample_data = X.head(10)
        shap_values = interpreter.compute_shap_values(sample_data)

        # Generate and display (or save) the SHAP summary plot
        interpreter.plot_summary(shap_values, sample_data)
        logger.info("SHAP summary plot successfully created and saved.")

    except Exception as e:
        logger.error(f"Error during model interpretation: {e}", exc_info=True)

# Execute model interpretation when this script runs
if __name__ == "__main__":
    interpret_model()