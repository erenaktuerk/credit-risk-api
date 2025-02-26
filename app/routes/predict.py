from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from app.utils import load_model

router = APIRouter()

# Load the machine learning model using the utility function
try:
    model = load_model("app/models/model.pkl")
    model_columns = joblib.load('app/models/model_columns.pkl')  # Load model columns
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Define the input schema for prediction
class CreditRiskInput(BaseModel):
    income: float
    age: int
    loan_amount: float
    duration: int
    credit_score: int
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_int_rate: float
    loan_status: str
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int
    cb_person_emp_length: int
    loan_term: int
    cb_person_age: int
    cb_person_income: float
    loan_purpose: str

@router.post("/predict/", response_model=dict)
def predict_risk(input_data: CreditRiskInput):
    """
    Predict credit risk based on the input data.
    """
    try:
        # Prepare the data for prediction
        data_dict = input_data.dict()
        input_data_df = pd.DataFrame([data_dict])

        # Apply One-Hot Encoding
        input_data_df = pd.get_dummies(input_data_df, drop_first=True)

        # Ensure the input data has the same columns as the model was trained on
        missing_cols = set(model_columns) - set(input_data_df.columns)
        for col in missing_cols:
            input_data_df[col] = 0  # Add missing columns with 0 values

        # Reorder columns to match the model training set
        input_data_df = input_data_df[model_columns]

        # Apply imputation and scaling
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        input_data_df = imputer.fit_transform(input_data_df)
        input_data_df = scaler.fit_transform(input_data_df)

        # Make prediction using the model
        prediction = model.predict(input_data_df)
        
        # Map prediction to risk level
        risk = "Low Risk" if prediction[0] == 0 else "High Risk"
        return {"risk": risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")