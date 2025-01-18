from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np

router = APIRouter()

# Load model
model = joblib.load("app/models/model.pkl")

# Input schema
class CreditRiskInput(BaseModel):
    income: float
    age: int
    loan_amount: float
    duration: int
    credit_score: int

# Prediction endpoint
@router.post("/predict/")
def predict_risk(input_data: CreditRiskInput):
    data = np.array([[input_data.income, input_data.age, input_data.loan_amount, 
                      input_data.duration, input_data.credit_score]])
    prediction = model.predict(data)
    risk = "Low Risk" if prediction[0] == 0 else "High Risk"
    return {"risk": risk}