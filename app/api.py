from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and other necessary preprocessing tools
model = joblib.load("app/models/model.pkl")  # Load the trained model
model_columns = joblib.load('app/models/model_columns.pkl')  # Load model columns

# Define the data structure for the input (LoanData) using Pydantic
class LoanData(BaseModel):
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

# Define the prediction endpoint using the POST method
@app.post("/api/predict/")
def predict(loan_data: LoanData):
    # Convert input data to a dictionary and then to a DataFrame
    data_dict = loan_data.dict()  # Convert Pydantic model to dictionary
    input_data = pd.DataFrame([data_dict])  # Convert dictionary to DataFrame

    # Apply One-Hot Encoding and ensure that all expected categories are present
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure that the input data has the same columns as the model was trained on
    missing_cols = set(model_columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0  # Add missing columns with 0 values

    # Reorder columns to match the model training set
    input_data = input_data[model_columns]  # model_columns should be saved during training

    # Initialize imputer and scaler
    imputer = SimpleImputer(strategy="median")  # Imputer for missing values
    scaler = StandardScaler()  # Scaler for feature scaling

    # Impute missing values and scale the features
    input_data = imputer.fit_transform(input_data)  # Apply imputation
    input_data = scaler.fit_transform(input_data)  # Apply scaling

    # Make a prediction using the trained model
    prediction = model.predict(input_data)

    # Return the prediction as a response in JSON format
    return {"prediction": prediction[0]}