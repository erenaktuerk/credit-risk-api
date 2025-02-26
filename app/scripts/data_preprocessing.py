import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load raw data
data = pd.read_csv("data/raw/raw_data.csv")

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object"]).columns

# Impute missing values in numeric columns using the median
num_imputer = SimpleImputer(strategy="median")
data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

# Impute missing values in categorical columns using the most frequent value
cat_imputer = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Cap outliers in numeric columns using the IQR method
def cap_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return col.clip(lower_bound, upper_bound)

for col in numeric_cols:
    data[col] = cap_outliers(data[col])

# Feature Engineering: create a new feature "income_to_loan_ratio"
if "loan_amnt" in data.columns and "person_income" in data.columns:
    data["loan_amnt"].replace(0, 1e-6, inplace=True)
    data["income_to_loan_ratio"] = data["person_income"] / data["loan_amnt"]

# Process target variable: map "loan_grade" if it is categorical
if "loan_grade" in data.columns and data["loan_grade"].dtype == object:
    data["loan_grade"] = data["loan_grade"].map({"good": 0, "bad": 1})

# Drop irrelevant columns
if "loan_status" in data.columns and data["loan_status"].isna().all():
    data.drop(columns=["loan_status"], inplace=True)

# One-hot encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Verify the data shape after encoding
print(f"Data shape after encoding: {data.shape}")

# Save the cleaned data
data.to_csv("data/cleaned_data/cleaned_data.csv", index=False)
print("Data preprocessing completed and saved.")