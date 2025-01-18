import pandas as pd
# Load raw data
data = pd.read_csv("data/raw/raw_data.csv")

# Example of simple preprocessing steps (customizable)
data = data.dropna()  # Remove rows with missing values
data["loan_grade"] = data["loan_grade"].map({"good": 0, "bad": 1})  # Convert categorical target variable to numeric

# Save the cleaned data
data.to_csv("data/cleaned_data/cleaned_data.csv", index=False)