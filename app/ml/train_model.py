import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump

# Load dataset
data_path = "data/cleaned_data/cleaned_data.csv"
data = pd.read_csv(data_path)

# Encode target variable
label_encoder = LabelEncoder()
data['loan_grade'] = label_encoder.fit_transform(data['loan_grade'])

# Drop irrelevant columns
data.drop(columns=["loan_status"], inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop(columns=["loan_grade"])
y = data["loan_grade"]

# Define the pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced"))
])

# Hyperparameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1],  # Narrow range for faster optimization
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and evaluation
print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
dump(best_model, "app/models/model.pkl")
print("Model saved successfully.")