import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump

# Load the cleaned dataset
data_path = "data/cleaned_data/cleaned_data.csv"
data = pd.read_csv(data_path)

# Verify that the dataset contains the expected number of features
print(f"Data columns: {data.columns}")
print(f"Data shape: {data.shape}")  # Ensure the correct number of features

# Encode the target variable (loan_grade) into numerical values
label_encoder = LabelEncoder()
data['loan_grade'] = label_encoder.fit_transform(data['loan_grade'])

# Drop irrelevant columns that do not contribute to the model's prediction
# In this case, 'loan_status' is being dropped, which contains only NaN values
data.drop(columns=["loan_status"], inplace=True, errors='ignore')

# One-hot encode categorical variables to convert them into a format suitable for machine learning models
# Drop the first category to avoid multicollinearity
data = pd.get_dummies(data, drop_first=True)

# Print the shape of the data after encoding to verify the number of features
print(f"Data shape after encoding: {data.shape}")

# Define the feature set (X) and the target variable (y)
X = data.drop(columns=["loan_grade"])  # Features exclude 'loan_grade'
y = data["loan_grade"]  # Target variable is 'loan_grade'

# Save the column names (features)
model_columns = X.columns.tolist()
dump(model_columns, 'app/models/model_columns.pkl')  # Save model columns

# Define the preprocessing and modeling pipeline
# The pipeline will handle missing values, standardize features, and train a Logistic Regression model
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="mean")),      # Fill missing numeric values with the mean
    ('scaler', StandardScaler()),                     # Standardize features to have zero mean and unit variance
    ('classifier', LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))
])

# Define the hyperparameter grid for tuning the Logistic Regression model
# We will tune the regularization strength 'C' to find the best performing model
param_grid = {
    'classifier__C': [0.01, 0.1, 1]  # Regularization strength for logistic regression
}

# Split the dataset into training and testing sets, ensuring that class distribution is preserved
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set up Stratified K-Fold cross-validation to evaluate the model performance
# StratifiedKFold ensures that each fold has the same distribution of target classes
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search with cross-validation to find the best model
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv_strategy, scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Output the best hyperparameters and the best model
print(f"Best hyperparameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)  # Make predictions on the test set
test_accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model on the test set
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))  # Print detailed classification metrics

# Perform cross-validation on the training set to evaluate the model's generalization ability
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, n_jobs=-1)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average Cross-Validation Score: {cv_scores.mean():.4f}")

# Save the best model to a file so it can be used later without retraining
dump(best_model, "app/models/model.pkl")
print("Model saved successfully.")