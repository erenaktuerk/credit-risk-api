Credit Risk Prediction API

This project delivers a professional, production-ready solution for predicting credit risk in loan applications. It combines advanced machine learning techniques, optimized model training, and a FastAPI-based RESTful interface — all built with real-world deployment and problem-solving in mind. The project reflects industry best practices in data preprocessing, model interpretability, and API development, making it a robust and scalable decision-support system for financial institutions.

Key Features

1. Comprehensive Data Preprocessing
	•	Cleaning: Removes duplicates and handles missing values with median/mode imputation.
	•	Outlier Treatment: Applies IQR-based outlier capping.
	•	Feature Engineering: Creates derived features like income-to-loan ratio for enhanced model performance.
	•	Encoding: One-Hot encodes categorical variables for model compatibility.

2. Robust Model Training & Optimization
	•	Advanced Algorithms: Uses Logistic Regression with hyperparameter tuning.
	•	Cross-Validation: Applies stratified cross-validation for improved model generalization.
	•	Model Persistence: Saves trained models and feature sets for consistent, reproducible predictions.

3. Real-Time Prediction API
	•	FastAPI Deployment: Provides a fast and scalable RESTful interface.
	•	Interactive Documentation: Offers Swagger UI for easy testing and integration.
	•	Production-Ready: Ensures modular, well-documented, and deployment-focused code.

4. Model Interpretation & Explainability
	•	SHAP Implementation: Uses SHapley Additive exPlanations to provide deep insights into feature importance and model behavior.
	•	Interpretation Module: A dedicated model_interpretation.py file visualizes and explains model decisions for transparency and trustworthiness.

Tools & Frameworks
	•	Python: Core programming language.
	•	FastAPI: High-performance web framework.
	•	scikit-learn: Library for machine learning and model evaluation.
	•	Pandas & NumPy: Essential for data manipulation and numerical operations.
	•	Uvicorn: ASGI server for running FastAPI applications.
	•	SHAP: Library for model explainability.
	•	Joblib: For saving models and feature sets.

Installation & Usage

1. Clone the Repository

git clone https://github.com/erenaktuerk/credit-risk-api
cd credit-risk-api

2. Install Dependencies

pip install -r requirements.txt

3. Train the Model

Preprocess data, train the model, and save it alongside feature columns:

python app/ml/train_model.py

4. Start the API Server

Deploy the FastAPI server:

uvicorn app.main:app --reload

Access the API at http://127.0.0.1:8000 and interactive docs at /docs.

5. Interpret Model Behavior

Visualize feature importance and model decisions:

python app/ml/model_interpretation.py

Next Steps
	•	Model Improvement: Explore more sophisticated models like XGBoost, Random Forest, or Neural Networks.
	•	Feature Engineering: Refine features and consider dimensionality reduction.
	•	Deployment Enhancements: Containerize with Docker and set up CI/CD pipelines.
	•	Monitoring & Logging: Integrate real-time performance monitoring and detailed logging.

License

This project is licensed under the MIT License.