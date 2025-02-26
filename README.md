Credit Risk Prediction API

This project provides a complete, production-oriented solution for predicting credit risk for loan applications. By integrating advanced data preprocessing, robust model training with hyperparameter optimization, and a FastAPI-based RESTful interface, the solution demonstrates how to build and deploy machine learning models for real-world financial decision support. The project highlights best practices in data handling, model optimization, and API development.

Features
	•	Comprehensive Data Preprocessing:
	•	Cleans raw data (removes duplicates, handles missing values with median/mode imputation).
	•	Applies outlier capping using the IQR method.
	•	Creates derived features (e.g., income-to-loan ratio).
	•	One-Hot Encodes categorical variables for model compatibility.
	•	Robust Model Training:
	•	Uses Logistic Regression with hyperparameter tuning via GridSearchCV.
	•	Incorporates stratified cross-validation to ensure model generalization.
	•	Saves both the trained model and the feature set for consistent predictions.
	•	Real-Time Prediction API:
	•	Deploys the model using FastAPI, offering a RESTful interface.
	•	Provides interactive API documentation (Swagger UI) for easy testing and integration.
	•	Ensures production-readiness with modular, well-documented code.

Tools & Frameworks
	•	Python: Primary programming language.
	•	FastAPI: Framework for building the RESTful API.
	•	scikit-learn: Library for machine learning models, preprocessing, and evaluation.
	•	Pandas & NumPy: For data manipulation and numerical operations.
	•	Uvicorn: ASGI server to run the FastAPI application.
	•	Joblib: For model persistence and saving feature sets.

Installation & Usage
	1.	Clone the Repository:

git clone https://github.com/erenaktuerk/credit-risk-api
cd credit-risk-api


	2.	Install Dependencies:

pip install -r requirements.txt


	3.	Train the Model:
Run the training script to preprocess data, train the model, and save both the model and the feature columns:

python app/ml/train_model.py


	4.	Start the API Server:
Launch the FastAPI server with Uvicorn:

uvicorn app.main:app --reload

The API will be accessible at http://127.0.0.1:8000 and the interactive docs under /docs.

Next Steps
	•	Model Improvement:
	•	Evaluate alternative models (e.g., XGBoost, Random Forest, Neural Networks).
	•	Refine feature engineering and consider dimensionality reduction if needed.
	•	Enhanced Deployment:
	•	Containerize the application using Docker for a reproducible production environment.
	•	Implement CI/CD pipelines for automated testing and deployment.
	•	Monitoring & Logging:
	•	Integrate detailed logging and monitoring for model performance and API usage in production.

License

This project is licensed under the MIT License.