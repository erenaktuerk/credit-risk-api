Credit Risk Prediction API

This project aims to predict credit risk for loan applications using machine learning. It preprocesses the data, trains a classification model, and exposes predictions through a FastAPI-based RESTful API.

Features
	•	Data Preprocessing: Cleans and prepares the data for modeling.
	•	Model Training: Uses a logistic regression model to predict credit risk.
	•	API Deployment: Exposes the model for real-time predictions via FastAPI.

Tools & Frameworks
	•	Python: Primary programming language
	•	FastAPI: Framework for building the RESTful API
	•	scikit-learn: For machine learning models and evaluation
	•	Pandas: For data manipulation and preprocessing
	•	Uvicorn: ASGI server for running the FastAPI app

Installation
	1.	Clone the repository:

git clone <https://github.com/erenaktuerk/credit-risk-api>


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Train the model and run the API:
	•	First, train the model:

python app/ml/train_model.py


	•	Start the API server:

uvicorn app.main:app --reload


The model will be trained and saved, and the API will be accessible at http://127.0.0.1:8000.

License

MIT License