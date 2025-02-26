import joblib
import os

def load_model(model_path: str):
    """
    Load a machine learning model from the specified file path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)