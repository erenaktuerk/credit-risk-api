import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from typing import Union

class ModelInterpreter:
    """
    A class for interpreting machine learning model predictions using SHAP (SHapley Additive exPlanations) values.

    This class provides methods to explain individual predictions and global feature importance
    for credit risk assessment models. It supports both tree-based and black-box models.
    """

    def __init__(self, model: BaseEstimator, data: pd.DataFrame):
        """
        Initializes the ModelInterpreter with a trained model and sample data.

        Args:
            model (BaseEstimator): The trained machine learning model (e.g., LogisticRegression).
            data (pd.DataFrame): The preprocessed feature data used for model training.
        """
        self.model = model
        self.data = data
        self.explainer = None

    def initialize_explainer(self, background_data: pd.DataFrame) -> None:
        """
        Public method to initialize the SHAP explainer using the provided background data.

        Args:
            background_data (pd.DataFrame): A subset of the training data used as background.
        """
        # Use TreeExplainer if the model supports predict_proba, else use KernelExplainer.
        if hasattr(self.model, "predict_proba"):
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"TreeExplainer initialization failed: {e}")
                self.explainer = shap.KernelExplainer(lambda x: self.model.predict(x),
                                                      shap.sample(background_data, 100))
        else:
            self.explainer = shap.KernelExplainer(lambda x: self.model.predict(x),
                                                  shap.sample(background_data, 100))

    def compute_shap_values(self, input_data: pd.DataFrame):
        """
        Computes SHAP values for the input data.

        Args:
            input_data (pd.DataFrame): The data for which to compute SHAP values.

        Returns:
            The computed SHAP values.
        """
        if self.explainer is None:
            raise ValueError("Explainer is not initialized. Call initialize_explainer() first.")
        return self.explainer.shap_values(input_data)

    def plot_summary(self, shap_values, input_data: pd.DataFrame) -> None:
        """
        Generates and displays a SHAP summary plot.

        Args:
            shap_values: The computed SHAP values.
            input_data (pd.DataFrame): The input data used to compute SHAP values.
        """
        plt.figure()
        # Use the feature names from self.data
        shap.summary_plot(shap_values, input_data, feature_names=self.data.columns)

    def explain_global(self, max_display: int = 10) -> None:
        """
        Visualizes global feature importance based on SHAP values.

        Args:
            max_display (int): The maximum number of features to display in the summary plot.
        """
        shap_values = self.explainer(self.data)
        plt.figure()
        shap.summary_plot(shap_values, self.data, max_display=max_display)

    def explain_local(self, instance: pd.DataFrame) -> None:
        """
        Visualizes the SHAP values for a single prediction (local explanation).

        Args:
            instance (pd.DataFrame): A single row of feature data for prediction.
        """
        if instance.shape[0] != 1:
            raise ValueError("Instance must be a single row of data.")
        shap_values = self.explainer(instance)
        plt.figure()
        shap.plots.waterfall(shap_values[0])

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculates and returns global feature importance based on SHAP values.

        Returns:
            pd.DataFrame: A DataFrame with features and their mean absolute SHAP values.
        """
        shap_values = self.explainer(self.data)
        importance = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame(
            {"feature": self.data.columns, "importance": importance}
        ).sort_values(by="importance", ascending=False)
        return feature_importance