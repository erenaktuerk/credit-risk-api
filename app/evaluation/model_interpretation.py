import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from typing import Union

class ModelInterpreter:
    """
    A class for interpreting machine learning model predictions using SHAP (SHapley Additive exPlanations) values.
    
    This class provides methods to explain individual predictions and global feature importance for credit risk
    assessment models. It supports both tree-based models (using TreeExplainer) and black-box models (using KernelExplainer).
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
        self.explainer = None  # The SHAP explainer will be initialized later using initialize_explainer()

    def initialize_explainer(self, background_data: pd.DataFrame) -> None:
        """
        Public method to initialize the SHAP explainer using the provided background data.
        
        This method determines which explainer to use based on the model's capabilities. For tree-based
        models (i.e., models with a 'predict_proba' method), it uses TreeExplainer. Otherwise, it falls back to KernelExplainer.
        
        Args:
            background_data (pd.DataFrame): A subset of the training data used as the background for SHAP calculations.
        """
        if hasattr(self.model, "predict_proba"):
            try:
                # Attempt to use TreeExplainer for models that support predict_proba
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                # In case TreeExplainer initialization fails, fall back to KernelExplainer
                print(f"TreeExplainer initialization failed: {e}")
                self.explainer = shap.KernelExplainer(lambda x: self.model.predict(x),
                                                      shap.sample(background_data, 100))
        else:
            # Use KernelExplainer as a generic fallback for models without predict_proba
            self.explainer = shap.KernelExplainer(lambda x: self.model.predict(x),
                                                  shap.sample(background_data, 100))
    
    def compute_shap_values(self, input_data: pd.DataFrame):
        """
        Computes SHAP values for the given input data using the initialized SHAP explainer.
        
        Args:
            input_data (pd.DataFrame): The data for which to compute SHAP values.
            
        Returns:
            The computed SHAP values (as returned by the explainer).
            
        Raises:
            ValueError: If the explainer has not been initialized.
        """
        if self.explainer is None:
            raise ValueError("Explainer is not initialized. Please call initialize_explainer() first.")
        return self.explainer.shap_values(input_data)
    
    def plot_summary(self, shap_values, input_data: pd.DataFrame) -> None:
        """
        Generates and displays a SHAP summary plot to visualize global feature importance.
        
        This plot shows the distribution of SHAP values for each feature and helps identify which features
        have the most influence on the models output.
        
        Args:
            shap_values: The computed SHAP values.
            input_data (pd.DataFrame): The input data used to compute the SHAP values.
        """
        plt.figure()
        # Use the feature names from the original data to label the plot
        shap.summary_plot(shap_values, input_data, feature_names=self.data.columns)
    
    def explain_global(self, max_display: int = 10) -> None:
        """
        Visualizes global feature importance by generating a SHAP summary plot using all training data.
        
        Args:
            max_display (int): The maximum number of features to display in the summary plot.
        """
        # Compute SHAP values for the entire dataset
        shap_values = self.explainer(self.data)
        plt.figure()
        shap.summary_plot(shap_values, self.data, max_display=max_display)
    
    def explain_local(self, instance: pd.DataFrame) -> None:
        """
        Generates a local explanation for a single prediction using a SHAP waterfall plot.
        
        Args:
            instance (pd.DataFrame): A single row of feature data (must have shape (1, n_features)).
            
        Raises:
            ValueError: If the input instance does not have exactly one row.
        """
        if instance.shape[0] != 1:
            raise ValueError("Instance must be a single row of data.")
        shap_values = self.explainer(instance)
        plt.figure()
        shap.plots.waterfall(shap_values[0])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculates global feature importance based on the mean absolute SHAP values.
        
        Returns:
            pd.DataFrame: A DataFrame containing features and their corresponding mean absolute SHAP values,
            sorted by importance in descending order.
        """
        shap_values = self.explainer(self.data)
        # Calculate the mean absolute SHAP value for each feature
        importance = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame(
            {"feature": self.data.columns, "importance": importance}
        ).sort_values(by="importance", ascending=False)
        return feature_importance

    def plot_dependence(self, shap_values, input_data: pd.DataFrame, feature: str) -> None:
        """
        Generates and displays a SHAP dependence plot for a specific feature.
        
        The dependence plot shows how the SHAP value for the selected feature varies with the feature's value,
        providing insight into the relationship between the feature and the model output.
        
        Args:
            shap_values: The computed SHAP values.
            input_data (pd.DataFrame): The input data used to compute SHAP values.
            feature (str): The name of the feature to plot.
        """
        plt.figure()
        shap.dependence_plot(feature, shap_values, input_data, feature_names=self.data.columns)