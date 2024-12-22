# model_selection.py
import logging
from typing import Dict

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# XGBoost import
from xgboost import XGBClassifier


class ModelSelector:
    """
    Provides a list of candidate ML models for traffic classification.
    """

    def __init__(self):
        """
        Initializes the ModelSelector.
        """
        logging.info("ModelSelector initialized.")

    def get_all_models(self) -> Dict[str, object]:
        """
        Returns a dictionary of model_name -> model_instance.

        Returns:
            A dictionary mapping model names (str) to model instances (objects).
        """
        # A minimal set of commonly used models
        models = {
            "LogisticRegression": LogisticRegression(
                solver='liblinear',
                random_state=42
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            "SVC": SVC(
                probability=True,  # needed for predict_proba
                kernel='rbf',
                gamma='scale',
                random_state=42
            ),
            "XGBClassifier": XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        }

        logging.info(f"Available models: {list(models.keys())}")
        return models
