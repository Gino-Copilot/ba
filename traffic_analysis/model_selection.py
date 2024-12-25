# file: model_selector.py

import logging
from typing import Dict, Tuple, Optional

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    logging.warning("xgboost is not installed, XGBClassifier will be unavailable.")


class ModelSelector:
    """
    Provides a list of candidate ML models for traffic classification,
    optionally including a param_grid for each model if you want to use GridSearch.
    """

    def __init__(self):
        logging.info("ModelSelector initialized.")

    def get_all_models(self) -> Dict[str, Tuple[object, dict]]:
        """
        Returns a dict:
          model_name -> (model_instance, param_grid_dict)

        If no param grid is desired for a model, provide an empty dict.
        """
        # Example param grids for demonstration:
        # (Remember to prefix parameters with "model__" to match the Pipeline's "model" step)
        param_grid_lr = {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"]  # for solver='liblinear'
        }

        param_grid_rf = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
        }

        param_grid_svc = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }

        # XGBoost might require different param names:
        # (only if xgboost is installed)
        param_grid_xgb = {
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5, 7]
        }

        # Initialize each model
        lr = LogisticRegression(solver='liblinear', random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        svc = SVC(probability=True, kernel='rbf', gamma='scale', random_state=42)

        # If XGB is installed, create it; otherwise skip
        xgb = XGBClassifier(eval_metric='logloss', random_state=42) if XGBClassifier else None

        # Build dictionary
        models = {
            "LogisticRegression": (
                lr,
                param_grid_lr
            ),
            "RandomForestClassifier": (
                rf,
                param_grid_rf
            ),
            "SVC": (
                svc,
                param_grid_svc
            )
        }

        if xgb is not None:
            models["XGBClassifier"] = (xgb, param_grid_xgb)

        logging.info(f"Available models (with param grids): {list(models.keys())}")
        return models
