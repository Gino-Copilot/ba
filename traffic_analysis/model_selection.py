import logging
from typing import Dict

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# XGBoost
from xgboost import XGBClassifier


class ModelSelector:
    """
    Provides a list of candidate ML models for traffic classification.
    """

    def __init__(self):
        logging.info("ModelSelector initialized.")

    def get_all_models(self) -> Dict[str, object]:
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
                probability=True,
                kernel='rbf',
                gamma='scale',
                random_state=42
            ),
            "XGBClassifier": XGBClassifier(
                eval_metric='logloss',
                random_state=42
            )
        }
        logging.info(f"Available models: {list(models.keys())}")
        return models
