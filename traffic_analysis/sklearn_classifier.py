import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

class ScikitLearnTrafficClassifier:
    """
    Trains and evaluates a scikit-learn model within a Pipeline (including scaling).
    Uses GridSearchCV for hyperparameter tuning and a separate test set for final evaluation.
    Integrates with DataVisualizer to generate relevant plots and logs.
    """

    def __init__(self,
                 model,
                 output_manager,
                 data_visualizer,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 cv_folds: int = 5):
        self.model = model
        self.output_manager = output_manager
        self.data_visualizer = data_visualizer
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

        self.best_estimator_ = None
        self.X_test = None
        self.y_test = None

        logging.info(f"ScikitLearnTrafficClassifier initialized for model: {self.model.__class__.__name__}")

    def train(self, df: pd.DataFrame, target_column: str = 'label') -> Dict[str, Any]:
        try:
            logging.info("Starting training process with Pipeline and GridSearchCV...")

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

            X = df.drop(columns=[target_column])
            y = df[target_column]
            if X.empty:
                raise ValueError("No features to train on.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if y.nunique() > 1 else None
            )

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", self.model)
            ])

            param_grids = {
                "LogisticRegression": {
                    "model__C": [0.1, 1, 10],
                    "model__penalty": ['l2']
                },
                "RandomForestClassifier": {
                    "model__n_estimators": [50, 100],
                    "model__max_depth": [None, 5, 10]
                },
                "SVC": {
                    "model__C": [0.1, 1, 10],
                    "model__gamma": ['scale', 'auto']
                },
                "XGBClassifier": {
                    "model__n_estimators": [50, 100],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.01, 0.1]
                }
            }

            model_class_name = self.model.__class__.__name__
            if model_class_name not in param_grids:
                logging.warning(f"No param grid found for {model_class_name}. Using empty param grid.")
                grid_params = {}
            else:
                grid_params = param_grids[model_class_name]

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid_params,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )

            logging.info(f"Running GridSearchCV for {model_class_name} with param grid: {grid_params}")
            grid_search.fit(X_train, y_train)

            self.best_estimator_ = grid_search.best_estimator_
            logging.info(f"Best params for {model_class_name}: {grid_search.best_params_}")

            y_pred = self.best_estimator_.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            metrics["best_params"] = grid_search.best_params_
            metrics["cv_results"] = grid_search.cv_results_

            self.X_test = X_test
            self.y_test = y_test

            logging.info(f"Final test metrics for best {model_class_name}: {metrics}")

            self.data_visualizer.add_model_result(model_class_name, metrics)
            self.data_visualizer.plot_confusion_matrix(y_test, y_pred, model_class_name, labels=[0,1])

            # Falls predict_proba:
            if hasattr(self.best_estimator_["model"], "predict_proba"):
                X_test_scaled = self.best_estimator_["scaler"].transform(X_test)
                self.data_visualizer.plot_roc_curve(self.best_estimator_["model"], X_test_scaled, y_test, model_class_name)
                self.data_visualizer.plot_precision_recall_curve(self.best_estimator_["model"], X_test_scaled, y_test, model_class_name)

            # Falls feature_importances_:
            if hasattr(self.best_estimator_["model"], "feature_importances_"):
                self.data_visualizer.plot_feature_importance(
                    self.best_estimator_["model"], list(X.columns), model_class_name
                )

            return metrics

        except Exception as e:
            logging.error(f"Error during model training/evaluation: {e}")
            return {}

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        metrics_dict = {}
        try:
            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )
            metrics_dict.update(report)
            metrics_dict['accuracy'] = report['accuracy']

            if hasattr(self.best_estimator_["model"], "predict_proba"):
                X_test_scaled = self.best_estimator_["scaler"].transform(self.X_test)
                y_prob = self.best_estimator_["model"].predict_proba(X_test_scaled)[:, 1]
                metrics_dict['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")

        return metrics_dict
