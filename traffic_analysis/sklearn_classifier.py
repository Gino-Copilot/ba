# sklearn_classifier.py

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

class ScikitLearnTrafficClassifier:
    """
    Trains and evaluates a given scikit-learn model on the provided dataset.
    It also integrates with DataVisualizer to generate relevant plots and logs.
    """

    def __init__(self, model, output_manager, data_visualizer, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the ScikitLearnTrafficClassifier.

        Args:
            model: A scikit-learn compatible model (e.g., RandomForest, SVC, etc.).
            output_manager: Instance of OutputManager for saving outputs.
            data_visualizer: Instance of DataVisualizer for plotting metrics.
            test_size: Fraction of the dataset to use for testing.
            random_state: Random seed for reproducible splits and model consistency.
        """
        self.model = model
        self.output_manager = output_manager
        self.data_visualizer = data_visualizer

        self.test_size = test_size
        self.random_state = random_state

        # These will be set during the train() method
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

        self.scaler = StandardScaler()
        logging.info(f"ScikitLearnTrafficClassifier initialized for model: {self.model.__class__.__name__}")

    def train(self, df: pd.DataFrame, target_column: str = 'label') -> Dict[str, Any]:
        """
        Trains the model on the given DataFrame and evaluates on a test split.
        Returns a dictionary of metrics.

        Args:
            df: A DataFrame containing features and a target column.
            target_column: Name of the target column (default is 'label').

        Returns:
            A dictionary containing relevant performance metrics.
        """
        try:
            logging.info("Starting training process...")

            # Basic validation
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            if len(X) == 0:
                raise ValueError("No features to train on.")

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if len(y.unique()) > 1 else None
            )

            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test

            # Train model
            self.model.fit(self.X_train_scaled, self.y_train)
            logging.info("Model training completed.")

            # Make predictions
            y_pred = self.model.predict(self.X_test_scaled)

            # Calculate metrics
            metrics = self._calculate_metrics(y_pred, y_test)
            logging.info(f"Model metrics: {metrics}")

            # Add results to the data_visualizer
            self.data_visualizer.add_model_result(self.model.__class__.__name__, metrics)

            # (Optional) Plot confusion matrix
            self.data_visualizer.plot_confusion_matrix(self.y_test, y_pred, self.model.__class__.__name__, labels=y.unique().tolist())

            # (Optional) Plot ROC/PR curves if predict_proba is available
            if hasattr(self.model, "predict_proba"):
                self.data_visualizer.plot_roc_curve(self.model, self.X_test_scaled, self.y_test, self.model.__class__.__name__)
                self.data_visualizer.plot_precision_recall_curve(self.model, self.X_test_scaled, self.y_test, self.model.__class__.__name__)

            # (Optional) Plot feature importance if available
            if hasattr(self.model, "feature_importances_"):
                self.data_visualizer.plot_feature_importance(self.model, list(X.columns), self.model.__class__.__name__)

            return metrics

        except Exception as e:
            logging.error(f"Error during model training/evaluation: {e}")
            return {}

    def _calculate_metrics(self, y_pred, y_test) -> Dict[str, Any]:
        """
        Calculates various performance metrics (accuracy, f1-score, etc.).
        Returns them in a dictionary for logging/visualization.

        Args:
            y_pred: Predicted labels from the model.
            y_test: True labels.

        Returns:
            A dictionary of metrics, including classification_report + optional ROC-AUC.
        """
        metrics_dict = {}
        try:
            # Classification Report (dict) includes accuracy, precision, recall, f1, etc.
            # By default, classification_report() returns a string,
            # but output_dict=True gives us a structured dict.
            report = classification_report(
                y_test, y_pred,
                output_dict=True,
                zero_division=0  # to handle edge cases
            )
            metrics_dict.update(report)

            # The top-level 'accuracy' key is often more convenient:
            metrics_dict['accuracy'] = report['accuracy']

            # If the model supports predict_proba, we can also compute ROC-AUC
            if hasattr(self.model, "predict_proba"):
                y_prob = self.model.predict_proba(self.X_test_scaled)[:, 1]
                roc_val = roc_auc_score(y_test, y_prob)
                metrics_dict['roc_auc'] = float(roc_val)

        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")

        return metrics_dict
