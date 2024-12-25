# file: sklearn_classifier.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


class ScikitLearnTrafficClassifier:
    """
    A wrapper class for scikit-learn classifiers with integrated preprocessing,
    optional GridSearch, cross-validation, and metric calculation.
    """

    def __init__(
        self,
        model,
        output_manager,
        data_visualizer,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        param_grid: Optional[dict] = None,
        gridsearch_scoring: str = "accuracy"
    ):
        """
        Args:
            model: A scikit-learn classifier instance
            output_manager: Instance handling file paths and saving
            data_visualizer: Instance for creating visualizations
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed (default: 42)
            cv_folds: Number of cross-validation folds (default: 5)
            param_grid: Dict of hyperparameters for optional GridSearchCV (default: None)
                        e.g. {"model__C": [0.1, 1, 10]}
            gridsearch_scoring: Metric name for scoring in GridSearchCV (default: "accuracy")
        """
        self.model = model
        self.output_manager = output_manager
        self.data_visualizer = data_visualizer
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.param_grid = param_grid or {}  # empty if none provided
        self.gridsearch_scoring = gridsearch_scoring

        self.X_test = None
        self.y_test = None
        self.best_estimator_ = None
        self.pipeline = None

    def train(self, df: pd.DataFrame, target_column: str = 'label') -> Dict[str, Any]:
        """
        Trains the model (with or without GridSearch) and returns performance metrics.

        Steps:
          1) Validate DataFrame
          2) Split into features (X) and target (y)
          3) Build pipeline: [StandardScaler -> model]
          4) If param_grid is not empty, run GridSearchCV -> best_estimator_
          5) Predict on test set, produce classification_report
          6) Return metrics dict

        Args:
            df: DataFrame containing features and target
            target_column: Name of the target column

        Returns:
            A dict with various performance metrics + possibly best_params
        """
        try:
            if not isinstance(df, pd.DataFrame):
                logging.error("Input must be a pandas DataFrame.")
                return {}

            if df.empty or df.shape[1] == 0:
                logging.error("DataFrame is empty or has no columns.")
                return {}

            if target_column not in df.columns:
                logging.error(f"Target column '{target_column}' not found in {df.columns.tolist()}")
                return {}

            # Need at least 2 classes in the target
            unique_labels = df[target_column].unique()
            if len(unique_labels) < 2:
                logging.error(f"Need at least 2 classes, found only: {unique_labels}")
                return {}

            # Check NaN/Inf
            nan_cols = df.columns[df.isna().any()].tolist()
            inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
            if nan_cols or inf_cols:
                logging.error(f"Found NaN in columns: {nan_cols}")
                logging.error(f"Found Inf in columns: {inf_cols}")
                return {}

            # Split features/target
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            # Check numeric columns
            for col in X.columns:
                if not np.issubdtype(X[col].dtype, np.number):
                    logging.error(f"Non-numeric column found: {col} ({X[col].dtype})")
                    return {}

            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            # Basic pipeline
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', self.model)
            ])

            if self.param_grid:
                # Perform GridSearchCV
                logging.info(f"Running GridSearch with param_grid={self.param_grid}")
                from sklearn.model_selection import GridSearchCV

                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=self.param_grid,
                    scoring=self.gridsearch_scoring,
                    cv=self.cv_folds,
                    n_jobs=-1,
                    verbose=1
                )
                grid.fit(X_train, y_train)

                self.pipeline = grid.best_estimator_  # best pipeline
                best_params = grid.best_params_
                best_score = grid.best_score_
                logging.info(f"Best params: {best_params}")
                logging.info(f"Best CV score={best_score:.4f} ({self.gridsearch_scoring})")
            else:
                # Direct fit without GridSearch
                logging.info("No param_grid provided; fitting pipeline directly.")
                pipeline.fit(X_train, y_train)
                self.pipeline = pipeline

            # Predict on test set
            y_pred = self.pipeline.predict(X_test)

            # Save references
            self.X_test = X_test
            self.y_test = y_test
            self.best_estimator_ = self.pipeline

            # Classification report
            from sklearn.metrics import classification_report
            metrics = classification_report(y_test, y_pred, output_dict=True)

            # If we used gridsearch, store best_* in metrics
            if self.param_grid:
                metrics["gridsearch_best_params"] = best_params
                metrics["gridsearch_best_score"] = best_score

            # Log final results
            accuracy = metrics.get("accuracy", None)
            if accuracy is not None:
                logging.info(f"Test Accuracy={accuracy:.3f}")
            else:
                logging.info("No accuracy found in classification_report.")

            return metrics

        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            return {}
