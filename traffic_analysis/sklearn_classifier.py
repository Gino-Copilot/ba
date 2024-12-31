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
    A wrapper for scikit-learn classifiers with optional GridSearch,
    cross-validation, and metric reporting.
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
        model: scikit-learn classifier
        output_manager: handles output paths
        data_visualizer: for optional plots
        test_size: fraction for test split
        random_state: random seed
        cv_folds: cross-val folds
        param_grid: hyperparams for GridSearch
        gridsearch_scoring: scoring metric for GridSearch
        """
        self.model = model
        self.output_manager = output_manager
        self.data_visualizer = data_visualizer
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.param_grid = param_grid or {}
        self.gridsearch_scoring = gridsearch_scoring

        self.X_test = None
        self.y_test = None
        self.best_estimator_ = None
        self.pipeline = None

    def train(self, df: pd.DataFrame, target_column: str = 'label') -> Dict[str, Any]:
        """
        Trains the model and returns performance metrics.

        Steps:
          1) Validate input
          2) Split X/y
          3) Build pipeline [Scaler -> model]
          4) Optional GridSearchCV
          5) Predict + classification report
          6) Return metrics
          7) Save GridSearch results if param_grid
        """
        try:
            # 1) Validate input
            if not isinstance(df, pd.DataFrame):
                logging.error("Input must be a DataFrame.")
                return {}
            if df.empty or df.shape[1] == 0:
                logging.error("DataFrame is empty or has no columns.")
                return {}
            if target_column not in df.columns:
                logging.error(f"Target column '{target_column}' not found.")
                return {}
            unique_labels = df[target_column].unique()
            if len(unique_labels) < 2:
                logging.error(f"Need >=2 classes, found only: {unique_labels}")
                return {}

            # Check NaN/Inf
            nan_cols = df.columns[df.isna().any()].tolist()
            inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
            if nan_cols or inf_cols:
                logging.error(f"NaN in: {nan_cols}, Inf in: {inf_cols}")
                return {}

            # 2) Split X/y
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            # Drop non-numeric
            non_numeric = [c for c in X.columns if X[c].dtype == object]
            if non_numeric:
                logging.info(f"Dropping non-numeric cols: {non_numeric}")
                X.drop(columns=non_numeric, inplace=True, errors='ignore')

            # Check numeric
            for col in X.columns:
                if not np.issubdtype(X[col].dtype, np.number):
                    logging.error(f"Column '{col}' not numeric.")
                    return {}

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            # 3) Pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', self.model)
            ])

            # 4) GridSearch or direct fit
            if self.param_grid:
                logging.info(f"GridSearch with param_grid={self.param_grid}")
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=self.param_grid,
                    scoring=self.gridsearch_scoring,
                    cv=self.cv_folds,
                    n_jobs=-1,
                    verbose=1
                )
                grid.fit(X_train, y_train)

                self.pipeline = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_

                logging.info(f"Best params: {best_params}")
                logging.info(f"Best CV score={best_score:.4f}")

                # Save GridSearch results
                model_type = type(self.model).__name__
                cv_results_df = pd.DataFrame(grid.cv_results_)
                cv_csv_path = self.output_manager.get_path("training", "gridsearch", f"{model_type}_cv_results.csv")
                cv_results_df.to_csv(cv_csv_path, index=False)
                logging.info(f"GridSearch results saved to: {cv_csv_path}")

                best_params_path = self.output_manager.get_path("training", "gridsearch", f"{model_type}_best_params.txt")
                with open(best_params_path, "w") as f:
                    f.write("Best Params:\n")
                    f.write(str(best_params) + "\n")
                    f.write(f"Best Score: {best_score:.4f}\n")

            else:
                logging.info("No param_grid provided; fitting directly.")
                pipeline.fit(X_train, y_train)
                self.pipeline = pipeline

            # 5) Predict + metrics
            y_pred = self.pipeline.predict(X_test)
            self.X_test = X_test
            self.y_test = y_test
            self.best_estimator_ = self.pipeline

            metrics = classification_report(y_test, y_pred, output_dict=True)
            if self.param_grid:
                metrics["gridsearch_best_params"] = best_params
                metrics["gridsearch_best_score"] = best_score

            acc = metrics.get("accuracy", None)
            if acc is not None:
                logging.info(f"Test Accuracy={acc:.3f}")
            else:
                logging.info("No accuracy reported in classification_report.")

            # 6) Return metrics
            return metrics

        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            return {}
