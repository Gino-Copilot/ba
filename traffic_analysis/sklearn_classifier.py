import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
from .shap_analyzer import SHAPAnalyzer


class ScikitLearnTrafficClassifier:
    def __init__(self, model, output_manager):
        """
        Initialize the classifier

        Args:
            model: Scikit-learn model instance
            output_manager: Instance of OutputManager for handling output paths
        """
        self.model = model
        self.output_manager = output_manager
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.best_params = {}

        # Store results
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.precision_weighted = None
        self.recall_weighted = None
        self.f1_weighted = None
        self.roc_auc = None
        self.report = None

    def perform_grid_search(self, X, y):
        """
        Perform grid search for model optimization

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Model with best parameters
        """
        # Define parameter grids for different models
        param_grids = {
            "RandomForestClassifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            "XGBClassifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            "SVC": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }

        model_name = self.model.__class__.__name__
        if model_name in param_grids:
            print(f"\nPerforming grid search for {model_name}...")
            grid_search = GridSearchCV(
                self.model,
                param_grids[model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X, y)
            self.best_params = grid_search.best_params_

            # Save grid search results
            results_path = self.output_manager.get_path(
                "models", "metrics", "grid_search_results.csv"
            )
            pd.DataFrame(grid_search.cv_results_).to_csv(results_path)

            print(f"Best parameters: {self.best_params}")
            return grid_search.best_estimator_
        return self.model

    def evaluate_temporal_independence(self, df):
        """
        Check temporal independence of features

        Args:
            df: Input DataFrame

        Returns:
            Series: Temporal correlations
        """
        temporal_correlations = {}
        for column in df.drop('label', axis=1).columns:
            if df[column].dtype in [np.float64, np.int64]:
                temporal_correlations[column] = df[column].autocorr()

        # Save temporal correlations
        corr_path = self.output_manager.get_path(
            "models", "metrics", "temporal_correlations.csv"
        )
        pd.Series(temporal_correlations).to_csv(corr_path)

        return pd.Series(temporal_correlations)

    def train(self, df):
        """
        Train and evaluate the model

        Args:
            df: Input DataFrame with features and target
        """
        try:
            # Get model name for clear output
            model_name = self.model.__class__.__name__
            print(f"\n{'=' * 50}")
            print(f"Training model: {model_name}")
            print(f"{'=' * 50}")

            # Prepare data
            X = df.drop('label', axis=1)
            y = df['label']

            # Encode labels and scale features
            y_encoded = self.label_encoder.fit_transform(y)
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            # Split data into train and test sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            print(f"\nPerforming grid search for {model_name}...")
            # Train model with grid search
            self.model = self.perform_grid_search(self.X_train, self.y_train)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

            # Convert labels back to original format
            y_test_original = self.label_encoder.inverse_transform(self.y_test)
            y_pred_original = self.label_encoder.inverse_transform(self.y_pred)

            # Calculate metrics
            self.report = classification_report(y_test_original, y_pred_original, output_dict=True)
            self.accuracy = self.report['accuracy']
            self.precision_weighted = self.report['weighted avg']['precision']
            self.recall_weighted = self.report['weighted avg']['recall']
            self.f1_weighted = self.report['weighted avg']['f1-score']

            # Calculate ROC AUC if possible
            if hasattr(self.model, 'predict_proba'):
                self.roc_auc = roc_auc_score(self.y_test,
                                             self.model.predict_proba(self.X_test)[:, 1])

            # Save classification report
            report_path = self.output_manager.get_path(
                "models", "metrics", f"{model_name}_classification_report.csv"
            )
            pd.DataFrame(self.report).transpose().to_csv(report_path)

            # Save best parameters if available
            if self.best_params:
                params_path = self.output_manager.get_path(
                    "models", "metrics", f"{model_name}_best_parameters.csv"
                )
                pd.DataFrame([self.best_params]).to_csv(params_path)

            # Save confusion matrix
            conf_matrix = confusion_matrix(y_test_original, y_pred_original)
            matrix_path = self.output_manager.get_path(
                "models", "metrics", f"{model_name}_confusion_matrix.csv"
            )
            pd.DataFrame(conf_matrix).to_csv(matrix_path)

            # Save complete model summary
            self._save_complete_summary(y_test_original, y_pred_original)

            # Perform SHAP analysis
            print(f"\nPerforming SHAP analysis for {model_name}...")
            shap_analyzer = SHAPAnalyzer(
                model=self.model,
                output_manager=self.output_manager,
                max_samples=200
            )
            shap_analyzer.explain_global(self.X_test)

            # Only analyze a few examples for local explanations
            n_examples = min(5, len(self.X_test))
            example_indices = np.random.choice(len(self.X_test), n_examples, replace=False)
            for idx in example_indices:
                shap_analyzer.explain_local(self.X_test, idx)

            # Print results with model name
            print(f"\nResults for {model_name}:")
            self._print_results(y_test_original, y_pred_original)

            print(f"\nAll results for {model_name} have been saved in the results directory")
            print(f"{'=' * 50}\n")

        except Exception as e:
            print(f"Error during training of {model_name}: {str(e)}")
            raise

    def _save_complete_summary(self, y_test_original, y_pred_original):
        """
        Save complete summary of model performance

        Args:
            y_test_original: Original test labels
            y_pred_original: Original prediction labels
        """
        summary_path = self.output_manager.get_path(
            "models", "summaries", "complete_summary.txt"
        )

        with open(summary_path, 'w') as f:
            f.write("=== Model Performance Summary ===\n\n")

            # Basic metrics
            f.write("Basic Metrics:\n")
            f.write(f"Accuracy: {self.accuracy:.4f}\n")
            f.write(f"Weighted Precision: {self.precision_weighted:.4f}\n")
            f.write(f"Weighted Recall: {self.recall_weighted:.4f}\n")
            f.write(f"Weighted F1-Score: {self.f1_weighted:.4f}\n")
            if self.roc_auc:
                f.write(f"ROC AUC: {self.roc_auc:.4f}\n")

            # Classification Report
            f.write("\nDetailed Classification Report:\n")
            f.write(classification_report(y_test_original, y_pred_original))

            # Model Parameters
            f.write("\nModel Parameters:\n")
            for param, value in self.model.get_params().items():
                f.write(f"{param}: {value}\n")

    def _print_results(self, y_test_original, y_pred_original):
        """
        Print model results to console

        Args:
            y_test_original: Original test labels
            y_pred_original: Original prediction labels
        """
        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred_original))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_original, y_pred_original))

        if self.roc_auc:
            print(f"\nROC AUC Score: {self.roc_auc:.4f}")

    def get_metrics(self):
        """
        Get all classification metrics

        Returns:
            dict: Dictionary containing all metrics
        """
        return {
            'accuracy': self.accuracy,
            'weighted avg': {
                'precision': self.precision_weighted,
                'recall': self.recall_weighted,
                'f1-score': self.f1_weighted
            },
            'roc_auc': self.roc_auc if hasattr(self, 'roc_auc') else None
        }

    def save_results(self, df):
        """
        Legacy method kept for backwards compatibility

        Args:
            df: Input DataFrame
        """
        pass  # All saving is now handled in the train method