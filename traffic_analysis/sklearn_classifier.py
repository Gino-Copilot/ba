import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import os
import matplotlib.pyplot as plt
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
        self.roc_auc = None
        self.report = None

        # Performance summary for multiple models
        self.performance_summary = []

    def perform_grid_search(self, X, y):
        """
        Perform grid search for model optimization

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Model with best parameters
        """
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

    def train(self, df):
        """
        Train and evaluate the model

        Args:
            df: Input DataFrame with features and target
        """
        try:
            model_name = self.model.__class__.__name__
            print(f"\n{'=' * 50}")
            print(f"Training model: {model_name}")
            print(f"{'=' * 50}")

            # Prepare data
            X = df.drop('label', axis=1)
            y = df['label']

            y_encoded = self.label_encoder.fit_transform(y)
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Train model with grid search
            self.model = self.perform_grid_search(self.X_train, self.y_train)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

            y_test_original = self.label_encoder.inverse_transform(self.y_test)
            y_pred_original = self.label_encoder.inverse_transform(self.y_pred)

            # Metrics
            self.report = classification_report(y_test_original, y_pred_original, output_dict=True)
            self.roc_auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1]) if hasattr(
                self.model, 'predict_proba') else None

            # Save classification report
            report_path = self.output_manager.get_path(
                "models", "metrics", f"{model_name}_classification_report.csv"
            )
            pd.DataFrame(self.report).transpose().to_csv(report_path)

            # Add model performance to summary
            self.performance_summary.append({
                "Model": model_name,
                "Accuracy": self.report['accuracy'],
                "F1-Score": self.report['weighted avg']['f1-score'],
                "Precision": self.report['weighted avg']['precision'],
                "Recall": self.report['weighted avg']['recall'],
                "AUC": self.roc_auc,
                "Best Parameters": self.best_params
            })

            # Visualizations
            self._plot_roc_curve(y_test_original)
            self._save_model_summary()

        except Exception as e:
            print(f"Error during training of {model_name}: {str(e)}")
            raise

    def _save_model_summary(self):
        """
        Save a detailed summary report for all models
        """
        try:
            # Create a DataFrame from performance_summary
            summary_df = pd.DataFrame(self.performance_summary)

            # Handle missing values for AUC or Best Parameters
            summary_df['AUC'] = summary_df['AUC'].fillna('N/A')
            summary_df['Best Parameters'] = summary_df['Best Parameters'].apply(
                lambda x: x if x else 'Not Available'
            )

            # Save as CSV
            summary_path = self.output_manager.get_path("reports", "summaries", "performance_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved performance summary to {summary_path}")

            # Save as text
            text_summary_path = self.output_manager.get_path("reports", "summaries", "performance_summary.txt")
            with open(text_summary_path, "w") as f:
                f.write("=== Model Performance Summary ===\n\n")
                for _, row in summary_df.iterrows():
                    f.write(f"Model: {row['Model']}\n")
                    f.write(f"  - Accuracy: {row['Accuracy']:.4f}\n")
                    f.write(f"  - F1-Score: {row['F1-Score']:.4f}\n")
                    f.write(f"  - Precision: {row['Precision']:.4f}\n")
                    f.write(f"  - Recall: {row['Recall']:.4f}\n")
                    f.write(f"  - AUC: {row['AUC']}\n")
                    f.write(f"  - Best Parameters: {row['Best Parameters']}\n")
                    f.write("-" * 40 + "\n")
            print(f"Saved detailed text summary to {text_summary_path}")

        except Exception as e:
            print(f"Error saving model summary: {str(e)}")

    def _plot_roc_curve(self, y_test_original):
        """
        Plot ROC curve for the model
        """
        try:
            if not hasattr(self.model, 'predict_proba'):
                print(f"Skipping ROC curve for {self.model.__class__.__name__}: No predict_proba method.")
                return

            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{self.model.__class__.__name__} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')

            plot_path = self.output_manager.get_path(
                "reports", "visualizations", f"{self.model.__class__.__name__}_roc_curve.png"
            )
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved ROC curve to {plot_path}")
        except Exception as e:
            print(f"Error while plotting ROC curve: {str(e)}")

    def visualize_comparisons(self):
        """
        Create comparison visualizations for all models
        """
        try:
            summary_df = pd.DataFrame(self.performance_summary)
            if summary_df.empty:
                print("No models to compare.")
                return

            # F1-Score comparison
            plt.figure(figsize=(10, 6))
            summary_df.plot.bar(x="Model", y=["Accuracy", "F1-Score"], rot=45)
            plt.title("Model Comparison: Accuracy and F1-Score")
            plt.ylabel("Score")
            plt.tight_layout()

            comparison_path = self.output_manager.get_path(
                "reports", "visualizations", "model_comparison.png"
            )
            plt.savefig(comparison_path, bbox_inches='tight')
            plt.close()
            print(f"Saved model comparison plot to {comparison_path}")
        except Exception as e:
            print(f"Error while creating comparison visualization: {str(e)}")
