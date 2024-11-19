import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datetime import datetime
import os
from .data_visualizer import DataVisualizer
from .shap_analyzer import SHAPAnalyzer


class ScikitLearnTrafficClassifier:
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.best_params = {}
        self.visualizer = DataVisualizer()

        # Store results for metrics
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
        """Performs grid search for supported models"""
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
            print(f"Best parameters: {self.best_params}")
            return grid_search.best_estimator_
        return self.model

    def evaluate_temporal_independence(self, df):
        """Checks temporal independence of features"""
        temporal_correlations = {}
        for column in df.drop('label', axis=1).columns:
            if df[column].dtype in [np.float64, np.int64]:
                temporal_correlations[column] = df[column].autocorr()
        return pd.Series(temporal_correlations)

    def _create_timestamp(self):
        """Create timestamp for file naming"""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def _create_output_dir(self, timestamp):
        """Create output directory for results"""
        results_dir = "analysis_results"
        model_name = self.model.__class__.__name__
        output_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def train(self, df):
        """Trains and evaluates the model"""
        try:
            # Create timestamp and output directory
            timestamp = self._create_timestamp()
            output_dir = self._create_output_dir(timestamp)

            # Prepare data
            X = df.drop('label', axis=1)
            y = df['label']

            # Label encoding and scaling
            y_encoded = self.label_encoder.fit_transform(y)
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Model training
            self.model = self.perform_grid_search(self.X_train, self.y_train)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

            # Transform labels back for reporting
            y_test_original = self.label_encoder.inverse_transform(self.y_test)
            y_pred_original = self.label_encoder.inverse_transform(self.y_pred)

            # Calculate metrics
            self.report = classification_report(y_test_original, y_pred_original, output_dict=True)
            self.accuracy = self.report['accuracy']
            self.precision_weighted = self.report['weighted avg']['precision']
            self.recall_weighted = self.report['weighted avg']['recall']
            self.f1_weighted = self.report['weighted avg']['f1-score']

            # Calculate ROC AUC if applicable
            if hasattr(self.model, 'predict_proba'):
                self.roc_auc = roc_auc_score(self.y_test,
                                             self.model.predict_proba(self.X_test)[:, 1])

            # Create visualizations
            self.visualizer.plot_feature_importance(self.model, X, output_dir, timestamp)
            self.visualizer.plot_confusion_matrix(y_test_original, y_pred_original,
                                                  output_dir, timestamp)
            roc_auc = self.visualizer.plot_roc_curve(self.model, self.X_test,
                                                     self.y_test, output_dir, timestamp)

            # Save performance metrics
            metrics = self.get_metrics()
            self.visualizer.add_model_result(self.model.__class__.__name__, metrics)

            # SHAP Analysis
            print("\nPerforming optimized SHAP analysis...")
            shap_analyzer = SHAPAnalyzer(
                model=self.model,
                output_dir=output_dir,
                max_samples=200
            )
            shap_analyzer.explain_global(self.X_test)
            shap_analyzer.explain_local(self.X_test, instance_index=0)

            # Save results to files
            pd.DataFrame(self.report).transpose().to_csv(
                os.path.join(output_dir, f'classification_report_{timestamp}.csv')
            )

            if self.best_params:
                pd.DataFrame([self.best_params]).to_csv(
                    os.path.join(output_dir, f'best_parameters_{timestamp}.csv')
                )

            temporal_corr = self.evaluate_temporal_independence(df)
            temporal_corr.to_csv(
                os.path.join(output_dir, f'temporal_correlation_{timestamp}.csv')
            )

            # Print results
            print("\nClassification Report:")
            print(classification_report(y_test_original, y_pred_original))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test_original, y_pred_original))
            if roc_auc:
                print(f"\nROC AUC Score: {roc_auc:.4f}")
            print(f"\nAll results have been saved to: {output_dir}")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def get_metrics(self):
        """Return the classification metrics as a dictionary"""
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
        """Save model results and visualizations"""
        # This method is now handled within the train method
        # Keeping it for backward compatibility
        pass