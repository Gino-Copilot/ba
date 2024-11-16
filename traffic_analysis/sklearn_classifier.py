import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time


class ScikitLearnTrafficClassifier:
    def __init__(self, model):
        self.model = model
        self.scaler = None
        self.label_encoder = None
        self.feature_importance = None
        self.results = {}
        self.best_params = {}

    def perform_grid_search(self, X, y):
        # Grid Search Parameter für verschiedene Modelle
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            "SVM (Linear)": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }

        model_name = str(type(self.model).__name__)
        if model_name in param_grids:
            print(f"\nDurchführe Grid Search für {model_name}...")
            grid_search = GridSearchCV(
                self.model,
                param_grids[model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            self.best_params = grid_search.best_params_
            print(f"Beste Parameter: {self.best_params}")
            return grid_search.best_estimator_
        return self.model

    def plot_feature_importance(self, X):
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            return plt
        return None

    def plot_roc_curve(self, X_test, y_test):
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.tight_layout()
            return plt
        return None

    def evaluate_temporal_independence(self, df):
        """Überprüft die zeitliche Unabhängigkeit der Features"""
        temporal_correlations = {}

        for column in df.drop('label', axis=1).columns:
            if df[column].dtype in [np.float64, np.int64]:
                temporal_correlations[column] = df[column].autocorr()

        return pd.Series(temporal_correlations)

    def train(self, df):
        X = df.drop('label', axis=1)
        y = df['label']

        # Label-Encoding für die Zielvariable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Feature-Skalierung
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Train-Test-Split mit enkodiertem y
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Grid Search und Training
        self.model = self.perform_grid_search(X_train, y_train)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Transformiere die Labels zurück für den Bericht
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)

        print("\nKlassifikationsbericht:",
              classification_report(y_test_original, y_pred_original))
        print("\nKonfusionsmatrix:\n",
              confusion_matrix(y_test_original, y_pred_original))

        # Erstelle Visualisierungen
        results_dir = "analysis_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = str(type(self.model).__name__)

        # Feature Importance Plot
        feature_importance_plot = self.plot_feature_importance(X)
        if feature_importance_plot:
            feature_importance_plot.savefig(f'{results_dir}/feature_importance_{model_name}_{timestamp}.png')
            plt.close()

        # ROC Curve
        roc_plot = self.plot_roc_curve(X_test, y_test)
        if roc_plot:
            roc_plot.savefig(f'{results_dir}/roc_curve_{model_name}_{timestamp}.png')
            plt.close()

        # Zeitliche Unabhängigkeit
        temporal_corr = self.evaluate_temporal_independence(df)
        temporal_corr.to_csv(f'{results_dir}/temporal_correlation_{model_name}_{timestamp}.csv')

        # Speichere beste Parameter
        if self.best_params:
            pd.DataFrame([self.best_params]).to_csv(
                f'{results_dir}/best_parameters_{model_name}_{timestamp}.csv'
            )

        # Feature Importance für Baum-basierte Modelle
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

    def save_results(self, df):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = "nfstream_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                f'{results_dir}/feature_importance_{timestamp}.csv',
                index=False
            )
        df.to_csv(f'{results_dir}/full_dataset_{timestamp}.csv', index=False)