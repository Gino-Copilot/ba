# sklearn_classifier.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

class ScikitLearnTrafficClassifier:
    def __init__(self, model, output_manager, data_visualizer):
        self.model = model
        self.output_manager = output_manager
        self.data_visualizer = data_visualizer
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_params = {}
        # Speichern der Test-Daten als Instanzvariablen
        self.X_test = None
        self.y_test = None
        self.X_test_scaled = None
        self.y_test_encoded = None

    def perform_grid_search(self, X, y):
        param_grids = {
            "RandomForestClassifier": {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            },
            "XGBClassifier": {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            },
            "SVC": {
                'C': [1, 10],
                'kernel': ['rbf'],
                'gamma': ['scale']
            }
        }

        model_name = self.model.__class__.__name__
        if model_name in param_grids:
            try:
                grid_search = GridSearchCV(
                    self.model,
                    param_grids[model_name],
                    cv=3,
                    scoring='f1',
                    n_jobs=-1
                )
                grid_search.fit(X, y)
                self.best_params = grid_search.best_params_

                # Speichern der Grid Search Ergebnisse
                results_df = pd.DataFrame(grid_search.cv_results_)
                results_df.to_csv(self.output_manager.get_path(
                    "models", "metrics", f"{model_name}_grid_search_results.csv"
                ))
                return grid_search.best_estimator_
            except Exception as e:
                print(f"Grid search failed for {model_name}: {str(e)}")
                return self.model
        return self.model

    def train(self, df):
        try:
            model_name = self.model.__class__.__name__
            print(f"\nTraining model: {model_name}")

            # 1. Split features and target
            X = df.drop('label', axis=1)
            y = df['label']

            # 2. First split to get a proper test set
            X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 3. Encode labels
            self.label_encoder.fit(y_train_full)
            y_train_encoded = self.label_encoder.transform(y_train_full)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)

            # 4. Scale features
            self.scaler.fit(X_train_full)
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train_full),
                columns=X.columns
            )
            self.X_test_scaled = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=X.columns
            )

            # 5. Train model with grid search
            self.model = self.perform_grid_search(X_train_scaled, y_train_encoded)
            self.model.fit(X_train_scaled, y_train_encoded)
            y_pred = self.model.predict(self.X_test_scaled)

            # 6. Evaluate
            report = classification_report(self.y_test_encoded, y_pred, output_dict=True)
            roc_auc = None
            if hasattr(self.model, 'predict_proba'):
                roc_auc = roc_auc_score(self.y_test_encoded,
                                      self.model.predict_proba(self.X_test_scaled)[:, 1])

            # Store results
            metrics = {
                'accuracy': report['accuracy'],
                'weighted avg': report['weighted avg'],
                'roc_auc': roc_auc,
                'best_params': self.best_params
            }

            # Visualizations
            self.data_visualizer.add_model_result(model_name, metrics)
            if hasattr(self.model, 'predict_proba'):
                self.data_visualizer.plot_roc_curve(
                    self.model, self.X_test_scaled, self.y_test_encoded, model_name
                )
            if hasattr(self.model, 'feature_importances_'):
                self.data_visualizer.plot_feature_importance(
                    self.model, X.columns, model_name
                )

            return metrics

        except Exception as e:
            print(f"Error in training {model_name}: {str(e)}")
            raise