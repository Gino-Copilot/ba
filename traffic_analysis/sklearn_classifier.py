import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import os
import time
from .data_visualizer import DataVisualizer
from .shap_analyzer import SHAPAnalyzer


class ScikitLearnTrafficClassifier:
   def __init__(self, model):
       self.model = model
       self.scaler = None
       self.label_encoder = None
       self.feature_importance = None
       self.results = {}
       self.best_params = {}
       self.visualizer = DataVisualizer()

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

   def train(self, df):
       """Trains and evaluates the model"""
       try:
           # Create output directories
           timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
           results_dir = "analysis_results"
           model_name = self.model.__class__.__name__
           output_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
           os.makedirs(output_dir, exist_ok=True)

           # Prepare data
           X = df.drop('label', axis=1)
           y = df['label']

           # Label encoding and scaling
           self.label_encoder = LabelEncoder()
           y_encoded = self.label_encoder.fit_transform(y)
           self.scaler = StandardScaler()
           X_scaled = self.scaler.fit_transform(X)
           X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

           # Train-test split
           X_train, X_test, y_train, y_test = train_test_split(
               X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
           )

           # Training
           self.model = self.perform_grid_search(X_train, y_train)
           self.model.fit(X_train, y_train)
           y_pred = self.model.predict(X_test)

           # Transform labels back
           y_test_original = self.label_encoder.inverse_transform(y_test)
           y_pred_original = self.label_encoder.inverse_transform(y_pred)

           # Create visualizations
           self.feature_importance = self.visualizer.plot_feature_importance(
               self.model, X, output_dir, timestamp
           )
           roc_auc = self.visualizer.plot_roc_curve(
               self.model, X_test, y_test, output_dir, timestamp
           )
           self.visualizer.plot_confusion_matrix(
               y_test_original, y_pred_original, output_dir, timestamp
           )

           # Optimized SHAP analysis
           print("\nPerforming optimized SHAP analysis...")
           shap_analyzer = SHAPAnalyzer(
               model=self.model,
               output_dir=output_dir,
               max_samples=200  # Limited to 200 samples for faster analysis
           )

           # Global SHAP analysis
           shap_analyzer.explain_global(X_test)

           # Local SHAP analysis for first instance
           shap_analyzer.explain_local(X_test, instance_index=0)

           # Save results
           report = classification_report(y_test_original, y_pred_original, output_dict=True)
           pd.DataFrame(report).transpose().to_csv(
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

   def save_results(self, df):
       """Saves additional results"""
       try:
           timestamp = time.strftime("%Y%m%d-%H%M%S")
           results_dir = "nfstream_results"
           os.makedirs(results_dir, exist_ok=True)

           if self.feature_importance is not None:
               self.feature_importance.to_csv(
                   os.path.join(results_dir, f'feature_importance_{timestamp}.csv'),
                   index=False
               )
           df.to_csv(os.path.join(results_dir, f'full_dataset_{timestamp}.csv'), index=False)

       except Exception as e:
           print(f"Error saving results: {str(e)}")
           raise