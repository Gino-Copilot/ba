import matplotlib
matplotlib.use('Agg')  # Backend festlegen
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


class SHAPAnalyzer:
    """
    Performs SHAP-based model interpretability analysis.
    """

    def __init__(self, model, output_manager, max_display=20, max_samples=500):
        self.model = model
        self.output_manager = output_manager
        self.max_display = max_display
        self.max_samples = max_samples
        self.model_name = self.model.__class__.__name__
        self.explainer = None
        self.shap_values = None

        warnings.filterwarnings('ignore')
        plt.style.use('seaborn-v0_8-darkgrid')
        logging.info(f"Initialized SHAP Analyzer for model: {self.model_name}")

    def explain_global(self, X: pd.DataFrame, y: pd.Series = None):
        try:
            logging.info(f"Starting global SHAP analysis for {self.model_name}...")

            if X is None or len(X) == 0:
                raise ValueError("Empty or invalid input data to SHAPAnalyzer.")

            X_sampled = self._sample_data(X)

            self.explainer = self._initialize_explainer(X_sampled)
            if self.explainer is None:
                logging.warning("Failed to initialize SHAP explainer.")
                return None, None

            self.shap_values = self._calculate_shap_values(X_sampled)
            if self.shap_values is None:
                logging.warning("Failed to compute SHAP values.")
                return None, None

            self._create_visualizations(X_sampled)
            self._save_analysis_results(X_sampled)

            logging.info("Global SHAP analysis completed successfully!")
            return self.explainer, self.shap_values

        except Exception as e:
            logging.error(f"Error in global SHAP analysis: {str(e)}")
            return None, None

    def explain_local(self, X, instance_indices):
        try:
            if self.explainer is None:
                logging.info("Explainer not found. Initializing for local explanations.")
                X_sampled = self._sample_data(X)
                self.explainer = self._initialize_explainer(X_sampled)

            if not instance_indices:
                logging.warning("No instance indices provided for local explanations.")
                return

            logging.info(f"Creating local explanations for {len(instance_indices)} instances...")
            for idx in instance_indices:
                self._explain_single_instance(X, idx)

            logging.info("Local explanations completed successfully!")
        except Exception as e:
            logging.error(f"Error in local SHAP analysis: {str(e)}")

    def _explain_single_instance(self, X, idx: int):
        try:
            if isinstance(X, pd.DataFrame):
                instance = X.iloc[[idx]]
                feature_names = X.columns
            else:
                instance = X[idx:idx + 1]
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            local_shap_values = self.explainer.shap_values(instance)
            # Falls hier 3D -> extrahiere nur Klasse 1
            if isinstance(local_shap_values, list) and len(local_shap_values) > 1:
                local_shap_values = local_shap_values[1]
            elif isinstance(local_shap_values, np.ndarray) and local_shap_values.ndim == 3:
                local_shap_values = local_shap_values[:, :, 1]

            self._plot_local_explanation(local_shap_values, instance, feature_names, idx)
        except Exception as e:
            logging.error(f"Error explaining instance {idx}: {str(e)}")

    def _is_tree_based_model(self) -> bool:
        return isinstance(self.model, (
            RandomForestClassifier,
            GradientBoostingClassifier,
            XGBClassifier
        ))

    def _sample_data(self, X):
        try:
            if len(X) <= self.max_samples:
                return X

            logging.info(f"Sampling input data from {len(X)} to {self.max_samples} rows.")
            if isinstance(X, pd.DataFrame):
                return X.sample(n=self.max_samples, random_state=42)
            else:
                indices = np.random.RandomState(42).choice(len(X), self.max_samples, replace=False)
                return X[indices]
        except Exception as e:
            logging.error(f"Error in data sampling: {str(e)}")
            return X

    def _initialize_explainer(self, X):
        try:
            if self._is_tree_based_model():
                logging.info("Using TreeExplainer for a tree-based model.")
                return shap.TreeExplainer(self.model)
            else:
                logging.info("Using KernelExplainer (model is not tree-based).")
                background = shap.sample(X, 100) if len(X) > 100 else X
                return shap.KernelExplainer(self.model.predict_proba, background)
        except Exception as e:
            logging.error(f"Error initializing SHAP explainer: {str(e)}")
            return None

    def _calculate_shap_values(self, X):
        try:
            shap_values = self.explainer.shap_values(X)

            # Falls shap_values = list => n_classes => n-element list
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            # Falls shap_values = 3D => (samples, features, classes)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            return shap_values
        except Exception as e:
            logging.error(f"Error calculating SHAP values: {str(e)}")
            return None

    def _create_visualizations(self, X):
        try:
            if self.shap_values is None:
                return
            self._plot_summary(X)
            self._plot_beeswarm(X)
            self._plot_feature_importance()
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")

    def _plot_summary(self, X):
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, X,
                max_display=self.max_display, show=False
            )
            plt.title(f"{self.model_name} - SHAP Summary Plot")
            path = self.output_manager.get_path("models", "shap", "summary_plot.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"SHAP summary plot saved: {path}")
        except Exception as e:
            logging.error(f"Error creating SHAP summary plot: {str(e)}")

    def _plot_beeswarm(self, X):
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, X,
                plot_type="bar",
                max_display=self.max_display, show=False
            )
            plt.title(f"{self.model_name} - SHAP Feature Importance")
            path = self.output_manager.get_path("models", "shap", "beeswarm_plot.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"SHAP beeswarm plot saved: {path}")
        except Exception as e:
            logging.error(f"Error creating SHAP beeswarm plot: {str(e)}")

    def _plot_feature_importance(self):
        try:
            abs_shap = np.abs(self.shap_values).mean(axis=0)
            features_count = len(abs_shap)
            importance_df = pd.DataFrame({
                'feature': range(features_count),
                'importance': abs_shap
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            plt.bar(range(features_count), importance_df['importance'])
            plt.xticks(range(features_count), importance_df['feature'], rotation=45)
            plt.title(f"{self.model_name} - SHAP Feature Importance (Mean Abs)")
            plt.tight_layout()

            plot_path = self.output_manager.get_path("models", "shap", "feature_importance.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"SHAP feature importance plot saved: {plot_path}")

        except Exception as e:
            logging.error(f"Error creating feature importance plot: {str(e)}")

    def _plot_local_explanation(self, shap_values, instance, feature_names, idx):
        try:
            plt.figure(figsize=(12, 8))
            shap.force_plot(
                self.explainer.expected_value,
                shap_values,
                instance,
                feature_names=feature_names,
                show=False,
                matplotlib=True
            )
            plt.title(f"Local Explanation for Instance {idx}")

            plot_path = self.output_manager.get_path("models", "shap", f"local_explanation_{idx}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Local SHAP explanation plot saved for instance {idx}: {plot_path}")

        except Exception as e:
            logging.error(f"Error creating local explanation plot for instance {idx}: {str(e)}")

    def _save_analysis_results(self, X):
        try:
            shap_values_path = self.output_manager.get_path("models", "shap", "shap_values.csv")

            if isinstance(X, pd.DataFrame):
                shap_df = pd.DataFrame(self.shap_values, columns=X.columns)
            else:
                shap_df = pd.DataFrame(self.shap_values)

            shap_df.to_csv(shap_values_path, index=False)
            logging.info(f"SHAP values saved: {shap_values_path}")

            abs_shap = np.abs(self.shap_values).mean(axis=0)
            if isinstance(X, pd.DataFrame):
                feat_names = X.columns
            else:
                feat_names = [f"feature_{i}" for i in range(len(abs_shap))]

            importance_df = pd.DataFrame({
                'feature': feat_names,
                'importance': abs_shap
            }).sort_values('importance', ascending=False)

            importance_path = self.output_manager.get_path("models", "shap", "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            logging.info(f"SHAP-based feature importance saved: {importance_path}")

        except Exception as e:
            logging.error(f"Error saving SHAP analysis results: {str(e)}")
