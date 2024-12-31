import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
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
        """
        Main entry point for global SHAP analysis.
        X: Feature DataFrame
        y: (Optional) target series if needed
        """
        try:
            logging.info(f"Starting global SHAP analysis for {self.model_name}...")

            if X is None or len(X) == 0:
                raise ValueError("Empty or invalid input data to SHAPAnalyzer.")

            # Possibly sample large data
            X_sampled = self._sample_data(X)

            # Initialize the SHAP explainer
            self.explainer = self._initialize_explainer(X_sampled)
            if self.explainer is None:
                logging.warning("Failed to initialize SHAP explainer.")
                return None, None

            # Calculate SHAP values
            self.shap_values = self._calculate_shap_values(X_sampled)
            if self.shap_values is None:
                logging.warning("Failed to compute SHAP values.")
                return None, None

            # Create plots
            self._create_visualizations(X_sampled)

            # Save CSV results
            self._save_analysis_results(X_sampled)

            logging.info("Global SHAP analysis completed successfully!")
            return self.explainer, self.shap_values

        except Exception as e:
            logging.error(f"Error in global SHAP analysis: {str(e)}")
            return None, None

    def explain_local(self, X, instance_indices):
        """
        Provides local SHAP explanations for given instance indices.
        """
        try:
            if self.explainer is None:
                logging.info("No explainer found. Initializing for local explanations.")
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
        """
        Creates a force plot for a single instance by index.
        """
        try:
            if isinstance(X, pd.DataFrame):
                instance = X.iloc[[idx]]
                feature_names = X.columns
            else:
                instance = X[idx:idx + 1]
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            local_shap_values = self.explainer.shap_values(instance)

            # If shap_values is a list => multiclass => pick class 1 for illustration
            if isinstance(local_shap_values, list) and len(local_shap_values) > 1:
                local_shap_values = local_shap_values[1]
            elif isinstance(local_shap_values, np.ndarray) and local_shap_values.ndim == 3:
                local_shap_values = local_shap_values[:, :, 1]

            self._plot_local_explanation(local_shap_values, instance, feature_names, idx)
        except Exception as e:
            logging.error(f"Error explaining instance {idx}: {str(e)}")

    def _is_tree_based_model(self) -> bool:
        """
        Checks if the model is tree-based (RF, GBDT, XGBoost).
        """
        return isinstance(self.model, (
            RandomForestClassifier,
            GradientBoostingClassifier,
            XGBClassifier
        ))

    def _sample_data(self, X):
        """
        Samples the DataFrame or array if it exceeds max_samples.
        """
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
        """
        Creates a SHAP explainer depending on model type (tree-based or not).
        """
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
        """
        Calls self.explainer.shap_values(X) and handles multiclass or 3D arrays.
        """
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            return shap_values
        except Exception as e:
            logging.error(f"Error calculating SHAP values: {str(e)}")
            return None

    def _create_visualizations(self, X):
        """
        Calls the main plotting methods if shap_values is not None.
        """
        try:
            if self.shap_values is None:
                return
            self._plot_summary(X)  # Default (dot) summary plot
            self._plot_beeswarm(X)  # BeeSwarm = dot-plot
            self._plot_summary_bar(X)  # Bar-Plot summary
            self._plot_mean_abs_importance(X)  # Mean absolute shap importance
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")

    def _plot_summary(self, X):
        """
        Plots a SHAP summary plot (dot plot) and saves it (the "classic" beeswarm).
        """
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                plot_type="dot",
                max_display=self.max_display,
                show=False
            )
            plt.title(f"{self.model_name} - SHAP Summary (Dot/Beeswarm)")
            path = self.output_manager.get_path("models", "shap", "summary_plot.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"SHAP summary plot saved: {path}")
        except Exception as e:
            logging.error(f"Error creating SHAP summary plot: {str(e)}")

    def _plot_beeswarm(self, X):
        """
        Alternative beeswarm plot.
        """
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                plot_type="dot",
                max_display=self.max_display,
                show=False
            )
            plt.title(f"{self.model_name} - SHAP Beeswarm Plot")
            path = self.output_manager.get_path("models", "shap", "beeswarm_plot.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"SHAP beeswarm plot saved: {path}")
        except Exception as e:
            logging.error(f"Error creating SHAP beeswarm plot: {str(e)}")

    def _plot_summary_bar(self, X):
        """
        Plots a SHAP summary plot in 'bar' mode and saves it.
        """
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                plot_type="bar",
                max_display=self.max_display,
                show=False
            )
            plt.title(f"{self.model_name} - SHAP Summary Bar")
            path = self.output_manager.get_path("models", "shap", "summary_bar_plot.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"SHAP bar summary plot saved: {path}")
        except Exception as e:
            logging.error(f"Error creating SHAP bar summary plot: {str(e)}")

    def _plot_mean_abs_importance(self, X):
        """
        Creates a bar chart of mean absolute SHAP values.
        """
        try:
            # Convert shap_values to numpy array if it isn't already
            shap_values_array = np.array(self.shap_values)

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

            # Get feature names
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(shap_values_array.shape[1])]

            # Create DataFrame for sorting
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            })

            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=True).reset_index(drop=True)

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f"{self.model_name} - Feature Importance")

            plot_path = self.output_manager.get_path("models", "shap", "shap_feature_importance_meanabs.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Save importance values to CSV
            csv_path = self.output_manager.get_path("models", "shap", "feature_importance.csv")
            importance_df.to_csv(csv_path, index=False)

            logging.info(f"Feature importance plot and CSV saved")
        except Exception as e:
            logging.error(f"Error creating feature importance plot: {str(e)}")

    def _plot_local_explanation(self, shap_values, instance, feature_names, idx):
        """
        Creates a force plot for a single instance and saves it.
        """
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
        """
        Saves SHAP values and feature importance to CSV files.
        """
        try:
            # Save SHAP values with appropriate column names
            if isinstance(X, pd.DataFrame):
                shap_df = pd.DataFrame(
                    self.shap_values,
                    columns=X.columns
                )
            else:
                shap_df = pd.DataFrame(
                    self.shap_values,
                    columns=[f"feature_{i}" for i in range(X.shape[1])]
                )

            shap_path = self.output_manager.get_path("models", "shap", "shap_values.csv")
            shap_df.to_csv(shap_path, index=False)

            # Calculate and save mean absolute SHAP values
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

            if isinstance(X, pd.DataFrame):
                feat_names = X.columns
            else:
                feat_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

            importance_df = pd.DataFrame({
                'feature': feat_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)

            importance_path = self.output_manager.get_path("models", "shap", "shap_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)

            logging.info("SHAP analysis results saved to CSV files")

        except Exception as e:
            logging.error(f"Error saving SHAP analysis results: {str(e)}")