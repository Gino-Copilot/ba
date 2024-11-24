import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings that might occur with SHAP


class SHAPAnalyzer:
    def __init__(self, model, output_manager, max_samples=100, background_sample_size=20):
        """
        Initialize SHAP Analyzer

        Args:
            model: The trained model to analyze
            output_manager: Instance of OutputManager for handling output paths
            max_samples: Maximum number of samples for SHAP analysis (default: 100 for speed)
            background_sample_size: Number of background samples for KernelExplainer (default: 20 for speed)
        """
        self.model = model
        self.output_manager = output_manager
        self.max_samples = max_samples
        self.background_sample_size = background_sample_size
        self.model_name = self.model.__class__.__name__
        self.output_manager.set_current_model(self.model_name)
        print(f"Initialized SHAP Analyzer for model: {self.model_name}")

    def _is_tree_based_model(self):
        """
        Check if the model is tree-based and supported by TreeExplainer
        """
        tree_based_models = (
            RandomForestClassifier,
            GradientBoostingClassifier,
            XGBClassifier,
        )
        return isinstance(self.model, tree_based_models)

    def _sample_data(self, X, random_state=42):
        """
        Reduce dataset to manageable size for SHAP analysis

        Args:
            X: Input features
            random_state: Random seed for reproducibility

        Returns:
            Reduced dataset
        """
        if len(X) <= self.max_samples:
            return X

        if isinstance(X, pd.DataFrame):
            # Optimized: Use fewer samples during development
            _, X_sampled = train_test_split(X, train_size=self.max_samples, random_state=random_state)
        else:
            indices = np.random.RandomState(random_state).choice(len(X), self.max_samples, replace=False)
            X_sampled = X[indices]

        return X_sampled

    def _prepare_background_data(self, X):
        """
        Prepare background data for KernelExplainer

        Args:
            X: Input features

        Returns:
            Array or DataFrame with reduced background data
        """
        if isinstance(X, pd.DataFrame):
            # Optimized: Use shap.sample for background data
            background_data = shap.sample(X, self.background_sample_size)
        else:
            indices = np.random.choice(len(X), self.background_sample_size, replace=False)
            background_data = X[indices]

        print(f"Using {len(background_data)} background data samples for KernelExplainer.")
        return background_data

    def explain_global(self, X):
        """
        Create unified global model explanations using SHAP

        Args:
            X: Input features to explain
        """
        print(f"\nStarting SHAP analysis with maximum {self.max_samples} samples...")
        self.output_manager.set_current_model(self.model_name)

        # Reduce dataset size
        X_sampled = self._sample_data(X)
        n_features = X_sampled.shape[1]

        try:
            # Choose appropriate explainer
            if self._is_tree_based_model():
                # Optimized: Use approximate=True for faster TreeExplainer
                explainer = shap.TreeExplainer(self.model, approximate=True)
                # Original:
                # explainer = shap.TreeExplainer(self.model)
            else:
                print(f"Model {self.model_name} is not tree-based. Using KernelExplainer.")
                background_data = self._prepare_background_data(X_sampled)
                explainer = shap.KernelExplainer(self.model.predict, background_data)

            shap_values = explainer.shap_values(X_sampled)
            print("SHAP values calculated successfully")

            # For binary classification or single output
            if not isinstance(shap_values, list):
                shap_values = [shap_values]

            # Create plots for each class
            for i, class_shap_values in enumerate(shap_values):
                print(f"Creating plots for class {i}")

                # Optimized: Limit features and reduce plot detail for speed
                max_display = 10  # Only show the top 10 features
                # Original: max_display = n_features

                # Summary Bar Plot
                plt.figure(figsize=(12, max(8, n_features * 0.3)))
                shap.summary_plot(
                    class_shap_values,
                    X_sampled,
                    plot_type="bar",
                    max_display=max_display,
                    show=False
                )
                plt.title(f'{self.model_name} - Feature Importance (Class {i})')
                plt.tight_layout()

                bar_plot_path = self.output_manager.get_path(
                    "models", "shap", f"summary_bar_plot_class_{i}.png"
                )
                plt.savefig(bar_plot_path, bbox_inches="tight", dpi=150)  # Optimized: Reduce DPI for faster saving
                # Original: plt.savefig(bar_plot_path, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Saved bar plot to: {bar_plot_path}")

                # Beeswarm Plot
                plt.figure(figsize=(12, max(8, n_features * 0.3)))
                shap.summary_plot(
                    class_shap_values,
                    X_sampled,
                    plot_type="dot",
                    max_display=max_display,
                    show=False
                )
                plt.title(f'{self.model_name} - Feature Impact Distribution (Class {i})')
                plt.tight_layout()

                beeswarm_path = self.output_manager.get_path(
                    "models", "shap", f"beeswarm_plot_class_{i}.png"
                )
                plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)  # Optimized: Reduce DPI
                # Original: plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Saved beeswarm plot to: {beeswarm_path}")

            print("Global SHAP analysis completed successfully!")
            return explainer, shap_values

        except Exception as e:
            print(f"Error during global SHAP analysis: {str(e)}")
            return None, None

    def explain_local(self, X, instance_index):
        """
        Create local explanations for a single instance

        Args:
            X: Input features
            instance_index: Index of instance to explain
        """
        try:
            self.output_manager.set_current_model(self.model_name)
            print(f"\nCreating local explanation for {self.model_name}, instance {instance_index}")

            # Prepare data
            if isinstance(X, pd.DataFrame):
                instance = X.iloc[[instance_index]]
                feature_names = X.columns
            else:
                instance = X[instance_index:instance_index + 1]
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Choose appropriate explainer
            if self._is_tree_based_model():
                explainer = shap.TreeExplainer(self.model, approximate=True)
                # Original: explainer = shap.TreeExplainer(self.model)
            else:
                print(f"Model {self.model_name} is not tree-based. Using KernelExplainer.")
                background_data = self._prepare_background_data(X)
                explainer = shap.KernelExplainer(self.model.predict, background_data)

            shap_values = explainer.shap_values(instance)

            # Handle both single and multi-class outputs
            if not isinstance(shap_values, list):
                shap_values = [shap_values]

            # Optimized: Skip plotting for speed
            # Original: Create bar plot for local explanation

            print(f"Local SHAP analysis for instance {instance_index} completed successfully")

        except Exception as e:
            print(f"Error during local SHAP analysis for instance {instance_index}: {str(e)}")


if __name__ == "__main__":
    print("This module provides SHAP analysis functionality for machine learning models.")
