import shap
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


class SHAPAnalyzer:
    def __init__(self, model, output_manager, max_samples=200):
        """
        Initialize SHAP Analyzer

        Args:
            model: The trained model to analyze
            output_manager: Instance of OutputManager for handling output paths
            max_samples: Maximum number of samples for SHAP analysis (default: 200)
        """
        self.model = model
        self.output_manager = output_manager
        self.max_samples = max_samples

    def _sample_data(self, X, random_state=42):
        """
        Reduce dataset to manageable size for SHAP analysis

        Args:
            X: Input features
            random_state: Random seed for reproducibility

        Returns:
            Sampled dataset
        """
        if len(X) <= self.max_samples:
            return X

        if hasattr(X, 'iloc'):
            _, X_sampled = train_test_split(
                X,
                train_size=self.max_samples,
                random_state=random_state
            )
        else:
            indices = np.random.RandomState(random_state).choice(
                len(X),
                self.max_samples,
                replace=False
            )
            X_sampled = X[indices]

        return X_sampled

    def explain_global(self, X):
        """
        Create global model explanations using SHAP

        Args:
            X: Input features to explain

        Returns:
            tuple: (explainer, shap_values) if successful, (None, None) if failed
        """
        print(f"\nStarting SHAP analysis with maximum {self.max_samples} samples...")

        # Reduce dataset size
        X_sampled = self._sample_data(X)

        try:
            # Choose appropriate explainer based on model type
            if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sampled)
            else:
                background = shap.kmeans(X_sampled, 10)
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba')
                    else self.model.predict,
                    background
                )
                shap_values = explainer.shap_values(X_sampled, nsamples=100)

            # Create and save plots
            if isinstance(shap_values, list):  # Multi-class case
                for i, class_shap_values in enumerate(shap_values):
                    # Summary Bar Plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        class_shap_values,
                        X_sampled,
                        plot_type="bar",
                        max_display=10,
                        show=False
                    )
                    plt.tight_layout()

                    bar_plot_path = self.output_manager.get_path(
                        "models", "shap", f"summary_bar_plot_class_{i}.png"
                    )
                    plt.savefig(bar_plot_path, bbox_inches="tight", dpi=300)
                    plt.close()

                    # Beeswarm Plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        class_shap_values,
                        X_sampled,
                        plot_type="dot",
                        max_display=10,
                        show=False
                    )
                    plt.tight_layout()

                    beeswarm_path = self.output_manager.get_path(
                        "models", "shap", f"beeswarm_plot_class_{i}.png"
                    )
                    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
                    plt.close()
            else:  # Binary classification or regression
                # Summary Bar Plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_sampled,
                    plot_type="bar",
                    max_display=10,
                    show=False
                )
                plt.tight_layout()

                bar_plot_path = self.output_manager.get_path(
                    "models", "shap", "summary_bar_plot.png"
                )
                plt.savefig(bar_plot_path, bbox_inches="tight", dpi=300)
                plt.close()

                # Beeswarm Plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_sampled,
                    plot_type="dot",
                    max_display=10,
                    show=False
                )
                plt.tight_layout()

                beeswarm_path = self.output_manager.get_path(
                    "models", "shap", "beeswarm_plot.png"
                )
                plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
                plt.close()

            print("Global SHAP analysis completed successfully!")
            return explainer, shap_values

        except Exception as e:
            print(f"Error during global SHAP analysis: {str(e)}")
            return None, None

    def explain_local(self, X, instance_index):
        """
        Create local explanations for a single instance and save as PNG

        Args:
            X: Input features
            instance_index: Index of instance to explain
        """
        try:
            # Prepare data
            if hasattr(X, 'iloc'):
                instance = X.iloc[[instance_index]]
                feature_names = X.columns
            else:
                instance = X[instance_index:instance_index + 1]
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Choose appropriate explainer
            if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(instance)
                expected_value = explainer.expected_value
            else:
                background = shap.kmeans(self._sample_data(X), 10)
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba')
                    else self.model.predict,
                    background
                )
                shap_values = explainer.shap_values(instance)
                expected_value = explainer.expected_value

            # Create and save local explanation plots
            if isinstance(shap_values, list):  # Multi-class
                for i, class_shap_values in enumerate(shap_values):
                    plt.figure(figsize=(12, 4))

                    # Create waterfall plot for this class
                    shap_values_for_class = class_shap_values[0] if class_shap_values.ndim > 1 else class_shap_values
                    expected_val = expected_value[i] if isinstance(expected_value,
                                                                   (list, np.ndarray)) else expected_value

                    # Sort features by absolute SHAP value
                    feature_importance = np.abs(shap_values_for_class)
                    feature_order = np.argsort(feature_importance)
                    ordered_features = [feature_names[i] for i in feature_order]
                    ordered_shap = shap_values_for_class[feature_order]

                    # Create bar plot
                    plt.barh(range(len(ordered_shap)), ordered_shap)
                    plt.yticks(range(len(ordered_shap)), ordered_features)
                    plt.xlabel('SHAP value')
                    plt.title(f'Local Explanation for Instance {instance_index} (Class {i})')
                    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

                    # Save plot
                    plot_path = self.output_manager.get_path(
                        "models", "shap", f"local_explanation_class_{i}_instance_{instance_index}.png"
                    )
                    plt.tight_layout()
                    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                    plt.close()

            else:  # Binary classification or regression
                plt.figure(figsize=(12, 4))

                # Ensure correct shape
                if shap_values.ndim > 2:  # For some models that return 3D arrays
                    shap_values = shap_values[0, :, 1]  # Take class 1 probabilities
                elif shap_values.ndim == 2:
                    shap_values = shap_values[0]

                # Sort features by absolute SHAP value
                feature_importance = np.abs(shap_values)
                feature_order = np.argsort(feature_importance)
                ordered_features = [feature_names[i] for i in feature_order]
                ordered_shap = shap_values[feature_order]

                # Create bar plot
                plt.barh(range(len(ordered_shap)), ordered_shap)
                plt.yticks(range(len(ordered_shap)), ordered_features)
                plt.xlabel('SHAP value')
                plt.title(f'Local Explanation for Instance {instance_index}')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

                # Save plot
                plot_path = self.output_manager.get_path(
                    "models", "shap", f"local_explanation_instance_{instance_index}.png"
                )
                plt.tight_layout()
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                plt.close()

            print(f"Local SHAP analysis for instance {instance_index} completed successfully")

        except Exception as e:
            print(f"Error during local SHAP analysis for instance {instance_index}: {str(e)}")
            if 'shap_values' in locals():
                print(f"SHAP values shape: {np.array(shap_values).shape}")