import shap
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


class SHAPAnalyzer:
    def __init__(self, model, output_dir="shap_results", max_samples=200):
        """
        Initialize the SHAP Analyzer.

        Args:
            model: The trained model
            output_dir: Directory for saving SHAP plots
            max_samples: Maximum number of samples for SHAP analysis
        """
        self.model = model
        self.output_dir = output_dir
        self.max_samples = max_samples
        os.makedirs(self.output_dir, exist_ok=True)

    def _sample_data(self, X, random_state=42):
        """
        Reduces dataset to a manageable size.
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
        Creates global explanations with summary plots.
        """
        print(f"\nStarting SHAP analysis with maximum {self.max_samples} samples...")

        # Reduce dataset
        X_sampled = self._sample_data(X)

        try:
            # Choose appropriate explainer
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
            if isinstance(shap_values, list):  # Multi-class problem
                for i, class_shap_values in enumerate(shap_values):
                    # Summary Bar Plot
                    summary_bar_path = os.path.join(self.output_dir, f"summary_bar_plot_class_{i}.png")
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        class_shap_values,
                        X_sampled,
                        plot_type="bar",
                        max_display=10,
                        show=False
                    )
                    plt.tight_layout()
                    plt.savefig(summary_bar_path, bbox_inches="tight", dpi=300)
                    plt.close()

                    # Beeswarm Plot
                    beeswarm_path = os.path.join(self.output_dir, f"beeswarm_plot_class_{i}.png")
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        class_shap_values,
                        X_sampled,
                        plot_type="dot",
                        max_display=10,
                        show=False
                    )
                    plt.tight_layout()
                    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
                    plt.close()
            else:  # Binary or regression
                # Summary Bar Plot
                summary_bar_path = os.path.join(self.output_dir, "summary_bar_plot.png")
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_sampled,
                    plot_type="bar",
                    max_display=10,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(summary_bar_path, bbox_inches="tight", dpi=300)
                plt.close()

                # Beeswarm Plot
                beeswarm_path = os.path.join(self.output_dir, "beeswarm_plot.png")
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_sampled,
                    plot_type="dot",
                    max_display=10,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
                plt.close()

            print("Global SHAP analysis completed successfully!")
            return explainer, shap_values

        except Exception as e:
            print(f"Error during global SHAP analysis: {str(e)}")
            return None, None

    def explain_local(self, X, instance_index):
        """
        Creates local explanations for a single instance.
        """
        try:
            # Background data for KernelExplainer if needed
            X_background = self._sample_data(X)

            if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
                explainer = shap.TreeExplainer(self.model)
            else:
                background = shap.kmeans(X_background, 10)
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba')
                    else self.model.predict,
                    background
                )

            # Select instance and ensure proper dimensionality
            instance = X.iloc[[instance_index]] if hasattr(X, 'iloc') else X[instance_index:instance_index + 1]
            shap_values = explainer.shap_values(instance)

            # Handle different types of SHAP values
            if isinstance(shap_values, list):  # Multi-class
                for class_index, class_shap_values in enumerate(shap_values):
                    force_plot_path = os.path.join(
                        self.output_dir,
                        f"force_plot_class_{class_index}_instance_{instance_index}.html"
                    )

                    base_value = (explainer.expected_value[class_index]
                                  if isinstance(explainer.expected_value, (list, np.ndarray))
                                  else explainer.expected_value)

                    force_plot = shap.force_plot(
                        base_value=base_value,
                        shap_values=class_shap_values[0] if class_shap_values.ndim > 1 else class_shap_values,
                        features=instance.iloc[0] if hasattr(instance, 'iloc') else instance[0],
                        show=False
                    )
                    shap.save_html(force_plot_path, force_plot)

            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:  # Shape (1, features, classes)
                n_features = shap_values.shape[1]
                n_classes = shap_values.shape[2]

                for class_index in range(n_classes):
                    force_plot_path = os.path.join(
                        self.output_dir,
                        f"force_plot_class_{class_index}_instance_{instance_index}.html"
                    )

                    # Extract values for this class
                    class_values = shap_values[0, :, class_index]
                    base_value = explainer.expected_value[class_index] if isinstance(explainer.expected_value,
                                                                                     np.ndarray) else explainer.expected_value

                    force_plot = shap.force_plot(
                        base_value=base_value,
                        shap_values=class_values,
                        features=instance.iloc[0] if hasattr(instance, 'iloc') else instance[0],
                        show=False
                    )
                    shap.save_html(force_plot_path, force_plot)

            else:  # Binary classification or regression
                force_plot_path = os.path.join(
                    self.output_dir,
                    f"force_plot_instance_{instance_index}.html"
                )

                base_value = (explainer.expected_value[0]
                              if isinstance(explainer.expected_value, (list, np.ndarray))
                              else explainer.expected_value)

                if isinstance(shap_values, np.ndarray):
                    if shap_values.ndim > 2:
                        shap_values = shap_values[0]
                    elif shap_values.ndim == 2:
                        shap_values = shap_values[0]

                force_plot = shap.force_plot(
                    base_value=base_value,
                    shap_values=shap_values,
                    features=instance.iloc[0] if hasattr(instance, 'iloc') else instance[0],
                    show=False
                )
                shap.save_html(force_plot_path, force_plot)

            print(f"Local SHAP analysis for instance {instance_index} saved to: {self.output_dir}")

        except Exception as e:
            print(f"Error during local SHAP analysis for instance {instance_index}: {str(e)}")
            print(f"SHAP expected_value type: {type(explainer.expected_value)}")
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, (list, np.ndarray)):
                print(f"SHAP values shape: {np.array(shap_values).shape}")