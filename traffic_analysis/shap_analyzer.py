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
        Initialize SHAP Analyzer.

        Args:
            model: the trainer model
            output_dir: place for SHAP plots
            max_samples: max sample for shap-analyze (otherwise the code runs forever - no it is around 5 min)
        """
        self.model = model
        self.output_dir = output_dir
        self.max_samples = max_samples
        os.makedirs(self.output_dir, exist_ok=True)

    def _sample_data(self, X, random_state=42):
        """
        Reduziert den Datensatz auf eine handhabbare Größe.
        """
        if len(X) <= self.max_samples:
            return X

        if hasattr(X, 'iloc'):
            # Für pandas DataFrames
            _, X_sampled = train_test_split(
                X,
                train_size=self.max_samples,
                random_state=random_state
            )
        else:
            # for numpy arrays
            indices = np.random.RandomState(random_state).choice(
                len(X),
                self.max_samples,
                replace=False
            )
            X_sampled = X[indices]

        return X_sampled

    def explain_global(self, X):
        """
        Erstellt globale Erklärungen mit Summary Plots.
        """
        print(f"\nBeginne SHAP-Analyse mit maximal {self.max_samples} Samples...")

        # reduced data
        X_sampled = self._sample_data(X)

        try:
            # choose explainer
            if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sampled)
            else:
                # optimize kernel explainer for models that are not tree
                background = shap.kmeans(X_sampled, 10)
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba')
                    else self.model.predict,
                    background
                )
                shap_values = explainer.shap_values(X_sampled, nsamples=100)

            # make and safe plots
            if isinstance(shap_values, list):  # multi class problem
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
            else:  # binary or regression
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

            print("Globale SHAP-Analyse erfolgreich abgeschlossen!")
            return explainer, shap_values

        except Exception as e:
            print(f"Fehler während der globalen SHAP-Analyse: {str(e)}")
            return None, None

    def explain_local(self, X, instance_index):
        """
        Erstellt lokale Erklärungen für eine einzelne Instanz.
        """
        try:
            # reduce data for kernel explainer if necessary
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

            # calculate shap values only for choosen instance
            instance = X.iloc[[instance_index]] if hasattr(X, 'iloc') else X[[instance_index]]
            shap_values = explainer.shap_values(instance)

            if isinstance(shap_values, list):  # multi clas problem
                for i, class_shap_values in enumerate(shap_values):
                    force_plot_path = os.path.join(
                        self.output_dir,
                        f"force_plot_class_{i}_instance_{instance_index}.png"
                    )
                    # adjusted force_plot call
                    plt.figure(figsize=(12, 4))
                    shap.plots.force(
                        base_value=explainer.expected_value[i],
                        shap_values=class_shap_values[0],  # Erste Dimension extrahieren
                        features=instance.iloc[0] if hasattr(instance, 'iloc') else instance[0],
                        matplotlib=True,
                        show=False
                    )
                    plt.tight_layout()
                    plt.savefig(force_plot_path, bbox_inches='tight', dpi=300)
                    plt.close()
            else:  # binary or regression
                force_plot_path = os.path.join(
                    self.output_dir,
                    f"force_plot_instance_{instance_index}.png"
                )
                # adjusted force_plot call
                plt.figure(figsize=(12, 4))
                shap.plots.force(
                    base_value=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                    else explainer.expected_value[0],
                    shap_values=shap_values[0] if shap_values.ndim > 1 else shap_values,
                    # extract first dimension only if necessary
                    features=instance.iloc[0] if hasattr(instance, 'iloc') else instance[0],
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(force_plot_path, bbox_inches='tight', dpi=300)
                plt.close()

            print(f"Lokale SHAP-Analyse für Instanz {instance_index} gespeichert in: {self.output_dir}")

        except Exception as e:
            print(f"Fehler während der lokalen SHAP-Analyse für Instanz {instance_index}: {str(e)}")
            print(f"SHAP expected_value Typ: {type(explainer.expected_value)}")
            print(f"SHAP values Typ: {type(shap_values)}")
            if isinstance(shap_values, (list, np.ndarray)):
                print(f"SHAP values Shape: {np.array(shap_values).shape}")