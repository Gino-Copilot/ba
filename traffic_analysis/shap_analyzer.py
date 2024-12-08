import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings


class SHAPAnalyzer:
    def __init__(self, model, output_manager, max_display=20, max_samples=500):
        """
        Initialisiert den SHAP Analyzer

        Args:
            model: Trainiertes Modell für die Analyse
            output_manager: Instance des OutputManagers
            max_display: Maximale Anzahl der anzuzeigenden Features in Plots
            max_samples: Maximale Anzahl der Samples für SHAP-Analyse
        """
        self.model = model
        self.output_manager = output_manager
        self.max_display = max_display
        self.max_samples = max_samples
        self.model_name = self.model.__class__.__name__
        self.explainer = None
        self.shap_values = None

        # Konfiguration
        warnings.filterwarnings('ignore')
        plt.style.use('seaborn-v0_8-darkgrid')  # Änderung vorgenommen
        print(f"Initialized SHAP Analyzer for model: {self.model_name}")

    def explain_global(self, X, y=None):
        """
        Erstellt globale Modellerklärungen mit SHAP

        Args:
            X: Feature-Matrix
            y: Labels (optional)

        Returns:
            tuple: (explainer, shap_values) oder (None, None) bei Fehler
        """
        try:
            print(f"\nStarting SHAP analysis for {self.model_name}...")

            # Validierung
            if X is None or len(X) == 0:
                raise ValueError("Empty or invalid input data")

            # Reduziere Datensatz auf handhabbare Größe
            X_sampled = self._sample_data(X)

            # Initialisiere explainer
            self.explainer = self._initialize_explainer(X_sampled)
            if self.explainer is None:
                return None, None

            # Berechne SHAP-Werte
            self.shap_values = self._calculate_shap_values(X_sampled)
            if self.shap_values is None:
                return None, None

            # Erstelle Visualisierungen
            self._create_visualizations(X_sampled)

            # Speichere Analysen
            self._save_analysis_results(X_sampled)

            print("Global SHAP analysis completed successfully!")
            return self.explainer, self.shap_values

        except Exception as e:
            print(f"Error in global SHAP analysis: {str(e)}")
            return None, None

    def explain_local(self, X, instance_indices):
        """
        Erstellt lokale Erklärungen für spezifische Instanzen

        Args:
            X: Feature-Matrix
            instance_indices: Liste von Indizes für zu erklärende Instanzen
        """
        try:
            print(f"\nCreating local explanations for {len(instance_indices)} instances...")

            if self.explainer is None:
                self.explainer = self._initialize_explainer(X)

            for idx in instance_indices:
                if isinstance(X, pd.DataFrame):
                    instance = X.iloc[[idx]]
                    feature_names = X.columns
                else:
                    instance = X[idx:idx + 1]
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

                local_shap = self.explainer.shap_values(instance)
                self._plot_local_explanation(local_shap, instance, feature_names, idx)

            print("Local explanations completed successfully!")

        except Exception as e:
            print(f"Error in local SHAP analysis: {str(e)}")

    def _is_tree_based_model(self):
        """Prüft, ob das Modell Tree-basiert ist"""
        return isinstance(self.model, (
            RandomForestClassifier,
            GradientBoostingClassifier,
            XGBClassifier
        ))

    def _sample_data(self, X):
        """Reduziert Datensatz auf handhabbare Größe"""
        try:
            if len(X) <= self.max_samples:
                return X

            if isinstance(X, pd.DataFrame):
                return X.sample(n=self.max_samples, random_state=42)
            else:
                indices = np.random.RandomState(42).choice(
                    len(X), self.max_samples, replace=False
                )
                return X[indices]

        except Exception as e:
            print(f"Error in data sampling: {str(e)}")
            return X

    def _initialize_explainer(self, X):
        """Initialisiert den passenden SHAP Explainer"""
        try:
            if self._is_tree_based_model():
                return shap.TreeExplainer(self.model)
            else:
                background = shap.sample(X, 100)
                return shap.KernelExplainer(
                    self.model.predict_proba,
                    background
                )
        except Exception as e:
            print(f"Error initializing explainer: {str(e)}")
            return None

    def _calculate_shap_values(self, X):
        """Berechnet SHAP-Werte"""
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                # Für binäre Klassifikation nehmen wir die positive Klasse
                shap_values = shap_values[1]
            return shap_values

        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return None

    def _create_visualizations(self, X):
        """Erstellt alle SHAP-Visualisierungen"""
        try:
            self._plot_summary(X)
            self._plot_beeswarm(X)
            self._plot_feature_importance()

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_summary(self, X):
        """Erstellt Summary Plot"""
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                max_display=self.max_display,
                show=False
            )
            plt.title(f'{self.model_name} - SHAP Summary Plot')
            plot_path = self.output_manager.get_path(
                "models", "shap", "summary_plot.png"
            )
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error creating summary plot: {str(e)}")

    def _plot_beeswarm(self, X):
        """Erstellt Beeswarm Plot"""
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                plot_type="bar",
                max_display=self.max_display,
                show=False
            )
            plt.title(f'{self.model_name} - SHAP Feature Importance')
            plot_path = self.output_manager.get_path(
                "models", "shap", "beeswarm_plot.png"
            )
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error creating beeswarm plot: {str(e)}")

    def _plot_feature_importance(self):
        """Erstellt Feature Importance Plot"""
        try:
            importance_df = pd.DataFrame({
                'feature': range(self.shap_values.shape[1]),
                'importance': np.abs(self.shap_values).mean(0)
            })
            importance_df = importance_df.sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importance_df)), importance_df['importance'])
            plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45)
            plt.title(f'{self.model_name} - SHAP Feature Importance')
            plt.tight_layout()

            plot_path = self.output_manager.get_path(
                "models", "shap", "feature_importance.png"
            )
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")

    def _plot_local_explanation(self, shap_values, instance, feature_names, idx):
        """Erstellt lokale Erklärungsplots"""
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
            plt.title(f'Local Explanation for Instance {idx}')

            plot_path = self.output_manager.get_path(
                "models", "shap", f"local_explanation_{idx}.png"
            )
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error creating local explanation plot: {str(e)}")

    def _save_analysis_results(self, X):
        """Speichert SHAP-Analyseergebnisse"""
        try:
            # Speichere SHAP-Werte
            if isinstance(X, pd.DataFrame):
                shap_df = pd.DataFrame(
                    self.shap_values,
                    columns=X.columns
                )
            else:
                shap_df = pd.DataFrame(self.shap_values)

            shap_df.to_csv(self.output_manager.get_path(
                "models", "shap", "shap_values.csv"
            ))

            # Speichere Feature Importance
            importance_df = pd.DataFrame({
                'feature': range(self.shap_values.shape[1]),
                'importance': np.abs(self.shap_values).mean(0)
            })
            importance_df = importance_df.sort_values('importance', ascending=False)

            importance_df.to_csv(self.output_manager.get_path(
                "models", "shap", "feature_importance.csv"
            ))

        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")
