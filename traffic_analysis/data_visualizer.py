# data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path


class DataVisualizer:
    """Klasse zur Visualisierung von Modellmetriken und Ergebnissen"""

    def __init__(self, output_manager):
        """
        Initialisiert den DataVisualizer

        Args:
            output_manager: Instance des OutputManagers für Dateipfade
        """
        self.output_manager = output_manager
        self.model_results: List[Dict[str, Any]] = []

        # Definiere Standard-Plot-Stil
        self.plot_style = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3
        }

        # Farbpalette für Plots
        self.colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6',
                       '#1abc9c', '#e67e22', '#34495e']

        # Konfiguration
        self._setup_visualization()
        logging.info("DataVisualizer initialized successfully")

    def _setup_visualization(self):
        """Konfiguriert die Visualisierungseinstellungen"""
        # Setze matplotlib Parameter
        for key, value in self.plot_style.items():
            plt.rcParams[key] = value

        # Konfiguriere Standard-Stil ohne seaborn
        plt.style.use('default')

        # Konfiguriere seaborn grundlegend
        sns.set_palette(self.colors)

    def add_model_result(self, model_name: str, metrics: Dict[str, Any]):
        """
        Fügt neue Modellergebnisse hinzu und aktualisiert Visualisierungen

        Args:
            model_name: Name des Modells
            metrics: Dictionary mit Modellmetriken
        """
        try:
            result = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['weighted avg']['f1-score'],
                'Precision': metrics['weighted avg']['precision'],
                'Recall': metrics['weighted avg']['recall'],
                'ROC AUC': metrics.get('roc_auc', None)
            }
            self.model_results.append(result)

            # Update visualizations
            self._create_performance_visualizations()
            self._save_model_summary()
            logging.info(f"Added results for model: {model_name}")

        except Exception as e:
            logging.error(f"Error adding model result for {model_name}: {str(e)}")

    def plot_roc_curve(self, model, X_test, y_test, model_name: str):
        """
        Erstellt ROC-Kurve für ein Modell

        Args:
            model: Trainiertes Modell
            X_test: Test-Features
            y_test: Test-Labels
            model_name: Name des Modells
        """
        try:
            plt.figure(figsize=self.plot_style['figure.figsize'])

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr,
                         label=f'ROC curve (AUC = {roc_auc:.2f})',
                         linewidth=2)
                plt.plot([0, 1], [0, 1], 'k--',
                         label='Random prediction',
                         alpha=0.8,
                         linewidth=1)

                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)

                self._save_plot('roc_curve', model_name)
            else:
                logging.warning(f"Model {model_name} does not support probability predictions")

        except Exception as e:
            logging.error(f"Error plotting ROC curve for {model_name}: {str(e)}")

    def plot_precision_recall_curve(self, model, X_test, y_test, model_name: str):
        """
        Erstellt Precision-Recall-Kurve für ein Modell

        Args:
            model: Trainiertes Modell
            X_test: Test-Features
            y_test: Test-Labels
            model_name: Name des Modells
        """
        try:
            if hasattr(model, 'predict_proba'):
                plt.figure(figsize=self.plot_style['figure.figsize'])

                y_prob = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall, precision)

                plt.plot(recall, precision,
                         label=f'PR curve (AUC = {pr_auc:.2f})',
                         linewidth=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {model_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)

                self._save_plot('precision_recall_curve', model_name)
            else:
                logging.warning(f"Model {model_name} does not support probability predictions")

        except Exception as e:
            logging.error(f"Error plotting PR curve for {model_name}: {str(e)}")

    def plot_feature_importance(self, model, feature_names, model_name: str):
        """
        Visualisiert Feature Importance
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                logging.warning(f"Model {model_name} does not support feature importance")
                return

            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
            plt.barh(range(len(importance)), importance['importance'],
                     align='center', color=self.colors[0])
            plt.yticks(range(len(importance)), importance['feature'])
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()

            self._save_plot('feature_importance', model_name)

        except Exception as e:
            logging.error(f"Error plotting feature importance for {model_name}: {str(e)}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str, labels: List[str] = None):
        """
        Erstellt Confusion Matrix Visualisierung
        """
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)

            # Erstelle Heatmap manuell mit plt
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()

            # Füge Zahlen in die Zellen ein
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")

            if labels:
                tick_marks = np.arange(len(labels))
                plt.xticks(tick_marks, labels)
                plt.yticks(tick_marks, labels)

            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            self._save_plot('confusion_matrix', model_name)

        except Exception as e:
            logging.error(f"Error plotting confusion matrix for {model_name}: {str(e)}")

    def _create_performance_visualizations(self):
        """Erstellt verschiedene Performance-Visualisierungen"""
        try:
            if not self.model_results:
                return

            df = pd.DataFrame(self.model_results)
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

            self._create_bar_plot(df, metrics)
            self._create_heatmap(df, metrics)
            self._create_radar_plot(df, metrics)

        except Exception as e:
            logging.error(f"Error creating performance visualizations: {str(e)}")

    def _create_bar_plot(self, df: pd.DataFrame, metrics: List[str]):
        """Erstellt Balkendiagramm für Modellvergleich"""
        try:
            plt.figure(figsize=self.plot_style['figure.figsize'])
            x = np.arange(len(df))
            width = 0.8 / len(metrics)

            for i, metric in enumerate(metrics):
                plt.bar(x + i * width, df[metric],
                        width,
                        label=metric,
                        color=self.colors[i % len(self.colors)])

            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * (len(metrics) - 1) / 2, df['Model'],
                       rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            self._save_plot('model_comparison', 'bar_plot')

        except Exception as e:
            logging.error(f"Error creating bar plot: {str(e)}")

    def _create_heatmap(self, df: pd.DataFrame, metrics: List[str]):
        """Erstellt Heatmap für Modellmetriken"""
        try:
            plt.figure(figsize=(10, len(df) * 0.5 + 2))

            # Erstelle Heatmap manuell mit plt
            data = df[metrics].values
            plt.imshow(data, aspect='auto', cmap='YlOrRd')
            plt.colorbar()

            # Beschriftungen
            plt.xticks(range(len(metrics)), metrics)
            plt.yticks(range(len(df)), df['Model'])

            # Füge Werte in die Zellen ein
            for i in range(len(df)):
                for j in range(len(metrics)):
                    plt.text(j, i, f'{data[i, j]:.3f}',
                             ha='center', va='center')

            plt.title('Performance Metrics Heatmap')
            plt.tight_layout()

            self._save_plot('metrics_heatmap', 'all_models')

        except Exception as e:
            logging.error(f"Error creating heatmap: {str(e)}")

    def _create_radar_plot(self, df: pd.DataFrame, metrics: List[str]):
        """Erstellt Radar-Plot für Modellvergleich"""
        try:
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            for idx, row in df.iterrows():
                values = [row[metric] for metric in metrics]
                values = np.concatenate((values, [values[0]]))

                ax.plot(angles, values, 'o-',
                        linewidth=2,
                        label=row['Model'],
                        color=self.colors[idx % len(self.colors)])
                ax.fill(angles, values,
                        alpha=0.25,
                        color=self.colors[idx % len(self.colors)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title('Model Performance Comparison (Radar Plot)')
            plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
            plt.tight_layout()

            self._save_plot('radar_plot', 'all_models')

        except Exception as e:
            logging.error(f"Error creating radar plot: {str(e)}")

    def _save_plot(self, plot_type: str, model_name: str):
        """Speichert Plot in entsprechendes Verzeichnis"""
        try:
            path = self.output_manager.get_path(
                "reports", "visualizations", f"{model_name}_{plot_type}.png"
            )
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Error saving plot {plot_type} for {model_name}: {str(e)}")

    def _save_model_summary(self):
        """Speichert Zusammenfassung der Modellergebnisse"""
        try:
            if not self.model_results:
                return

            df = pd.DataFrame(self.model_results)

            # Save CSV
            df.to_csv(self.output_manager.get_path(
                "reports", "summaries", "model_comparison.csv"
            ), index=False)

            # Save detailed text summary
            with open(self.output_manager.get_path(
                    "reports", "summaries", "model_comparison.txt"
            ), 'w') as f:
                f.write("=== Model Performance Summary ===\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")

                for _, row in df.iterrows():
                    f.write(f"Model: {row['Model']}\n")
                    f.write("-" * len(f"Model: {row['Model']}") + "\n")
                    for metric in ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC']:
                        if pd.notnull(row[metric]):
                            f.write(f"  {metric:.<20} {row[metric]:.4f}\n")
                    f.write("\n")

                # Add summary statistics
                f.write("\nSummary Statistics:\n")
                f.write("-" * 20 + "\n")
                metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
                for metric in metrics:
                    values = df[metric].dropna()
                    if not values.empty:
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean: {values.mean():.4f}\n")
                        f.write(f"  Std:  {values.std():.4f}\n")
                        f.write(f"  Max:  {values.max():.4f} ({df.loc[values.idxmax(), 'Model']})\n")
                        f.write(f"  Min:  {values.min():.4f} ({df.loc[values.idxmin(), 'Model']})\n")

        except Exception as e:
            logging.error(f"Error saving model summary: {str(e)}")