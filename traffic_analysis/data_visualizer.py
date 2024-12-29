# file: data_visualizer.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix
)


class DataVisualizer:
    """
    Provides various plotting functionalities for model metrics and performance visualization.
    """

    def __init__(self, output_manager):
        """
        Initializes the DataVisualizer with a given OutputManager for path handling.
        """
        self.output_manager = output_manager
        self.model_results: List[Dict[str, Any]] = []

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
        self.colors = [
            '#2ecc71', '#3498db', '#e74c3c', '#f1c40f',
            '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
        ]
        self._setup_visualization()
        logging.info("DataVisualizer initialized successfully.")

    def _setup_visualization(self):
        """
        Applies chosen plot style (rcParams) and color palette for Seaborn/Matplotlib.
        """
        for key, value in self.plot_style.items():
            plt.rcParams[key] = value
        plt.style.use('default')
        sns.set_palette(self.colors)

    def add_model_result(self, model_name: str, metrics: Dict[str, Any]):
        """
        Stores and logs model metrics (Accuracy, F1, Precision, Recall, ROC AUC).
        Generates overview plots for performance comparison among multiple models.
        """
        try:
            if 'accuracy' not in metrics or 'weighted avg' not in metrics:
                logging.warning(f"Metrics for {model_name} seem incomplete.")
                return

            result = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', np.nan),
                'F1-Score': metrics['weighted avg'].get('f1-score', np.nan),
                'Precision': metrics['weighted avg'].get('precision', np.nan),
                'Recall': metrics['weighted avg'].get('recall', np.nan),
                'ROC AUC': metrics.get('roc_auc', None)
            }
            self.model_results.append(result)

            self._create_performance_visualizations()
            self._save_model_summary()
            logging.info(f"Added results for model: {model_name}")

        except Exception as e:
            logging.error(f"Error adding model result for {model_name}: {e}")

    def plot_roc_curve(self, model, X_test, y_test, model_name: str):
        """
        Plots and saves the ROC curve for a given model, if predict_proba is available.
        """
        try:
            if not hasattr(model, 'predict_proba'):
                logging.warning(f"Model {model_name} does not support predict_proba.")
                return

            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=self.plot_style['figure.figsize'])
            plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")

            self._save_plot('roc_curve', model_name)
        except Exception as e:
            logging.error(f"Error plotting ROC curve for {model_name}: {e}")

    def plot_precision_recall_curve(self, model, X_test, y_test, model_name: str):
        """
        Plots and saves the Precision-Recall curve for a given model, if predict_proba is available.
        """
        try:
            if not hasattr(model, 'predict_proba'):
                logging.warning(f"Model {model_name} does not support predict_proba.")
                return

            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
            pr_auc = auc(recall, precision)

            plt.figure(figsize=self.plot_style['figure.figsize'])
            plt.plot(recall, precision, label=f'PR (AUC={pr_auc:.2f})', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()

            self._save_plot('precision_recall_curve', model_name)
        except Exception as e:
            logging.error(f"Error plotting Precision-Recall curve for {model_name}: {e}")

    def plot_feature_importance(self, model, feature_names, model_name: str):
        """
        Plots and saves feature importances for tree-based models (with feature_importances_).
        Updated: We rename output file to 'tree_feature_importance.png' to avoid confusion
        with any SHAP-based importance plots.
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                logging.warning(f"Model {model_name} has no feature_importances_.")
                return

            # Optional: Sort features by importance
            importance_vals = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_vals
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
            plt.barh(importance_df['feature'], importance_df['importance'], color=self.colors[0])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance (Tree-based) - {model_name}')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()

            # Save as 'tree_feature_importance.png'
            self._save_plot('tree_feature_importance', model_name)
        except Exception as e:
            logging.error(f"Error plotting feature importance for {model_name}: {e}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str, labels: List[str] = None):
        """
        Plots and saves a confusion matrix for predicted vs. true labels.
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            if labels:
                plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
                plt.yticks(np.arange(len(labels)) + 0.5, labels)
            plt.tight_layout()

            self._save_plot('confusion_matrix', model_name)
        except Exception as e:
            logging.error(f"Error plotting confusion matrix for {model_name}: {e}")

    def plot_model_comparison(self, metrics_list: List[tuple]):
        """
        Creates a simple bar chart comparing model accuracies (or other metrics).
        Expects a list of tuples like [(model_name, accuracy), ...].
        """
        if not metrics_list:
            logging.warning("No metrics provided for plot_model_comparison.")
            return

        model_names = [t[0] for t in metrics_list]
        accuracies = [t[1] for t in metrics_list]

        plt.figure(figsize=self.plot_style['figure.figsize'])
        bars = plt.bar(model_names, accuracies, color='skyblue')
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title("Model Comparison by Accuracy")

        # Annotate each bar with the accuracy value
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2.0,
                     bar.get_height(),
                     f"{acc:.3f}",
                     ha='center', va='bottom')

        plt.ylim([0, 1])  # if accuracy is a 0-1 scale
        plt.tight_layout()

        output_path = self.output_manager.get_path("reports", "visualizations", "model_comparison_accuracy.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Model comparison chart saved as {output_path}")

    def plot_pcap_size_distribution(self, pcap_sizes: Dict[str, List[int]]):
        """
        Plots a boxplot (or barplot) showing the distribution of pcap file sizes for each folder.
        pcap_sizes: { "proxy": [size_1, size_2, ...], "normal": [size_1, size_2, ...], ... }
        """
        try:
            # Convert dict to a DataFrame for easy plotting with Seaborn
            plot_data = []
            for folder_label, sizes in pcap_sizes.items():
                for s in sizes:
                    plot_data.append({"Folder": folder_label, "SizeBytes": s})

            df = pd.DataFrame(plot_data)
            if df.empty:
                logging.warning("No PCAP size data to plot.")
                return

            plt.figure(figsize=self.plot_style['figure.figsize'])
            sns.boxplot(x="Folder", y="SizeBytes", data=df)
            plt.title("PCAP Size Distribution per Folder")
            plt.ylabel("PCAP File Size (bytes)")

            self._save_plot('pcap_size_distribution', 'all_folders')
        except Exception as e:
            logging.error(f"Error plotting PCAP size distribution: {e}")

    def _create_performance_visualizations(self):
        """
        Called internally whenever a new model result is added, to generate
        comparative plots like bar-plot, heatmap, or radar-plot for multiple models.
        """
        try:
            if not self.model_results:
                return
            df = pd.DataFrame(self.model_results)
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

            if len(df) < 2:
                # Only one model -> skip multi-model comparison
                return

            self._create_bar_plot(df, metrics)
            self._create_heatmap(df, metrics)
            self._create_radar_plot(df, metrics)
        except Exception as e:
            logging.error(f"Error creating performance visualizations: {e}")

    def _create_bar_plot(self, df: pd.DataFrame, metrics: List[str]):
        """
        Creates a grouped bar plot for multiple metrics across all models.
        """
        try:
            plt.figure(figsize=self.plot_style['figure.figsize'])
            x = np.arange(len(df))
            width = 0.8 / len(metrics)

            for i, metric in enumerate(metrics):
                plt.bar(x + i * width,
                        df[metric],
                        width,
                        label=metric,
                        color=self.colors[i % len(self.colors)])

            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * (len(metrics) - 1) / 2, df['Model'], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            self._save_plot('model_comparison_bar', 'all_models')
        except Exception as e:
            logging.error(f"Error creating bar plot: {e}")

    def _create_heatmap(self, df: pd.DataFrame, metrics: List[str]):
        """
        Creates a heatmap of the main metrics for each model.
        """
        try:
            plt.figure(figsize=(8, max(4, len(df) * 0.5 + 2)))
            data = df[metrics].values
            sns.heatmap(
                data,
                annot=True,
                cmap='YlOrRd',
                xticklabels=metrics,
                yticklabels=df['Model']
            )
            plt.title('Performance Metrics Heatmap')
            plt.tight_layout()

            self._save_plot('metrics_heatmap', 'all_models')
        except Exception as e:
            logging.error(f"Error creating heatmap: {e}")

    def _create_radar_plot(self, df: pd.DataFrame, metrics: List[str]):
        """
        Creates a radar/spider plot for selected metrics across models.
        """
        try:
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            for idx, row in df.iterrows():
                values = [row[m] for m in metrics]
                values = np.concatenate((values, [values[0]]))
                ax.plot(angles, values, 'o-', linewidth=2,
                        label=row['Model'], color=self.colors[idx % len(self.colors)])
                ax.fill(angles, values, alpha=0.25,
                        color=self.colors[idx % len(self.colors)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title('Model Performance (Radar Plot)')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()

            self._save_plot('radar_plot', 'all_models')
        except Exception as e:
            logging.error(f"Error creating radar plot: {e}")

    def _save_plot(self, plot_type: str, model_name: str):
        """
        Saves the current figure with a standardized naming scheme.
        """
        try:
            path = self.output_manager.get_path(
                "reports", "visualizations", f"{model_name}_{plot_type}.png"
            )
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Plot saved: {path}")
        except Exception as e:
            logging.error(f"Error saving plot {plot_type} for {model_name}: {e}")

    def _save_model_summary(self):
        """
        Exports the current model_results to CSV and TXT summaries.
        """
        try:
            if not self.model_results:
                return

            df = pd.DataFrame(self.model_results)

            # CSV
            csv_path = self.output_manager.get_path("reports", "summaries", "model_comparison.csv")
            df.to_csv(csv_path, index=False)

            # TXT
            txt_path = self.output_manager.get_path("reports", "summaries", "model_comparison.txt")
            with open(txt_path, 'w') as f:
                f.write("=== Model Performance Summary ===\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")

                for _, row in df.iterrows():
                    model_name = row['Model']
                    f.write(f"Model: {model_name}\n")
                    f.write("-" * (len(model_name) + 7) + "\n")
                    for metric in ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC']:
                        val = row.get(metric, None)
                        if val is not None and not pd.isnull(val):
                            f.write(f"{metric:<15}: {val:.4f}\n")
                    f.write("\n")

                # summary stats
                stats_metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
                f.write("\nSummary Statistics:\n")
                f.write("-" * 20 + "\n")
                for metric in stats_metrics:
                    series = df[metric].dropna()
                    if not series.empty:
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean: {series.mean():.4f}\n")
                        f.write(f"  Std:  {series.std():.4f}\n")
                        f.write(f"  Max:  {series.max():.4f} ({df.loc[series.idxmax(), 'Model']})\n")
                        f.write(f"  Min:  {series.min():.4f} ({df.loc[series.idxmin(), 'Model']})\n")

            logging.info("Model summary saved successfully.")
        except Exception as e:
            logging.error(f"Error saving model summary: {e}")
