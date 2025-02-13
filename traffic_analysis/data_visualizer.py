# file: data_visualizer.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    Provides plotting methods for model metrics, feature importances, etc.
    """

    def __init__(self, output_manager):
        """
        output_manager: manages file paths for saving plots.
        """
        self.output_manager = output_manager
        self.model_results: List[Dict[str, Any]] = []

        # Basic style config
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
        logging.info("DataVisualizer initialized.")

    def _setup_visualization(self):
        """
        Applies chosen rcParams and Seaborn palette.
        """
        for key, value in self.plot_style.items():
            plt.rcParams[key] = value
        plt.style.use('default')
        sns.set_palette(self.colors)

    def add_model_result(self, model_name: str, metrics: Dict[str, Any]):
        """
        Store metrics (Accuracy, F1, etc.) in self.model_results
        and potentially create comparison plots.
        """
        try:
            if 'accuracy' not in metrics:
                logging.warning(f"Metrics for {model_name} have no 'accuracy'.")
                return

            wavg = metrics.get('weighted avg', {})
            result = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', np.nan),
                'F1-Score': wavg.get('f1-score', np.nan),
                'Precision': wavg.get('precision', np.nan),
                'Recall': wavg.get('recall', np.nan),
            }

            self.model_results.append(result)
            self._create_performance_visualizations()
            logging.info(f"Added model result for {model_name}.")
        except Exception as e:
            logging.error(f"Error in add_model_result for {model_name}: {e}")

    def _create_performance_visualizations(self):
        """
        Generates bar plots, heatmaps, etc. for all models in self.model_results.
        """
        try:
            if not self.model_results:
                return

            df = pd.DataFrame(self.model_results)
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

            if len(df) < 2:
                # Only one model => skip multi-model comparisons
                return

            # Possibly bar plot for each metric
            self._create_bar_plot(df, metrics)
            # Possibly a heatmap
            self._create_heatmap(df, metrics)

        except Exception as e:
            logging.error(f"Error creating performance visualizations: {e}")

    def _create_bar_plot(self, df: pd.DataFrame, metrics: List[str]):
        """
        Simple grouped bar plot for multiple metrics.
        """
        try:
            plt.figure(figsize=self.plot_style['figure.figsize'])
            x = np.arange(len(df))
            width = 0.8 / len(metrics)

            for i, metric in enumerate(metrics):
                plt.bar(
                    x + i * width,
                    df[metric],
                    width,
                    label=metric,
                    color=self.colors[i % len(self.colors)]
                )

            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Comparison')
            plt.xticks(x + width * (len(metrics)-1)/2, df['Model'], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            out_path = self.output_manager.get_path("reports", "visualizations", "model_comparison_bar.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Saved bar plot: {out_path}")

        except Exception as e:
            logging.error(f"Error creating bar plot: {e}")

    def _create_heatmap(self, df: pd.DataFrame, metrics: List[str]):
        """
        Heatmap of metrics for all models.
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

            out_path = self.output_manager.get_path("reports", "visualizations", "metrics_heatmap.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Saved heatmap: {out_path}")

        except Exception as e:
            logging.error(f"Error creating heatmap: {e}")

    def plot_roc_curve(self, model, X_test, y_test, model_name: str):
        """
        Plots and saves ROC curve if predict_proba is available.
        """
        try:
            if not hasattr(model, 'predict_proba'):
                logging.warning(f"Model {model_name} has no predict_proba, skipping ROC.")
                return

            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=self.plot_style['figure.figsize'])
            plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
            plt.plot([0,1], [0,1], 'k--', alpha=0.7)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()

            out_path = self.output_manager.get_path("reports", "visualizations", f"{model_name}_roc_curve.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"ROC curve saved: {out_path}")

        except Exception as e:
            logging.error(f"Error plotting ROC curve for {model_name}: {e}")

    def plot_precision_recall_curve(self, model, X_test, y_test, model_name: str):
        """
        Plots Precision-Recall curve if predict_proba is available.
        """
        try:
            if not hasattr(model, 'predict_proba'):
                logging.warning(f"Model {model_name} has no predict_proba, skipping PR curve.")
                return

            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
            pr_auc = auc(recall, precision)

            plt.figure(figsize=self.plot_style['figure.figsize'])
            plt.plot(recall, precision, label=f'PR AUC={pr_auc:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()

            out_path = self.output_manager.get_path("reports", "visualizations", f"{model_name}_pr_curve.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"PR curve saved: {out_path}")

        except Exception as e:
            logging.error(f"Error plotting PR curve for {model_name}: {e}")

    def plot_confusion_matrix(self, model_name: str, y_true, y_pred, labels: Optional[List[str]] = None):
        """
        Creates a confusion matrix plot for predicted vs. actual.
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if labels and len(labels) == cm.shape[0]:
                plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha='right')
                plt.yticks(np.arange(len(labels)) + 0.5, labels)

            plt.tight_layout()
            out_path = self.output_manager.get_path("reports", "visualizations", f"{model_name}_confusion_matrix.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Confusion matrix saved: {out_path}")

        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")

    def plot_model_comparison(self, metrics_list: List[tuple]):
        """
        Creates a bar chart comparing model accuracies (or other metrics).
        metrics_list: e.g. [("RF", 0.92), ("SVC", 0.88), ...]
        """
        if not metrics_list:
            logging.warning("No metrics for plot_model_comparison.")
            return

        try:
            model_names = [m[0] for m in metrics_list]
            accuracies = [m[1] for m in metrics_list]

            plt.figure(figsize=self.plot_style['figure.figsize'])
            bars = plt.bar(model_names, accuracies, color='skyblue')
            plt.xlabel("Models")
            plt.ylabel("Accuracy")
            plt.title("Model Comparison by Accuracy")

            for bar, acc in zip(bars, accuracies):
                plt.text(
                    bar.get_x() + bar.get_width()/2.0,
                    bar.get_height(),
                    f"{acc:.3f}",
                    ha='center', va='bottom'
                )

            plt.ylim([0,1])
            plt.tight_layout()

            out_path = self.output_manager.get_path("reports", "visualizations", "model_comparison_accuracy.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Model comparison chart saved: {out_path}")

        except Exception as e:
            logging.error(f"Error in plot_model_comparison: {e}")

    def plot_pcap_size_distribution(self, pcap_sizes: Dict[str, List[int]]):
        """
        Boxplot of PCAP file sizes per category (e.g. 'proxy', 'normal').
        """
        try:
            plot_data = []
            for label, sizes in pcap_sizes.items():
                for s in sizes:
                    plot_data.append({"Category": label, "SizeBytes": s})

            if not plot_data:
                logging.warning("No PCAP sizes to plot.")
                return

            df = pd.DataFrame(plot_data)
            plt.figure(figsize=self.plot_style['figure.figsize'])
            sns.boxplot(x="Category", y="SizeBytes", data=df)
            plt.title("PCAP Size Distribution")
            plt.ylabel("File Size (bytes)")
            plt.tight_layout()

            out_path = self.output_manager.get_path("reports", "visualizations", "pcap_size_distribution.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"PCAP size distribution saved: {out_path}")
        except Exception as e:
            logging.error(f"Error plotting PCAP size distribution: {e}")

    def safe_plot_feature_importance(self, pipeline, feature_names: List[str], model_name: str):
        """
        Safely plots feature importances for tree-based models, avoiding
        'only integer scalar arrays can be converted to a scalar index' errors.

        Steps:
          1) Check if final_model has feature_importances_.
          2) Convert to 1D array if 2D.
          3) Make sure we have same length as feature_names.
          4) Sort by importances (argsort).
          5) Plot a barh plot.
        """
        import logging
        import numpy as np
        import matplotlib.pyplot as plt

        # 1) final_model
        try:
            final_model = pipeline["model"]
        except Exception as ex:
            logging.error(f"Could not access pipeline['model']: {ex}")
            return

        if not hasattr(final_model, "feature_importances_"):
            logging.warning(f"Model '{model_name}' has no feature_importances_. Skipping.")
            return

        importances = final_model.feature_importances_

        # Debug logging
        logging.debug(f"[{model_name}] importances type={type(importances)}, shape={getattr(importances, 'shape', None)}")
        logging.debug(f"[{model_name}] feature_names type={type(feature_names)}, length={len(feature_names)}")

        # 2) If importances is 2D => average across axis=0 or pick first row
        if importances.ndim > 1:
            logging.warning(f"[{model_name}] importances is {importances.shape}-dim. Taking mean(axis=0).")
            importances = importances.mean(axis=0)

        importances = np.array(importances, dtype=np.float64)

        # Convert feature_names to a standard list
        feature_names_list = list(feature_names)

        # 3) Check length
        if len(importances) != len(feature_names_list):
            logging.error(
                f"[{model_name}] mismatch: len(importances)={len(importances)} vs. "
                f"len(feature_names)={len(feature_names_list)}. Skipping plot."
            )
            return

        # 4) Sort
        sorted_idx = np.argsort(importances).astype(int)
        sorted_importances = importances[sorted_idx]
        sorted_names = [feature_names_list[int(i)] for i in sorted_idx]

        # 5) barh plot
        try:
            plt.figure(figsize=(10, max(4, len(feature_names_list) * 0.3)))
            plt.barh(sorted_names, sorted_importances, color=self.colors[0])
            plt.xlabel("Importance")
            # Changed the ylabel for clarity:
            plt.ylabel("Feature Name")
            plt.title(f"Feature Importances - {model_name}")
            plt.tight_layout()

            out_path = self.output_manager.get_path("reports", "visualizations", f"{model_name}_feature_importance.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Feature importance plot saved: {out_path}")

        except Exception as e:
            logging.error(f"Error creating feature importance plot for {model_name}: {e}")
