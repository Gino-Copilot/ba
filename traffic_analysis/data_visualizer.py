import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from contextlib import contextmanager


class DataVisualizer:
    def __init__(self, output_manager):
        """
        Initialize DataVisualizer with an OutputManager

        Args:
            output_manager: Instance of OutputManager class for handling output paths
        """
        self.model_results = []
        self.output_manager = output_manager

    @contextmanager
    def plot_context(self, figsize=(10, 6)):
        """Simple context manager for matplotlib figures"""
        try:
            fig = plt.figure(figsize=figsize)
            yield fig
        finally:
            plt.close(fig)

    def plot_feature_importance(self, model, X):
        """
        Create and save feature importance plot

        Args:
            model: Trained model with feature_importances_ attribute
            X: DataFrame with features
        """
        if not hasattr(model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        plot_path = self.output_manager.get_path(
            "models", "plots", "feature_importance.png"
        )

        with self.plot_context(figsize=(12, 8)) as fig:
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance Analysis')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        return importance_df

    def plot_roc_curve(self, model, X_test, y_test):
        """
        Create and save ROC curve plot

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        if not hasattr(model, 'predict_proba'):
            return None

        from sklearn.metrics import roc_curve, auc
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plot_path = self.output_manager.get_path(
            "models", "plots", "roc_curve.png"
        )

        with self.plot_context(figsize=(8, 8)) as fig:
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        return roc_auc

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Create and save confusion matrix plot

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        plot_path = self.output_manager.get_path(
            "models", "plots", "confusion_matrix.png"
        )

        with self.plot_context(figsize=(8, 6)) as fig:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    def add_model_result(self, model_name, metrics):
        """
        Add results from a model

        Args:
            model_name: Name of the model
            metrics: Dictionary containing metrics
        """
        self.model_results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['weighted avg']['precision'],
            'Recall': metrics['weighted avg']['recall'],
            'F1-Score': metrics['weighted avg']['f1-score'],
            'ROC AUC': metrics.get('roc_auc', None)
        })

    def plot_comprehensive_comparison(self):
        """Create complete model comparison visualizations"""
        if not self.model_results:
            return

        df = pd.DataFrame(self.model_results)

        # 1. Create bar plot for all metrics
        plot_path = self.output_manager.get_path(
            "models", "plots", "model_metrics_comparison.png"
        )

        with self.plot_context(figsize=(15, 8)) as fig:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
            bar_width = 0.15
            index = np.arange(len(df))

            # Plot each metric as a bar
            for i, metric in enumerate(metrics):
                if metric in df.columns:
                    plt.bar(index + i * bar_width,
                            df[metric],
                            bar_width,
                            label=metric,
                            alpha=0.8)

            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Metrics Comparison')
            plt.xticks(index + bar_width * 2, df['Model'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        # 2. Create heatmap of metrics
        heatmap_path = self.output_manager.get_path(
            "models", "plots", "model_metrics_heatmap.png"
        )

        with self.plot_context(figsize=(12, 8)) as fig:
            metrics_df = df.set_index('Model')
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd',
                        cbar_kws={'label': 'Score'})
            plt.title('Model Performance Metrics Heatmap')
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')

    def save_comparison_table(self):
        """Save detailed comparison table and summary"""
        if not self.model_results:
            return

        df = pd.DataFrame(self.model_results)

        # Sort by accuracy (best models first)
        df = df.sort_values('Accuracy', ascending=False)

        # Format percentages for better readability
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")

        # Save to CSV
        csv_path = self.output_manager.get_path(
            "models", "metrics", "model_comparison.csv"
        )
        df.to_csv(csv_path, index=False)

        # Create and save summary
        summary = self._generate_summary()
        summary_path = self.output_manager.get_path(
            "models", "metrics", "model_comparison_summary.txt"
        )
        with open(summary_path, 'w') as f:
            f.write(summary)

    def _generate_summary(self):
        """
        Create summary of model performance

        Returns:
            str: Formatted summary text
        """
        df = pd.DataFrame(self.model_results)

        summary = "=== Model Performance Summary ===\n\n"

        # Find best model for each metric
        summary += "Best Performing Models:\n"
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        for metric in metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_model = df.iloc[best_idx]['Model']
                best_score = df.iloc[best_idx][metric]
                summary += f"{metric}: {best_model} ({best_score * 100:.2f}%)\n"

        # Calculate average performance
        summary += "\nAverage Performance:\n"
        for metric in metrics:
            if metric in df.columns:
                avg = df[metric].mean()
                std = df[metric].std()
                summary += f"{metric}: {avg * 100:.2f}% (±{std * 100:.2f}%)\n"

        return summary

    def plot_model_comparison(self, results):
        """
        Create model comparison plots and save results

        Args:
            results: Dictionary with model results
        """
        # Store results first
        for model_name, metrics in results.items():
            self.add_model_result(model_name, metrics)

        # Create detailed visualizations
        self.plot_comprehensive_comparison()
        self.save_comparison_table()

        # Create simple bar plot for quick overview
        plot_path = self.output_manager.get_path(
            "models", "plots", "simple_model_comparison.png"
        )

        accuracies = {name: data['accuracy'] for name, data in results.items()}

        with self.plot_context(figsize=(10, 6)) as fig:
            plt.bar(accuracies.keys(), accuracies.values())
            plt.xticks(rotation=45, ha='right')
            plt.title('Model Performance Comparison')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')