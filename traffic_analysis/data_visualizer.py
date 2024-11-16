import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from contextlib import contextmanager


class DataVisualizer:
    @contextmanager
    def plot_context(self, figsize=(10, 6)):
        """Context manager for handling matplotlib figures"""
        try:
            fig = plt.figure(figsize=figsize)
            yield fig
        finally:
            plt.close(fig)

    def plot_feature_importance(self, model, X, output_dir, timestamp):
        """Plot and save feature importance"""
        if not hasattr(model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        plot_path = os.path.join(output_dir, f'feature_importance_{timestamp}.png')

        with self.plot_context(figsize=(12, 8)) as fig:
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance Analysis')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        return importance_df

    def plot_roc_curve(self, model, X_test, y_test, output_dir, timestamp):
        """Plot and save ROC curve"""
        if not hasattr(model, 'predict_proba'):
            return None

        from sklearn.metrics import roc_curve, auc
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plot_path = os.path.join(output_dir, f'roc_curve_{timestamp}.png')

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

    def plot_confusion_matrix(self, y_true, y_pred, output_dir, timestamp):
        """Plot and save confusion matrix"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')

        with self.plot_context(figsize=(8, 6)) as fig:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')