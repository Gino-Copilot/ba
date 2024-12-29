# file: data_visualizer.py

import matplotlib
matplotlib.use('Agg')  # Backend festlegen
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureAnalyzer:
    """
    Analyzes and visualizes features of the given dataset.
    This class focuses on correlations, feature importance,
    mutual information, and group-based analysis.
    """

    def __init__(self, output_manager, target_column: str = 'label'):
        """
        Initializes the FeatureAnalyzer.

        Args:
            output_manager: Instance of OutputManager for handling output paths.
            target_column: Name of the target column (default is 'label').
        """
        self.output_manager = output_manager
        self.target_column = target_column
        self.results: Dict[str, any] = {}

        # Configuration for plots
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        sns.set_style('whitegrid')

        logging.info("FeatureAnalyzer initialized.")

    def analyze_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Performs a full feature analysis on the provided DataFrame.

        Args:
            df: The input DataFrame containing features and the target.

        Returns:
            A dictionary with analysis results (e.g., group stats,
            correlation matrix, feature importances, etc.).
        """
        try:
            logging.info("Starting feature analysis...")

            # Validate input
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")

            # Separate features from target
            features = [col for col in df.columns if col != self.target_column]
            if not features:
                raise ValueError("No valid features found.")

            # Grouping features (optional, depends on naming conventions)
            feature_groups = self._create_feature_groups(features)
            group_stats = self._analyze_feature_groups(df, feature_groups)

            # Correlation matrix
            correlation_matrix = self._calculate_correlations(df[features])

            # Random Forest importance
            rf_importance = self._calculate_feature_importance(df[features], df[self.target_column])

            # Mutual information
            mi_scores = self._calculate_mutual_information(df[features], df[self.target_column])

            # Save CSVs and plots
            self._save_correlations(correlation_matrix)
            self._save_importances(rf_importance, filename="feature_importance.csv")
            self._save_importances(mi_scores, filename="mutual_information.csv", score_column='mutual_info')
            self._create_visualizations(correlation_matrix, rf_importance, mi_scores)

            # Aggregate results
            self.results = {
                'group_stats': group_stats,
                'correlation_matrix': correlation_matrix,
                'feature_importance': rf_importance,
                'mutual_information': mi_scores,
                'feature_groups': feature_groups
            }

            self._save_analysis_summary()
            logging.info("Feature analysis completed successfully.")
            return self.results

        except Exception as e:
            logging.error(f"Error during feature analysis: {e}")
            return {}

    def _create_feature_groups(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Attempts to group features by name or type (e.g., timing, packet, byte).

        Args:
            features: List of all feature names.

        Returns:
            A dictionary of group_name -> list_of_features.
        """
        logging.info("Creating feature groups...")
        try:
            groups = {
                'timing': [f for f in features if any(x in f.lower() for x in ['time', 'duration', 'iat'])],
                'packet': [f for f in features if 'packet' in f.lower()],
                'byte': [f for f in features if 'byte' in f.lower()],
                'ratio': [f for f in features if 'ratio' in f.lower()],
                'other': []
            }

            grouped_features = set(sum(groups.values(), []))
            groups['other'] = [f for f in features if f not in grouped_features]
            return groups

        except Exception as e:
            logging.error(f"Error in _create_feature_groups: {e}")
            return {}

    def _analyze_feature_groups(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Analyzes each feature group with a simple RandomForestClassifier.

        Args:
            df: The full DataFrame (including the target column).
            feature_groups: A dictionary of group_name -> feature_list.

        Returns:
            A dictionary of group_name -> stats (accuracy, importance, etc.).
        """
        logging.info("Analyzing feature groups...")
        group_stats = {}

        for group_name, feat_list in feature_groups.items():
            if not feat_list:
                continue  # skip empty groups

            try:
                X_group = df[feat_list]
                y = df[self.target_column]

                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_group, y)

                stats = {
                    'feature_count': len(feat_list),
                    'accuracy': rf.score(X_group, y),
                    'importances': dict(zip(feat_list, rf.feature_importances_))
                }
                group_stats[group_name] = stats

                # Optional: Save group stats to CSV
                self._save_group_stats(group_name, stats)

            except Exception as e:
                logging.error(f"Error analyzing group '{group_name}': {e}")

        return group_stats

    def _calculate_correlations(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the correlation matrix for the provided features.

        Args:
            df_features: A DataFrame containing only feature columns.

        Returns:
            A pandas DataFrame representing the correlation matrix.
        """
        logging.info("Calculating correlation matrix...")
        try:
            return df_features.corr()
        except Exception as e:
            logging.error(f"Error in _calculate_correlations: {e}")
            return pd.DataFrame()

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Trains a small RandomForest to estimate feature importance.

        Args:
            X: Feature matrix.
            y: Target array or Series.

        Returns:
            A DataFrame sorted by feature importance in descending order.
        """
        logging.info("Calculating RandomForest feature importance...")
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            imp_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            return imp_df
        except Exception as e:
            logging.error(f"Error in _calculate_feature_importance: {e}")
            return pd.DataFrame()

    def _calculate_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculates mutual information scores between features and the target.

        Args:
            X: Feature matrix.
            y: Target array or Series.

        Returns:
            A DataFrame with columns ['feature', 'mutual_info'], sorted descending.
        """
        logging.info("Calculating mutual information scores...")
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_df = pd.DataFrame({
                'feature': X.columns,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
            return mi_df
        except Exception as e:
            logging.error(f"Error in _calculate_mutual_information: {e}")
            return pd.DataFrame()

    def _create_visualizations(
        self,
        correlation_matrix: pd.DataFrame,
        rf_importance: pd.DataFrame,
        mi_scores: pd.DataFrame
    ):
        """
        Creates and saves visualizations (heatmaps, bar plots, etc.).

        Args:
            correlation_matrix: The correlation matrix of features.
            rf_importance: DataFrame of feature importances from RandomForest.
            mi_scores: DataFrame of mutual information scores.
        """
        logging.info("Creating visualizations...")
        try:
            # Plot correlation heatmap
            if not correlation_matrix.empty:
                self._plot_correlation_heatmap(correlation_matrix)

            # Plot RF feature importance
            if not rf_importance.empty:
                self._plot_importance_bar(
                    rf_importance,
                    x_col='importance',
                    y_col='feature',
                    title='RandomForest Feature Importance',
                    filename='feature_importance.png'
                )

            # Plot mutual information scores
            if not mi_scores.empty:
                self._plot_importance_bar(
                    mi_scores,
                    x_col='mutual_info',
                    y_col='feature',
                    title='Mutual Information Scores',
                    filename='mutual_information.png'
                )
        except Exception as e:
            logging.error(f"Error in _create_visualizations: {e}")

    def _plot_correlation_heatmap(self, corr: pd.DataFrame):
        """Helper to plot and save correlation heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='RdBu', center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        self._save_figure("features", "correlations", "correlation_heatmap.png")

    def _plot_importance_bar(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        filename: str
    ):
        """Helper to plot and save bar chart for feature importances."""
        plt.figure(figsize=(10, min(15, len(df) * 0.4)))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title)
        plt.tight_layout()
        self._save_figure("features", "importance", filename)

    def _save_group_stats(self, group_name: str, stats: Dict[str, any]):
        """
        Saves group stats (e.g., accuracy, top features) to CSV.

        Args:
            group_name: Name of the group.
            stats: Dictionary with information about that group.
        """
        try:
            # You could store more details as needed
            feat_importances = stats.get('importances', {})
            df = pd.DataFrame(list(feat_importances.items()), columns=['feature', 'importance'])
            output_path = self.output_manager.get_path("features", "groups", f"group_{group_name}.csv")
            df.to_csv(output_path, index=False)
            logging.info(f"Group stats saved for {group_name}: {output_path}")
        except Exception as e:
            logging.error(f"Error in _save_group_stats for group '{group_name}': {e}")

    def _save_correlations(self, corr: pd.DataFrame):
        """
        Saves the correlation matrix to CSV.

        Args:
            corr: The correlation matrix DataFrame.
        """
        try:
            output_path = self.output_manager.get_path("features", "correlations", "correlation_matrix.csv")
            corr.to_csv(output_path)
            logging.info(f"Correlation matrix saved to: {output_path}")
        except Exception as e:
            logging.error(f"Error saving correlation matrix: {e}")

    def _save_importances(self, df: pd.DataFrame, filename: str, score_column: str = 'importance'):
        """
        Saves feature importance or mutual information scores to CSV.

        Args:
            df: The DataFrame with scores (e.g., importance or MI).
            filename: The output CSV filename.
            score_column: Name of the column with the score (default 'importance').
        """
        try:
            output_path = self.output_manager.get_path("features", "importance", filename)
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {score_column} to: {output_path}")
        except Exception as e:
            logging.error(f"Error saving importances: {e}")

    def _save_analysis_summary(self):
        """
        Writes a textual summary of the analysis results.
        """
        try:
            summary_path = self.output_manager.get_path("features", "summaries", "complete_analysis.txt")
            with open(summary_path, 'w') as file:
                file.write("=== Feature Analysis Summary ===\n\n")

                # Feature Groups
                file.write("Feature Groups:\n")
                for group_name, stats in self.results.get('group_stats', {}).items():
                    file.write(f"\n{group_name.upper()}:\n")
                    file.write(f"  Features: {stats['feature_count']}\n")
                    file.write(f"  Accuracy: {stats['accuracy']:.4f}\n")

                    # Example: show top 3 features
                    sorted_feats = sorted(stats['importances'].items(), key=lambda x: x[1], reverse=True)
                    top_3 = sorted_feats[:3]
                    file.write("  Top 3 features:\n")
                    for feat, imp in top_3:
                        file.write(f"    {feat}: {imp:.4f}\n")

                # Top 10 Features by Importance
                feature_importance_df = self.results.get('feature_importance', pd.DataFrame())
                if not feature_importance_df.empty:
                    file.write("\nTop 10 Features by RF Importance:\n")
                    for _, row in feature_importance_df.head(10).iterrows():
                        file.write(f"  {row['feature']}: {row['importance']:.4f}\n")

                # Top 10 Features by Mutual Information
                mi_df = self.results.get('mutual_information', pd.DataFrame())
                if not mi_df.empty:
                    file.write("\nTop 10 Features by Mutual Information:\n")
                    for _, row in mi_df.head(10).iterrows():
                        file.write(f"  {row['feature']}: {row['mutual_info']:.4f}\n")

            logging.info(f"Analysis summary saved to {summary_path}")

        except Exception as e:
            logging.error(f"Error saving analysis summary: {e}")

    def _save_figure(self, category: str, subcategory: str, filename: str):
        """
        Saves the current figure using the output_manager.

        Args:
            category: The top-level output category (e.g. 'features').
            subcategory: The subcategory (e.g. 'importance').
            filename: Name of the file to save.
        """
        try:
            path = self.output_manager.get_path(category, subcategory, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Figure saved: {path}")
        except Exception as e:
            logging.error(f"Error saving figure '{filename}': {e}")
