import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd


class FeatureAnalyzer:
    def __init__(self, output_manager):
        """
        Initialize Feature Analyzer with output management

        Args:
            output_manager: Instance of OutputManager for handling output paths
        """
        self.output_manager = output_manager

    def analyze_features(self, df, target='label'):
        """
        Complete feature analysis including correlations, importance, and group statistics

        Args:
            df: DataFrame with features and target
            target: Name of target column (default: 'label')

        Returns:
            dict: Results of all analyses
        """
        results = {}
        features = [col for col in df.columns if col != target]

        print("\n=== Complete Feature Analysis ===")

        # 1. Group features by type
        feature_groups = self._create_feature_groups(features)

        # 2. Analyze each group
        print("\n=== Feature Group Statistics ===")
        group_stats = self._analyze_feature_groups(df, feature_groups, target)

        # Save group statistics
        group_stats_path = self.output_manager.get_path(
            "features", "importance", "group_statistics.csv"
        )
        pd.DataFrame(group_stats).to_csv(group_stats_path)

        # 3. Calculate correlations
        print("\n=== Feature Correlations ===")
        correlation_matrix = df[features].corr()

        # Save correlation matrix
        corr_path = self.output_manager.get_path(
            "features", "correlations", "correlation_matrix.csv"
        )
        correlation_matrix.to_csv(corr_path)

        # 4. Calculate importance
        print("\n=== Feature Importance Ranking ===")
        importance_scores = self._calculate_feature_importance(df[features], df[target])

        # Save importance scores
        importance_path = self.output_manager.get_path(
            "features", "importance", "feature_importance.csv"
        )
        importance_scores.to_csv(importance_path)

        # 5. Calculate mutual information
        print("\n=== Mutual Information Scores ===")
        mi_scores = self._calculate_mutual_information(df[features], df[target])

        # Save mutual information scores
        mi_path = self.output_manager.get_path(
            "features", "importance", "mutual_information.csv"
        )
        mi_scores.to_csv(mi_path)

        results = {
            'group_stats': group_stats,
            'correlation_matrix': correlation_matrix,
            'feature_importance': importance_scores,
            'mutual_information': mi_scores
        }

        # Save complete analysis results
        results_path = self.output_manager.get_path(
            "features", "summaries", "complete_analysis.txt"
        )
        self._save_analysis_summary(results, results_path)

        return results

    def _create_feature_groups(self, features):
        """
        Group features by their type based on name patterns

        Args:
            features: List of feature names

        Returns:
            dict: Groups of features
        """
        return {
            'timing': [f for f in features if any(x in f.lower() for x in ['time', 'duration', 'iat'])],
            'packet': [f for f in features if 'packet' in f.lower()],
            'byte': [f for f in features if 'byte' in f.lower()],
            'entropy': [f for f in features if 'entropy' in f.lower()],
            'ratio': [f for f in features if 'ratio' in f.lower()]
        }

    def _analyze_feature_groups(self, df, feature_groups, target):
        """
        Analyze each feature group using Random Forest

        Args:
            df: DataFrame with features
            feature_groups: Dictionary of feature groups
            target: Target variable name

        Returns:
            dict: Statistics for each group
        """
        group_stats = {}

        for group_name, features in feature_groups.items():
            if not features:
                continue

            print(f"\nGroup: {group_name}")
            X_group = df[features]
            y = df[target]

            # Train Random Forest with group features
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_group, y)
            acc = rf.score(X_group, y)

            # Calculate group statistics
            stats = {
                'accuracy': acc,
                'feature_count': len(features),
                'top_features': dict(sorted(
                    zip(features, rf.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
                ))
            }

            # Save group-specific results
            group_path = self.output_manager.get_path(
                "features", "groups", f"group_{group_name}.csv"
            )
            pd.DataFrame(list(stats['top_features'].items()),
                         columns=['Feature', 'Importance']).to_csv(group_path)

            print(f"Accuracy using only {group_name} features: {acc:.4f}")
            print("Top 3 features in this group:")
            for feat, imp in list(stats['top_features'].items())[:3]:
                print(f"  {feat}: {imp:.4f}")

            group_stats[group_name] = stats

        return group_stats

    def _calculate_feature_importance(self, X, y):
        """
        Calculate feature importance using Random Forest

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            DataFrame: Feature importance scores
        """
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 most important features:")
        print(importance_df.head(10))

        return importance_df

    def _calculate_mutual_information(self, X, y):
        """
        Calculate mutual information between features and target

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            DataFrame: Mutual information scores
        """
        mi_scores = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        print("\nTop 10 features by mutual information:")
        print(mi_df.head(10))

        return mi_df

    def _save_analysis_summary(self, results, path):
        """
        Save complete analysis summary to file

        Args:
            results: Dictionary with all analysis results
            path: Output file path
        """
        with open(path, 'w') as f:
            f.write("=== Feature Analysis Summary ===\n\n")

            # Group statistics summary
            f.write("Feature Group Performance:\n")
            for group, stats in results['group_stats'].items():
                f.write(f"\n{group.upper()}:\n")
                f.write(f"Number of features: {stats['feature_count']}\n")
                f.write(f"Group accuracy: {stats['accuracy']:.4f}\n")
                f.write("Top 3 features:\n")
                for feat, imp in list(stats['top_features'].items())[:3]:
                    f.write(f"  {feat}: {imp:.4f}\n")

            # Top features by different metrics
            f.write("\nTop 10 Features by Importance:\n")
            for _, row in results['feature_importance'].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

            f.write("\nTop 10 Features by Mutual Information:\n")
            for _, row in results['mutual_information'].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['mutual_info']:.4f}\n")

    def analyze_feature_contribution(self, df, target='label'):
        """
        Analyze how each feature contributes to classification

        Args:
            df: DataFrame with features and target
            target: Target column name

        Returns:
            dict: Impact of each feature on model performance
        """
        features = [col for col in df.columns if col != target]
        results = {}

        print("\n=== Feature Contribution Analysis ===")

        # Train baseline model with all features
        rf_all = RandomForestClassifier(random_state=42)
        X_all = df[features]
        y = df[target]
        rf_all.fit(X_all, y)
        baseline_acc = rf_all.score(X_all, y)
        print(f"Baseline accuracy (all features): {baseline_acc:.4f}")

        # Test removing each feature
        for feature in features:
            features_without = [f for f in features if f != feature]
            rf = RandomForestClassifier(random_state=42)
            rf.fit(df[features_without], y)
            acc = rf.score(df[features_without], y)
            impact = baseline_acc - acc
            results[feature] = {
                'accuracy_without': acc,
                'impact': impact
            }
            print(f"\nWithout feature '{feature}':")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Impact: {impact:.4f}")

        # Save contribution analysis results
        contrib_path = self.output_manager.get_path(
            "features", "importance", "feature_contributions.csv"
        )
        pd.DataFrame(results).transpose().to_csv(contrib_path)

        return results