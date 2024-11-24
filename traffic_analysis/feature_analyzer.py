import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd


class FeatureAnalyzer:
    def __init__(self, visualizer=None):
        """
        Initialize Feature Analyzer
        Args:
            visualizer: DataVisualizer instance for plotting results
        """
        self.visualizer = visualizer

    def analyze_features(self, df, target='label'):
        """Complete feature analysis"""
        results = {}
        features = [col for col in df.columns if col != target]

        print("\n=== Complete Feature Analysis ===")

        # 1. Group features by type
        feature_groups = {
            'timing': [f for f in features if any(x in f.lower() for x in ['time', 'duration', 'iat'])],
            'packet': [f for f in features if 'packet' in f.lower()],
            'byte': [f for f in features if 'byte' in f.lower()],
            'entropy': [f for f in features if 'entropy' in f.lower()],
            'ratio': [f for f in features if 'ratio' in f.lower()]
        }

        # 2. Analyze each group
        print("\n=== Feature Group Statistics ===")
        group_stats = self._analyze_feature_groups(df, feature_groups, target)

        # 3. Calculate correlations
        correlation_matrix = df[features].corr()

        # 4. Calculate importance
        importance_scores = self._calculate_feature_importance(df[features], df[target])

        # 5. Calculate mutual information
        mi_scores = self._calculate_mutual_information(df[features], df[target])

        # If visualizer is provided, create plots
        if self.visualizer:
            self.visualizer.plot_correlation_matrix(correlation_matrix)
            self.visualizer.plot_feature_distributions(df, features, target)

        return {
            'group_stats': group_stats,
            'correlation_matrix': correlation_matrix,
            'feature_importance': importance_scores,
            'mutual_information': mi_scores
        }

    def _analyze_feature_groups(self, df, feature_groups, target):
        """Analyze features by their groups"""
        group_stats = {}

        for group_name, features in feature_groups.items():
            if not features:
                continue

            print(f"\nGroup: {group_name}")
            X_group = df[features]
            y = df[target]

            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_group, y)
            acc = rf.score(X_group, y)

            stats = {
                'accuracy': acc,
                'feature_count': len(features),
                'top_features': dict(sorted(
                    zip(features, rf.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
                ))
            }

            print(f"Accuracy using only {group_name} features: {acc:.4f}")
            print("Top 3 features in this group:")
            for feat, imp in list(stats['top_features'].items())[:3]:
                print(f"  {feat}: {imp:.4f}")

            group_stats[group_name] = stats

        return group_stats

    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance using Random Forest"""
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
        """Calculate mutual information scores"""
        mi_scores = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        print("\nTop 10 features by mutual information:")
        print(mi_df.head(10))

        return mi_df

    def analyze_feature_contribution(self, df, target='label'):
        """Analyze how each feature contributes to classification"""
        features = [col for col in df.columns if col != target]
        results = {}

        print("\n=== Feature Contribution Analysis ===")

        rf_all = RandomForestClassifier(random_state=42)
        X_all = df[features]
        y = df[target]
        rf_all.fit(X_all, y)
        baseline_acc = rf_all.score(X_all, y)
        print(f"Baseline accuracy (all features): {baseline_acc:.4f}")

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

        return results
