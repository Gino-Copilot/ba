import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import os


class FeatureAnalyzer:
   def __init__(self):
       self.visualizer = None

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

       # 3. Check correlations
       print("\n=== Feature Correlations ===")
       correlation_matrix = df[features].corr()
       self._plot_correlation_matrix(correlation_matrix, 'feature_correlations.png')

       # 4. Calculate importance
       print("\n=== Feature Importance Ranking ===")
       importance_scores = self._calculate_feature_importance(df[features], df[target])

       # 5. Calculate mutual information
       print("\n=== Mutual Information Scores ===")
       mi_scores = self._calculate_mutual_information(df[features], df[target])

       # 6. Plot distributions
       print("\n=== Feature Distributions ===")
       self._plot_feature_distributions(df, features, target)

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

           # Train Random Forest with group features only
           rf = RandomForestClassifier(random_state=42)
           rf.fit(X_group, y)
           acc = rf.score(X_group, y)

           # Calculate statistics
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

   def _plot_correlation_matrix(self, correlation_matrix, filename):
       """Create correlation matrix visualization"""
       output_dir = "feature_correlations"  # Create folder for correlations
       os.makedirs(output_dir, exist_ok=True)  # Make folder if it doesn't exist
       filepath = os.path.join(output_dir, filename)

       plt.figure(figsize=(12, 8))
       sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
       plt.title('Feature Correlation Matrix')
       plt.tight_layout()
       plt.savefig(filepath)  # Save to new path
       plt.close()

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

   def _plot_feature_distributions(self, df, features, target):
       """Plot distributions for the most important features"""
       output_dir = "feature_distributions"  # Create folder for distributions
       os.makedirs(output_dir, exist_ok=True)  # Make folder if it doesn't exist
       filepath = os.path.join(output_dir, 'feature_distributions.png')

       top_features = features[:5]  # Only first 5 features

       plt.figure(figsize=(15, 10))
       for i, feature in enumerate(top_features, 1):
           plt.subplot(2, 3, i)
           for label in df[target].unique():
               sns.kdeplot(df[df[target] == label][feature], label=label)
           plt.title(f'Distribution: {feature}')
           plt.legend()

       plt.tight_layout()
       plt.savefig(filepath)  # Save to path
       plt.close()

   def analyze_feature_contribution(self, df, target='label'):
       """Analyze how each feature contributes to classification"""
       features = [col for col in df.columns if col != target]
       results = {}

       print("\n=== Feature Contribution Analysis ===")

       # Baseline with all features
       rf_all = RandomForestClassifier(random_state=42)
       X_all = df[features]
       y = df[target]
       rf_all.fit(X_all, y)
       baseline_acc = rf_all.score(X_all, y)
       print(f"Baseline accuracy (all features): {baseline_acc:.4f}")

       # Leave-one-out analysis
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