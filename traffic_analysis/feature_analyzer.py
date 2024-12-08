# feature_analyzer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureAnalyzer:
    def __init__(self, output_manager):
        """
        Initialisiert den FeatureAnalyzer

        Args:
            output_manager: Instance des OutputManagers für das Handling von Ausgabepfaden
        """
        self.output_manager = output_manager
        self.results = {}

    def analyze_features(self, df, target='label'):
        """
        Führt eine vollständige Feature-Analyse durch

        Args:
            df: DataFrame mit den zu analysierenden Daten
            target: Name der Zielvariable (default: 'label')

        Returns:
            dict: Dictionary mit allen Analyseergebnissen
        """
        try:
            print("\nStarting feature analysis...")
            features = [col for col in df.columns if col != target]

            # Grundlegende Validierung
            if len(features) == 0:
                raise ValueError("No features found in DataFrame")
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame")

            # Analyse durchführen
            feature_groups = self._create_feature_groups(features)
            group_stats = self._analyze_feature_groups(df, feature_groups, target)
            correlation_matrix = self._calculate_correlations(df[features])
            importance_scores = self._calculate_feature_importance(df[features], df[target])
            mi_scores = self._calculate_mutual_information(df[features], df[target])

            # Ergebnisse speichern
            self._save_correlation_matrix(correlation_matrix)
            self._save_importance_scores(importance_scores)
            self._save_mutual_information_scores(mi_scores)

            # Visualisierungen erstellen
            self._create_visualizations(correlation_matrix, importance_scores, mi_scores)

            # Gesamtergebnisse zusammenstellen
            self.results = {
                'group_stats': group_stats,
                'correlation_matrix': correlation_matrix,
                'feature_importance': importance_scores,
                'mutual_information': mi_scores,
                'feature_groups': feature_groups
            }

            self._save_analysis_summary()
            print("Feature analysis completed successfully!")
            return self.results

        except Exception as e:
            print(f"Error during feature analysis: {str(e)}")
            raise

    def _create_feature_groups(self, features):
        """Gruppiert Features nach ihren Charakteristiken"""
        try:
            groups = {
                'timing': [f for f in features if any(x in f.lower() for x in ['time', 'duration', 'iat'])],
                'packet': [f for f in features if 'packet' in f.lower()],
                'byte': [f for f in features if 'byte' in f.lower()],
                'ratio': [f for f in features if 'ratio' in f.lower()],
                'other': []
            }

            # Füge Features zur 'other' Gruppe hinzu, die noch nicht kategorisiert wurden
            categorized = set(sum(groups.values(), []))
            groups['other'] = [f for f in features if f not in categorized]

            return groups
        except Exception as e:
            print(f"Error in feature grouping: {str(e)}")
            return {}

    def _analyze_feature_groups(self, df, feature_groups, target):
        """Analysiert jede Feature-Gruppe einzeln"""
        group_stats = {}

        for group_name, features in feature_groups.items():
            if not features:
                continue

            try:
                X_group = df[features]
                y = df[target]

                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_group, y)

                stats = {
                    'accuracy': rf.score(X_group, y),
                    'feature_count': len(features),
                    'top_features': dict(sorted(
                        zip(features, rf.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    )),
                    'mean_importance': np.mean(rf.feature_importances_)
                }

                # Speichere detaillierte Gruppen-Statistiken
                self._save_group_stats(group_name, stats)
                group_stats[group_name] = stats

            except Exception as e:
                print(f"Error analyzing feature group {group_name}: {str(e)}")
                continue

        return group_stats

    def _calculate_correlations(self, features_df):
        """Berechnet die Korrelationsmatrix"""
        try:
            return features_df.corr()
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame()

    def _calculate_feature_importance(self, X, y):
        """Berechnet Feature Importance mit Random Forest"""
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)

            return pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame()

    def _calculate_mutual_information(self, X, y):
        """Berechnet Mutual Information zwischen Features und Target"""
        try:
            mi_scores = mutual_info_classif(X, y)
            return pd.DataFrame({
                'feature': X.columns,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
        except Exception as e:
            print(f"Error calculating mutual information: {str(e)}")
            return pd.DataFrame()

    def _create_visualizations(self, correlation_matrix, importance_df, mi_df):
        """Erstellt alle Visualisierungen"""
        try:
            self._plot_correlation_heatmap(correlation_matrix)
            self._plot_feature_importance(importance_df)
            self._plot_mutual_information(mi_df)
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_correlation_heatmap(self, correlation_matrix):
        """Erstellt Heatmap der Feature-Korrelationen"""
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(self.output_manager.get_path(
                "features", "correlations", "correlation_heatmap.png"
            ))
            plt.close()
        except Exception as e:
            print(f"Error plotting correlation heatmap: {str(e)}")

    def _plot_feature_importance(self, importance_df):
        """Visualisiert Feature Importance"""
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(self.output_manager.get_path(
                "features", "importance", "feature_importance.png"
            ))
            plt.close()
        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")

    def _plot_mutual_information(self, mi_df):
        """Visualisiert Mutual Information Scores"""
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=mi_df, x='mutual_info', y='feature')
            plt.title('Mutual Information with Target')
            plt.tight_layout()
            plt.savefig(self.output_manager.get_path(
                "features", "importance", "mutual_information.png"
            ))
            plt.close()
        except Exception as e:
            print(f"Error plotting mutual information: {str(e)}")

    def _save_correlation_matrix(self, correlation_matrix):
        """Speichert die Korrelationsmatrix"""
        try:
            correlation_matrix.to_csv(self.output_manager.get_path(
                "features", "correlations", "correlation_matrix.csv"
            ))
        except Exception as e:
            print(f"Error saving correlation matrix: {str(e)}")

    def _save_importance_scores(self, importance_df):
        """Speichert die Feature Importance Scores"""
        try:
            importance_df.to_csv(self.output_manager.get_path(
                "features", "importance", "feature_importance.csv"
            ))
        except Exception as e:
            print(f"Error saving importance scores: {str(e)}")

    def _save_mutual_information_scores(self, mi_df):
        """Speichert die Mutual Information Scores"""
        try:
            mi_df.to_csv(self.output_manager.get_path(
                "features", "importance", "mutual_information.csv"
            ))
        except Exception as e:
            print(f"Error saving mutual information scores: {str(e)}")

    def _save_group_stats(self, group_name, stats):
        """Speichert die Statistiken für eine Feature-Gruppe"""
        try:
            pd.DataFrame(list(stats['top_features'].items()),
                         columns=['Feature', 'Importance']).to_csv(
                self.output_manager.get_path(
                    "features", "groups", f"group_{group_name}.csv"
                )
            )
        except Exception as e:
            print(f"Error saving group stats for {group_name}: {str(e)}")

    def _save_analysis_summary(self):
        """Speichert eine vollständige Analyse-Zusammenfassung"""
        try:
            summary_path = self.output_manager.get_path(
                "features", "summaries", "complete_analysis.txt"
            )

            with open(summary_path, 'w') as f:
                f.write("=== Feature Analysis Summary ===\n\n")

                # Gruppen-Statistiken
                f.write("Feature Groups:\n")
                for group, stats in self.results['group_stats'].items():
                    f.write(f"\n{group.upper()}:\n")
                    f.write(f"Features: {stats['feature_count']}\n")
                    f.write(f"Accuracy: {stats['accuracy']:.4f}\n")
                    f.write(f"Mean Importance: {stats['mean_importance']:.4f}\n")
                    f.write("Top 3 features:\n")
                    for feat, imp in list(stats['top_features'].items())[:3]:
                        f.write(f"  {feat}: {imp:.4f}\n")

                # Top Features nach verschiedenen Metriken
                f.write("\nTop 10 Features by Importance:\n")
                for _, row in self.results['feature_importance'].head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

                f.write("\nTop 10 Features by Mutual Information:\n")
                for _, row in self.results['mutual_information'].head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['mutual_info']:.4f}\n")

        except Exception as e:
            print(f"Error saving analysis summary: {str(e)}")