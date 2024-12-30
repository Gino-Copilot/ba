# file: feature_analyzer.py

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})
sns.set_style('whitegrid')

class FeatureAnalyzer:
    """
    Analyzes features: groups, correlation, RF importance, mutual info.
    """

    def __init__(self, output_manager, target_column='label'):
        self.output_manager = output_manager
        self.target_column = target_column
        self.results: Dict[str, any] = {}
        logging.info("FeatureAnalyzer initialized.")

    def analyze_features(self, df: pd.DataFrame) -> Dict[str, any]:
        try:
            logging.info("Starting feature analysis.")
            if self.target_column not in df.columns:
                raise ValueError(f"No target col '{self.target_column}'.")

            # Drop non-numeric except target
            obj_cols = [c for c in df.columns if df[c].dtype == object and c != self.target_column]
            if obj_cols:
                logging.info(f"Dropping non-numeric cols: {obj_cols}")
                df = df.drop(columns=obj_cols, errors='ignore')

            # List features
            feats = [c for c in df.columns if c != self.target_column]
            if not feats:
                logging.warning("No numeric features remain.")
                return {}

            # Groups
            groups = self._create_groups(feats)
            gstats = self._analyze_groups(df, groups)

            # Correlation
            corr = self._corr(df[feats])
            # RF importance
            rf_imp = self._rf_importance(df[feats], df[self.target_column])
            # Mutual info
            mi = self._mutual_info(df[feats], df[self.target_column])

            # Save
            self._save_correlations(corr)
            self._save_importances(rf_imp, "feature_importance.csv")
            self._save_importances(mi, "mutual_information.csv", score_col='mutual_info')
            self._plot_all(corr, rf_imp, mi)

            self.results = {
                'group_stats': gstats,
                'correlation_matrix': corr,
                'feature_importance': rf_imp,
                'mutual_information': mi,
                'feature_groups': groups
            }
            self._save_summary()
            logging.info("Feature analysis done.")
            return self.results

        except Exception as e:
            logging.error(f"Error in analyze_features: {e}")
            return {}

    def _create_groups(self, feats: List[str]) -> Dict[str, List[str]]:
        groups = {
            'timing': [f for f in feats if any(x in f.lower() for x in ['time','duration','iat'])],
            'packet': [f for f in feats if 'packet' in f.lower()],
            'byte':   [f for f in feats if 'byte'   in f.lower()],
            'ratio':  [f for f in feats if 'ratio'  in f.lower()],
            'other':  []
        }
        used = set(sum(groups.values(), []))
        groups['other'] = [f for f in feats if f not in used]
        return groups

    def _analyze_groups(self, df: pd.DataFrame, groups: Dict[str,List[str]]) -> Dict[str,any]:
        stats = {}
        y = df[self.target_column]
        for gname, glist in groups.items():
            try:
                valid = [f for f in glist if f in df.columns]
                if not valid:
                    continue
                Xg = df[valid]
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(Xg, y)
                stats[gname] = {
                    'feature_count': len(valid),
                    'accuracy': rf.score(Xg, y),
                    'importances': dict(zip(valid, rf.feature_importances_))
                }
                self._save_group_stats(gname, stats[gname])
            except Exception as ex:
                logging.error(f"Group '{gname}' error: {ex}")
        return stats

    def _corr(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        try:
            if df_feat.shape[1] < 2:
                return pd.DataFrame()  # not enough for correlation
            return df_feat.corr(numeric_only=True)
        except Exception as e:
            logging.error(f"Corr error: {e}")
            return pd.DataFrame()

    def _rf_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        try:
            if X.empty: return pd.DataFrame()
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X,y)
            df_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            return df_imp
        except Exception as e:
            logging.error(f"RF importance error: {e}")
            return pd.DataFrame()

    def _mutual_info(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        try:
            if X.empty: return pd.DataFrame()
            mis = mutual_info_classif(X, y, random_state=42)
            df_mi = pd.DataFrame({
                'feature': X.columns,
                'mutual_info': mis
            }).sort_values('mutual_info', ascending=False)
            return df_mi
        except Exception as e:
            logging.error(f"MI error: {e}")
            return pd.DataFrame()

    def _plot_all(self, corr, rf_imp, mi):
        try:
            if not corr.empty:
                plt.figure(figsize=(12,10))
                sns.heatmap(corr, annot=False, cmap='RdBu', center=0)
                plt.title("Correlation Heatmap")
                plt.tight_layout()
                self._save_fig("features","correlations","correlation_heatmap.png")

            if not rf_imp.empty:
                self._plot_bar(rf_imp, 'importance', 'feature',
                               "RF Feature Importance",
                               "feature_importance.png")

            if not mi.empty:
                self._plot_bar(mi, 'mutual_info', 'feature',
                               "Mutual Info",
                               "mutual_information.png")
        except Exception as e:
            logging.error(f"Plot error: {e}")

    def _plot_bar(self, df, xcol, ycol, title, fname):
        plt.figure(figsize=(10, min(15, len(df)*0.4)))
        sns.barplot(data=df, x=xcol, y=ycol)
        plt.title(title)
        plt.tight_layout()
        self._save_fig("features","importance",fname)

    def _save_fig(self, cat, subcat, fname):
        try:
            path = self.output_manager.get_path(cat, subcat, fname)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved fig: {path}")
        except Exception as e:
            logging.error(f"Save fig error: {e}")

    def _save_group_stats(self, gname, stats):
        try:
            feats = stats.get('importances', {})
            df = pd.DataFrame(list(feats.items()), columns=['feature','importance'])
            p = self.output_manager.get_path("features","groups",f"group_{gname}.csv")
            df.to_csv(p, index=False)
        except Exception as e:
            logging.error(f"Group stats error: {e}")

    def _save_correlations(self, corr: pd.DataFrame):
        try:
            if corr.empty: return
            p = self.output_manager.get_path("features","correlations","correlation_matrix.csv")
            corr.to_csv(p)
        except Exception as e:
            logging.error(f"Save corr error: {e}")

    def _save_importances(self, df: pd.DataFrame, fname: str, score_col='importance'):
        try:
            if df.empty: return
            p = self.output_manager.get_path("features","importance",fname)
            df.to_csv(p, index=False)
        except Exception as e:
            logging.error(f"Save importances error: {e}")

    def _save_summary(self):
        try:
            p = self.output_manager.get_path("features","summaries","complete_analysis.txt")
            with open(p, 'w') as f:
                f.write("=== Feature Analysis Summary ===\n\n")
                gstat = self.results.get('group_stats', {})
                for g, s in gstat.items():
                    f.write(f"\n{g.upper()}:\n")
                    f.write(f"  feat_count: {s['feature_count']}\n")
                    f.write(f"  accuracy: {s['accuracy']:.4f}\n")
                    top3 = sorted(s['importances'].items(), key=lambda x: x[1], reverse=True)[:3]
                    f.write("  top3:\n")
                    for a,b in top3:
                        f.write(f"    {a}: {b:.4f}\n")

                rf_df = self.results.get('feature_importance', pd.DataFrame())
                if not rf_df.empty:
                    f.write("\nTop RF Feats:\n")
                    for _, row in rf_df.head(10).iterrows():
                        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

                mi_df = self.results.get('mutual_information', pd.DataFrame())
                if not mi_df.empty:
                    f.write("\nTop MI Feats:\n")
                    for _, row in mi_df.head(10).iterrows():
                        f.write(f"  {row['feature']}: {row['mutual_info']:.4f}\n")
        except Exception as e:
            logging.error(f"Summary error: {e}")
