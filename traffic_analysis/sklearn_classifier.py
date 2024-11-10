import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

class ScikitLearnTrafficClassifier:
    def __init__(self):
        self.classifier = None
        self.scaler = None
        self.feature_importance = None

    def train(self, df):
        X = df.drop('label', axis=1)
        y = df['label']
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print("\nKlassifikationsbericht:", classification_report(y_test, y_pred))
        print("\nKonfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)

    def save_results(self, df):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = "nfstream_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.feature_importance.to_csv(f'{results_dir}/feature_importance_{timestamp}.csv', index=False)
        df.to_csv(f'{results_dir}/full_dataset_{timestamp}.csv', index=False)
