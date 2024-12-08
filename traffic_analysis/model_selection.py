from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Dict, Any


class ModelSelector:
    """
    Klasse zur Verwaltung und Konfiguration von Machine Learning Modellen
    """

    def __init__(self):
        self.models = self._initialize_models()
        self.param_grids = self._initialize_param_grids()

    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialisiert die verfügbaren Modelle mit Standardkonfigurationen

        Returns:
            Dict[str, Any]: Dictionary mit Modellnamen und initialisierten Modellen
        """
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            "SVM": SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            "Logistic Regression": LogisticRegression(
                random_state=42,
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),
            "AdaBoost": AdaBoostClassifier(
                random_state=42
            ),
            "KNN": KNeighborsClassifier(
                n_jobs=-1
            ),
            "SGD": SGDClassifier(
                random_state=42,
                n_jobs=-1
            )
        }

    def _initialize_param_grids(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialisiert die Parameter-Grids für Grid Search

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mit Modellnamen und zugehörigen Parameter-Grids
        """
        return {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            "Decision Tree": {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            "AdaBoost": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            },
            "KNN": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            "SGD": {
                'loss': ['hinge', 'log'],
                'penalty': ['l1', 'l2'],
                'alpha': [0.0001, 0.001]
            }
        }

    def get_all_models(self) -> Dict[str, Any]:
        """
        Gibt alle verfügbaren Modelle zurück

        Returns:
            Dict[str, Any]: Dictionary mit allen Modellen
        """
        return self.models

    def get_model(self, name: str) -> Any:
        """
        Gibt ein spezifisches Modell zurück

        Args:
            name: Name des gewünschten Modells

        Returns:
            Any: Das angeforderte Modell oder None wenn nicht gefunden
        """
        return self.models.get(name)

    def get_param_grid(self, name: str) -> Dict[str, Any]:
        """
        Gibt das Parameter-Grid für ein spezifisches Modell zurück

        Args:
            name: Name des Modells

        Returns:
            Dict[str, Any]: Parameter-Grid für Grid Search
        """
        return self.param_grids.get(name, {})

    def get_available_models(self) -> list:
        """
        Gibt eine Liste aller verfügbaren Modellnamen zurück

        Returns:
            list: Liste der Modellnamen
        """
        return list(self.models.keys())

    def is_model_available(self, name: str) -> bool:
        """
        Prüft ob ein Modell verfügbar ist

        Args:
            name: Name des zu prüfenden Modells

        Returns:
            bool: True wenn verfügbar, sonst False
        """
        return name in self.models

    def __str__(self) -> str:
        """
        String-Repräsentation der Klasse

        Returns:
            str: Übersicht über verfügbare Modelle
        """
        return f"Available models: {', '.join(self.get_available_models())}"


# Singleton-Instanz für globale Nutzung
model_selector = ModelSelector()

if __name__ == "__main__":
    # Beispielnutzung
    print("Available models:")
    for name in model_selector.get_available_models():
        print(f"- {name}")
        model = model_selector.get_model(name)
        param_grid = model_selector.get_param_grid(name)
        print(f"  Parameters for grid search: {list(param_grid.keys())}")