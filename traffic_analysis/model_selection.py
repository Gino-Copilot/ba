from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# Definiere ein Dictionary mit Modellen und deren Konfigurationen
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, random_state=42, eval_metric='logloss'
    ),
    "SVM (Linear)": SVC(kernel='linear', probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Stochastic Gradient Descent": SGDClassifier(max_iter=1000, tol=1e-3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}

def get_model(name):
    """Gibt das Modell zurück, das zum angegebenen Namen passt"""
    return MODELS.get(name, None)
