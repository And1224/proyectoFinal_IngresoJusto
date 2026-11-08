"""Etapa 4 del flujo: comparación de candidatos, ajuste del modelo y elección del umbral."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import build_preprocessor


def make_pipeline(estimator, **prep_kwargs) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(**prep_kwargs)), ("modelo", estimator)])


def candidates(seed: int = config.SEED) -> dict:
    return {
        "Línea base (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Regresión logística": LogisticRegression(max_iter=2000, random_state=seed),
        "Árbol de decisión (profundidad 8)": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "KNN (k = 25)": KNeighborsClassifier(n_neighbors=25),
    }


def compare_candidates(X_train, y_train, seed: int = config.SEED) -> pd.DataFrame:
    """Validación cruzada estratificada de 5 pliegues, solo sobre entrenamiento."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, est in candidates(seed).items():
        res = cross_validate(make_pipeline(est), X_train[config.FEATURES], y_train, cv=cv,
                             scoring=["accuracy", "f1", "roc_auc"], n_jobs=1)
        rows.append({"modelo": name,
                     "accuracy": res["test_accuracy"].mean(),
                     "f1": res["test_f1"].mean(),
                     "roc_auc": res["test_roc_auc"].mean(),
                     "f1_desv": res["test_f1"].std()})
    return pd.DataFrame(rows).round(4)


def fit_logistic(X_train, y_train, features=None, seed: int = config.SEED, **prep_kwargs):
    """Ajusta la regresión logística y elige C por validación cruzada (ROC-AUC)."""
    features = features or config.FEATURES
    pipe = make_pipeline(LogisticRegression(max_iter=2000, random_state=seed), **prep_kwargs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, {"modelo__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc", cv=cv)
    search.fit(X_train[features], y_train)
    return search.best_estimator_, search.best_params_["modelo__C"]


def choose_threshold(model, X_val, y_val, features=None) -> tuple[float, pd.DataFrame]:
    """Umbral de decisión que maximiza F1 en validación (nunca en prueba)."""
    features = features or config.FEATURES
    scores = model.predict_proba(X_val[features])[:, 1]
    grid = np.round(np.arange(0.10, 0.91, 0.01), 2)
    f1s = [f1_score(y_val, (scores >= t).astype(int)) for t in grid]
    curve = pd.DataFrame({"umbral": grid, "f1": f1s})
    return float(grid[int(np.argmax(f1s))]), curve


def fit_baseline(X_train, y_train):
    model = make_pipeline(DummyClassifier(strategy="most_frequent"))
    return model.fit(X_train[config.FEATURES], y_train)
