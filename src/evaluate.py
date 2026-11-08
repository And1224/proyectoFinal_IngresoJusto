"""Etapa 5 del flujo: métricas de desempeño y matriz de confusión."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_score)) > 1 else 0.5,
        "vp": int(tp), "fp": int(fp), "vn": int(tn), "fn": int(fn),
    }
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def predict_with_threshold(model, X, threshold, features):
    scores = model.predict_proba(X[features])[:, 1]
    return (scores >= threshold).astype(int), scores
