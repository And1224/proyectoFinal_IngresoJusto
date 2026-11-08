"""Etapa 3 del flujo: preprocesamiento. Todo vive dentro de un Pipeline de scikit-learn,
así que imputación, escalado y codificación se ajustan solo con datos de entrenamiento."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(categorical=None, numeric=None, skewed=None) -> ColumnTransformer:
    categorical = categorical or config.CATEGORICAL
    numeric = numeric or config.NUMERIC
    skewed = skewed or config.SKEWED
    num = Pipeline([("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler())])
    skw = Pipeline([("imputar", SimpleImputer(strategy="median")),
                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("escalar", StandardScaler())])
    cat = Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("asim", skw, skewed), ("cat", cat, categorical)])
