"""Etapas 1 y 2 del flujo: carga, limpieza y partición de los datos."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=config.DATA_RAW) -> pd.DataFrame:
    """Lee el CSV original de UCI Adult. Los valores faltantes vienen como '?'."""
    return pd.read_csv(path, header=None, names=config.COLUMNS,
                       skipinitialspace=True, na_values="?")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Elimina duplicados exactos y separa la variable objetivo (1 = ingreso > 50K)."""
    df = df.drop_duplicates().reset_index(drop=True)
    y = (df[config.TARGET] == config.POSITIVE_LABEL).astype(int).rename("y")
    X = df.drop(columns=[config.TARGET])
    return X, y


def split(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED) -> dict:
    """Partición estratificada 70/15/15. Los duplicados ya se quitaron, así que ninguna
    fila de prueba puede repetirse en entrenamiento (evita fuga de información)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    val_rel = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=seed)
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}


def profile(X: pd.DataFrame, y: pd.Series) -> dict:
    """Resumen descriptivo para el README y el informe."""
    data = X.assign(y=y)
    return {
        "filas": int(len(X)),
        "columnas": int(X.shape[1]),
        "tasa_positiva": round(float(y.mean()), 4),
        "faltantes": {c: int(v) for c, v in X.isna().sum().items() if v > 0},
        "tasa_positiva_por_sexo": data.groupby("sex")["y"].mean().round(4).to_dict(),
        "tasa_positiva_por_raza": data.groupby("race")["y"].mean().round(4).to_dict(),
        "proporcion_por_sexo": X["sex"].value_counts(normalize=True).round(4).to_dict(),
        "proporcion_por_raza": X["race"].value_counts(normalize=True).round(4).to_dict(),
    }
