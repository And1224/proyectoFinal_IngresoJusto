"""Configuración central del proyecto: rutas, semilla, columnas y criterios de aceptación."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "adult.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

SEED = 42

# Columnas del archivo original (UCI Adult, sin encabezado)
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
TARGET = "income"
POSITIVE_LABEL = ">50K"

# Variables que usa el modelo
NUMERIC = ["age", "education_num", "hours_per_week"]
SKEWED = ["capital_gain", "capital_loss"]          # muy asimétricas: se transforman con log1p
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship"]
FEATURES = NUMERIC + SKEWED + CATEGORICAL

# Atributos sensibles: NO entran al modelo, se conservan solo para auditar equidad
SENSITIVE = ["sex", "race"]

# Columnas descartadas y motivo (se documentan en el README y en el ADR-002)
EXCLUDED = {
    "fnlwgt": "peso muestral del censo, no describe a la persona",
    "education": "redundante con education_num (misma información en orden numérico)",
    "native_country": "atributo sensible (origen nacional) y 90 % de una sola categoría",
    "sex": "atributo protegido: solo se usa para auditar equidad",
    "race": "atributo protegido: solo se usa para auditar equidad",
}

# Partición estratificada: 70 % entrenamiento, 15 % validación, 15 % prueba
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Criterios de aceptación (README, sección "Calidad y evaluación")
ACCEPTANCE = {
    "f1_min": 0.65,               # F1 de la clase positiva en prueba
    "roc_auc_min": 0.85,          # capacidad de ordenar por probabilidad
    "f1_gain_over_baseline": 0.30,
    "eod_sex_max": 0.10,          # diferencia de igualdad de oportunidades (TPR) entre sexos
    "di_sex_min": 0.80,           # razón de impacto dispar (regla de los cuatro quintos)
}
