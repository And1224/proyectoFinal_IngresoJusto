"""Criterios de aceptación del software (README, sección "Calidad y evaluación").

AC1–AC3 son técnicos y deben cumplirse. AC4 y AC5 son criterios éticos: hoy NO se cumplen,
y por eso el modelo solo se acepta como apoyo a la decisión humana, nunca como decisión
automática. Se marcan como xfail estricto: si un cambio futuro los hace cumplir, la prueba
avisa para actualizar la documentación.
"""
import pytest

from src import config
from src.data import clean, load_raw, split
from src.train import fit_logistic

A = config.ACCEPTANCE


def test_ac1_f1_minimo(results):
    assert results["metricas_prueba"]["f1"] >= A["f1_min"]


def test_ac2_roc_auc_minimo(results):
    assert results["metricas_prueba"]["roc_auc"] >= A["roc_auc_min"]


def test_ac3_supera_linea_base(results):
    gain = results["metricas_prueba"]["f1"] - results["linea_base_prueba"]["f1"]
    assert gain >= A["f1_gain_over_baseline"]
    assert results["metricas_prueba"]["accuracy"] > results["linea_base_prueba"]["accuracy"]


@pytest.mark.xfail(strict=True, reason="Criterio ético no cumplido: brecha de recall entre sexos > 0,10")
def test_ac4_igualdad_de_oportunidades_por_sexo(results):
    assert results["equidad"]["sexo"]["diferencia_igualdad_oportunidades"] <= A["eod_sex_max"]


@pytest.mark.xfail(strict=True, reason="Criterio ético no cumplido: razón de impacto dispar < 0,80")
def test_ac5_impacto_dispar_por_sexo(results):
    assert results["equidad"]["sexo"]["razon_impacto_dispar"] >= A["di_sex_min"]


def test_reproducibilidad_con_la_misma_semilla():
    X, y = clean(load_raw())
    p = split(X, y)
    (X_tr, y_tr), (X_va, _) = p["train"], p["val"]
    m1, c1 = fit_logistic(X_tr, y_tr)
    m2, c2 = fit_logistic(X_tr, y_tr)
    assert c1 == c2
    assert (m1.predict_proba(X_va[config.FEATURES]) == m2.predict_proba(X_va[config.FEATURES])).all()
