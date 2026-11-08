"""Etapa 6 del flujo: auditoría de equidad por grupo (sexo y raza).

Métricas por grupo:
- tasa de selección: proporción de personas que el modelo marca como ingreso alto;
- TPR (recall): de quienes sí tienen ingreso alto, cuántos detecta el modelo;
- FPR: de quienes no lo tienen, cuántos marca por error.

Resúmenes de disparidad:
- diferencia de paridad demográfica (máx. − mín. de la tasa de selección);
- razón de impacto dispar (mín. / máx. de la tasa de selección; < 0,8 incumple la regla de los cuatro quintos);
- diferencia de igualdad de oportunidades (máx. − mín. de TPR);
- diferencia de probabilidades igualadas (mayor brecha entre TPR y FPR).
"""
import pandas as pd


def group_table(sensitive: pd.Series, y_true, y_pred, min_n: int = 30) -> pd.DataFrame:
    df = pd.DataFrame({"g": sensitive.values, "y": list(y_true), "p": list(y_pred)})
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d[d.y == 1], d[d.y == 0]
        sel = d[d.p == 1]
        rows.append({
            "grupo": g, "n": len(d),
            "tasa_base": d.y.mean(),
            "tasa_seleccion": d.p.mean(),
            "precision": (sel.y.mean() if len(sel) else float("nan")),
            "tpr": (pos.p.mean() if len(pos) else float("nan")),
            "fpr": (neg.p.mean() if len(neg) else float("nan")),
            "muestra_suficiente": len(pos) >= min_n,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).round(4).reset_index(drop=True)


def disparities(table: pd.DataFrame) -> dict:
    t = table[table.muestra_suficiente]
    sel, tpr, fpr = t.tasa_seleccion, t.tpr, t.fpr
    return {
        "diferencia_paridad_demografica": round(float(sel.max() - sel.min()), 4),
        "razon_impacto_dispar": round(float(sel.min() / sel.max()), 4) if sel.max() > 0 else float("nan"),
        "diferencia_igualdad_oportunidades": round(float(tpr.max() - tpr.min()), 4),
        "diferencia_probabilidades_igualadas": round(float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min())), 4),
        "grupos_evaluados": list(t.grupo),
    }
