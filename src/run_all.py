"""Ejecuta el flujo completo y deja los resultados en reports/ y models/.

Uso:  python -m src.run_all
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import config
from .data import clean, load_raw, profile, split
from .evaluate import classification_metrics, predict_with_threshold
from .fairness import disparities, group_table
from .train import choose_threshold, compare_candidates, fit_baseline, fit_logistic

INK, ACCENT, ACCENT2, MUTED = "#1F2933", "#2B6CB0", "#C05621", "#9AA5B1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=200, facecolor="white")
    plt.close(fig)


def run(seed: int = config.SEED, make_figures: bool = True) -> dict:
    # 1. Datos ---------------------------------------------------------------
    raw = load_raw()
    X, y = clean(raw)
    parts = split(X, y, seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = parts["train"], parts["val"], parts["test"]

    # 2. Comparación de candidatos (validación cruzada en entrenamiento) -------
    comparison = compare_candidates(X_tr, y_tr, seed)

    # 3. Modelo elegido: regresión logística + umbral elegido en validación ----
    model, best_c = fit_logistic(X_tr, y_tr, seed=seed)
    threshold, curve = choose_threshold(model, X_va, y_va)
    y_pred, y_score = predict_with_threshold(model, X_te, threshold, config.FEATURES)
    test_metrics = classification_metrics(y_te, y_pred, y_score)
    val_pred, val_score = predict_with_threshold(model, X_va, threshold, config.FEATURES)
    val_metrics = classification_metrics(y_va, val_pred, val_score)
    default_pred = (y_score >= 0.5).astype(int)
    test_metrics_05 = classification_metrics(y_te, default_pred, y_score)

    baseline = fit_baseline(X_tr, y_tr)
    base_pred = baseline.predict(X_te[config.FEATURES])
    base_metrics = classification_metrics(y_te, base_pred, baseline.predict_proba(X_te[config.FEATURES])[:, 1])

    # 4. Auditoría de equidad --------------------------------------------------
    t_sex = group_table(X_te["sex"], y_te, y_pred)
    t_race = group_table(X_te["race"], y_te, y_pred)
    d_sex, d_race = disparities(t_sex), disparities(t_race)

    # Experimento: ¿qué pasa si el modelo SÍ usa sexo y raza?
    feats_s = config.FEATURES + config.SENSITIVE
    model_s, _ = fit_logistic(X_tr, y_tr, features=feats_s, seed=seed,
                              categorical=config.CATEGORICAL + config.SENSITIVE)
    thr_s, _ = choose_threshold(model_s, X_va, y_va, features=feats_s)
    pred_s, score_s = predict_with_threshold(model_s, X_te, thr_s, feats_s)
    metrics_s = classification_metrics(y_te, pred_s, score_s)
    d_sex_s = disparities(group_table(X_te["sex"], y_te, pred_s))

    # Variables proxy: qué tanto revela 'relationship' el sexo de la persona
    ct = pd.crosstab(X["relationship"], X["sex"], normalize="index").round(4)
    proxies = {r: ct.loc[r].to_dict() for r in ["Husband", "Wife"]}

    # 5. Criterios de aceptación ------------------------------------------------
    A = config.ACCEPTANCE
    acceptance = {
        "AC1_f1": {"umbral": A["f1_min"], "valor": test_metrics["f1"], "cumple": test_metrics["f1"] >= A["f1_min"]},
        "AC2_roc_auc": {"umbral": A["roc_auc_min"], "valor": test_metrics["roc_auc"],
                        "cumple": test_metrics["roc_auc"] >= A["roc_auc_min"]},
        "AC3_supera_linea_base": {"umbral": A["f1_gain_over_baseline"],
                                  "valor": round(test_metrics["f1"] - base_metrics["f1"], 4),
                                  "cumple": test_metrics["f1"] - base_metrics["f1"] >= A["f1_gain_over_baseline"]},
        "AC4_igualdad_oportunidades_sexo": {"umbral": A["eod_sex_max"],
                                            "valor": d_sex["diferencia_igualdad_oportunidades"],
                                            "cumple": d_sex["diferencia_igualdad_oportunidades"] <= A["eod_sex_max"]},
        "AC5_impacto_dispar_sexo": {"umbral": A["di_sex_min"], "valor": d_sex["razon_impacto_dispar"],
                                    "cumple": d_sex["razon_impacto_dispar"] >= A["di_sex_min"]},
    }

    # Coeficientes (interpretabilidad)
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["modelo"].coef_[0], index=names)
    coefs.index = [n.split("__", 1)[1] for n in coefs.index]
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:12]

    results = {
        "dataset": profile(X, y),
        "duplicados_eliminados": int(len(raw) - len(X)),
        "particion": {k: int(len(v[1])) for k, v in parts.items()},
        "tasa_positiva_particion": {k: round(float(v[1].mean()), 4) for k, v in parts.items()},
        "modelo": {"tipo": "LogisticRegression", "C": best_c, "umbral": threshold,
                   "variables": config.FEATURES, "excluidas": config.EXCLUDED},
        "metricas_validacion": val_metrics,
        "metricas_prueba": test_metrics,
        "metricas_prueba_umbral_05": test_metrics_05,
        "linea_base_prueba": base_metrics,
        "comparacion_cv": comparison.to_dict(orient="records"),
        "equidad": {"sexo": d_sex, "raza": d_race,
                    "sexo_con_atributos_sensibles": d_sex_s,
                    "metricas_con_atributos_sensibles": metrics_s,
                    "proxy_relationship_sexo": proxies},
        "aceptacion": acceptance,
        "coeficientes_top": top.round(4).to_dict(),
    }

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=bool))
    comparison.to_csv(config.REPORTS / "comparacion_modelos.csv", index=False)
    t_sex.to_csv(config.REPORTS / "equidad_sexo.csv", index=False)
    t_race.to_csv(config.REPORTS / "equidad_raza.csv", index=False)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "umbral": threshold, "variables": config.FEATURES},
                config.MODELS / "modelo_logistico.joblib")

    if make_figures:
        _figures(X, y, curve, threshold, test_metrics, y_te, y_score, top, t_sex, t_race, comparison)
    return results


def _figures(X, y, curve, threshold, m, y_te, y_score, top, t_sex, t_race, comparison):
    # Distribución de la clase, total y por sexo
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    counts = y.value_counts().sort_index()
    ax[0].bar(["≤ 50K", "> 50K"], counts.values, color=[MUTED, ACCENT])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}".replace(",", " ") + f"\n({v / counts.sum():.0%})", ha="center", va="bottom", fontsize=9)
    ax[0].set_title("Clases del objetivo"); ax[0].set_ylabel("Personas"); ax[0].set_ylim(0, counts.max() * 1.25)
    rate = X.assign(y=y).groupby("sex")["y"].mean().rename({"Female": "Mujeres", "Male": "Hombres"})
    ax[1].bar(rate.index, rate.values, color=[ACCENT2, ACCENT])
    for i, v in enumerate(rate.values):
        ax[1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    ax[1].set_title("Proporción con ingreso > 50K por sexo"); ax[1].set_ylim(0, 0.4)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, "01_clases.png")

    # F1 en validación según el umbral
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(curve.umbral, curve.f1, color=ACCENT, lw=2)
    ax.axvline(threshold, color=ACCENT2, ls="--", lw=1.2)
    ax.text(threshold + 0.01, curve.f1.min(), f"umbral elegido = {threshold:.2f}", color=ACCENT2, fontsize=9)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Umbral de decisión"); ax.set_ylabel("F1 en validación"); ax.set_title("Elección del umbral")
    _save(fig, "02_umbral.png")

    # Matriz de confusión
    cm = np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]])
    labels = np.array([["VN", "FP"], ["FN", "VP"]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i, j]}\n{cm[i, j]:,}".replace(",", " ") + f"\n{cm[i, j] / cm.sum():.1%}",
                    ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Predicho ≤ 50K", "Predicho > 50K"]); ax.set_yticks([0, 1], ["Real ≤ 50K", "Real > 50K"])
    ax.set_title("Matriz de confusión (prueba)")
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "03_matriz_confusion.png")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_te, y_score)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"Regresión logística (AUC = {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", label="Línea base (AUC = 0,500)")
    ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC (prueba)"); ax.legend(loc="lower right", fontsize=8, frameon=False)
    _save(fig, "04_roc.png")

    # Coeficientes
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    t = top.iloc[::-1]
    ax.barh(t.index, t.values, color=[ACCENT if v > 0 else ACCENT2 for v in t.values])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Coeficiente (log-odds; variables escaladas)")
    ax.set_title("Variables con más peso en la decisión")
    _save(fig, "05_coeficientes.png")

    # Equidad por grupo
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for a, (tab, title) in zip(ax, [(t_sex, "Por sexo"), (t_race[t_race.muestra_suficiente], "Por raza (grupos con ≥ 30 positivos)")]):
        xs = np.arange(len(tab)); w = 0.38
        a.bar(xs - w / 2, tab.tasa_seleccion, w, color=MUTED, label="Tasa de selección")
        a.bar(xs + w / 2, tab.tpr, w, color=ACCENT, label="Tasa de verdaderos positivos (recall)")
        a.set_xticks(xs, [str(g).replace("Female", "Mujeres").replace("Male", "Hombres")
                          .replace("Asian-Pac-Islander", "Asiática/Pac.").replace("White", "Blanca")
                          .replace("Black", "Negra") for g in tab.grupo], fontsize=9)
        a.set_title(title); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        a.set_ylim(0, 1)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(config.FIGURES / "06_equidad.png", dpi=200, facecolor="white")
    plt.close(fig)

    # Comparación de candidatos
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    c = comparison.set_index("modelo")[["f1", "roc_auc"]]
    xs = np.arange(len(c)); w = 0.38
    ax.bar(xs - w / 2, c.f1, w, color=ACCENT, label="F1")
    ax.bar(xs + w / 2, c.roc_auc, w, color=MUTED, label="ROC-AUC")
    ax.set_xticks(xs, [n.replace(" (", "\n(") for n in c.index], fontsize=8.5)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8); ax.set_title("Validación cruzada de 5 pliegues (entrenamiento)")
    _save(fig, "07_comparacion_modelos.png")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ["particion", "modelo", "metricas_prueba", "linea_base_prueba", "aceptacion"]},
                     indent=2, ensure_ascii=False, default=str))
