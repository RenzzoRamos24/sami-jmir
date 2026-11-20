"""Análisis v3 (cohorte de septiembre), paso 1: LOOCV anidado de 6 combinaciones.

Conjuntos de variables fijados de antemano × modelo:

  (a) 3 variables : phq_total, gad_total, ítem 9 >= 1
  (b) 16 ítems crudos
  (c) 29 variables (el conjunto actual, con SelectKBest dentro del fold)
  × SVM / regresión logística (StandardScaler, class_weight=balanced)

Protocolo idéntico al de `train_svm_clinico.py`: misma grilla de C/gamma/kernel
(y k en (c)), misma semilla (20261013), CV interna estratificada de 5
particiones, puntuación f1_macro. Todo se importa de ese script para que no
pueda divergir.

Extras en la misma pasada:
  - SVM (c) con calibración de Platt anidada (punto 2 del encargo);
  - ablación 6(b): las 29 + 2 variables de BETO (punto 7).

Salida: reports/loocv_comparacion_modelos.csv (predicción caso por caso, solo
códigos de caso y números; sin texto libre) y reports/v3_loocv_params.json.

Uso:  venv/bin/python scripts/analisis_v3_loocv.py --casos data/clinico/casos_etiquetados.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_svm_clinico as T  # noqa: E402
from app.services.svm_features import construir_features  # noqa: E402

warnings.filterwarnings("ignore")
SEMILLA = T.SEMILLA
C_GRID = T.GRID_SVC["svc__C"]
BETO_COLS = ["beto_max_depresion", "beto_n_detectadas"]


def f3(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([X[:, :9].sum(1), X[:, 9:16].sum(1),
                            (X[:, 8] >= 1).astype(float)])


def f31(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([construir_features(X[:, :16]), X[:, 16:]])


def svc():
    return SVC(class_weight="balanced", random_state=SEMILLA)


def lr():
    return LogisticRegression(class_weight="balanced", max_iter=5000,
                              random_state=SEMILLA)


def pipe(modelo: str, conjunto: str) -> tuple[Pipeline, dict]:
    est, nom = (svc(), "svc") if modelo == "SVM" else (lr(), "lr")
    grid = (dict(T.GRID_SVC) if modelo == "SVM" else {"lr__C": C_GRID})
    pasos = {"a": [("feat", FunctionTransformer(f3)), ("sc", StandardScaler())],
             "b": [("sc", StandardScaler())],
             "c": [("feat", FunctionTransformer(construir_features)),
                   ("sc", StandardScaler()), ("sel", SelectKBest(f_classif))],
             "d": [("feat", FunctionTransformer(f31)),
                   ("sc", StandardScaler()), ("sel", SelectKBest(f_classif))],
             }[conjunto]
    if conjunto in ("c", "d"):
        grid = {**grid, **T.GRID_K}
    return Pipeline(pasos + [(nom, est)]), grid


def vuelta(i, X, y, modelo, conjunto, platt):
    warnings.filterwarnings("ignore")
    tr = np.delete(np.arange(len(y)), i)
    p, grid = pipe(modelo, conjunto)
    gs = T.buscar(lambda: p, grid)
    gs.fit(X[tr], y[tr])
    xi = X[i:i + 1]
    out = {"pred": int(gs.predict(xi)[0]),
           "dec": float(gs.decision_function(xi)[0]),
           "params": {k: (v if isinstance(v, str) else float(v))
                      for k, v in gs.best_params_.items()},
           "f1_cv": float(gs.best_score_)}
    if platt:
        # Platt dentro del fold: el calibrador se ajusta con márgenes
        # fuera-de-muestra de la CV interna del entrenamiento, nunca con el caso i.
        oof = cross_val_predict(clone(gs.best_estimator_), X[tr], y[tr],
                                cv=T.cv_interna(), method="decision_function")
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(oof[:, None], y[tr])
        out["prob_platt"] = float(cal.predict_proba([[out["dec"]]])[0, 1])
    return out


def loocv(X, y, modelo, conjunto, platt=False):
    return Parallel(n_jobs=-1)(delayed(vuelta)(i, X, y, modelo, conjunto, platt)
                               for i in range(len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", type=Path, required=True)
    ap.add_argument("--beto", type=Path,
                    default=ROOT / "reports" / "beto_features_102.csv")
    a = ap.parse_args()

    p, X, y, reglas = T.cargar(a.casos)
    beto = pd.read_csv(a.beto).set_index("aplicacion_id")
    Xb = np.hstack([X, beto.loc[p.aplicacion_id, BETO_COLS].to_numpy(float)])

    df = pd.DataFrame({
        "aplicacion_id": p.aplicacion_id.astype(int),
        "caso": "CASO-" + p.aplicacion_id.astype(int).astype(str),
        "y_riesgo_medio_o_mas": y,
        "requiere_derivacion": p.requiere_derivacion.astype(int),
        "pred_reglas": reglas,
        "phq_total": X[:, :9].sum(1).astype(int),
        "gad_total": X[:, 9:].sum(1).astype(int),
        "item9": X[:, 8].astype(int),
    })
    params = {}
    jobs = [(m, c) for c in "abc" for m in ("SVM", "LR")] + [("SVM", "d")]
    for modelo, conj in jobs:
        nombre = f"{modelo}_{conj}" if conj != "d" else "SVM_c_mas_BETO"
        print(f"[{nombre}] LOOCV…", flush=True)
        r = loocv(Xb if conj == "d" else X, y, modelo, conj,
                  platt=(nombre == "SVM_c"))
        df[f"pred_{nombre}"] = [s["pred"] for s in r]
        df[f"margen_{nombre}"] = [round(s["dec"], 6) for s in r]
        if nombre == "SVM_c":
            df["prob_platt_SVM_c"] = [round(s["prob_platt"], 6) for s in r]
        params[nombre] = [{"aplicacion_id": int(i), **s}
                          for i, s in zip(df.aplicacion_id, r)]
        print(f"   exactitud {(df[f'pred_{nombre}'] == y).mean():.4f}", flush=True)

    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports" / "loocv_comparacion_modelos.csv", index=False)
    (ROOT / "reports" / "v3_loocv_params.json").write_text(
        json.dumps(params, ensure_ascii=False))
    print("listo")


if __name__ == "__main__":
    main()
