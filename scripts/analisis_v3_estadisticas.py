"""Análisis v3, paso 3: métricas con IC, McNemar, calibración, coeficientes, ROC.

Lee `reports/loocv_comparacion_modelos.csv` (paso 1) y el CSV de casos; no toca
producción. Escribe:
    reports/v3_estadisticas.json     todo, legible por máquina
    reports/v3_tablas.md             tablas en markdown (las usa el informe)
    reports/roc_svm_phq.png          figura 300 dpi
    reports/roc_puntos.csv           puntos de las curvas

Bootstrap: 10 000 remuestreos de casos, semilla 20261013, MISMOS índices para
todos los modelos (así los IC son comparables). IC por percentiles.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_svm_clinico as T  # noqa: E402
from app.services.svm_features import NOMBRES_FEATURES, construir_features  # noqa: E402

warnings.filterwarnings("ignore")
SEMILLA = 20261013
NB = 10000
R = ROOT / "reports"


# ── métricas vectorizadas ────────────────────────────────────────────────────
def conteos(y, p):
    return (int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()),
            int(((p == 0) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum()))


def desde_conteos(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    d = lambda a, b: a / b if b else np.nan  # noqa: E731
    f1_1 = d(2 * tp, 2 * tp + fp + fn)
    f1_0 = d(2 * tn, 2 * tn + fn + fp)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return {"exactitud": po, "f1_macro": (f1_1 + f1_0) / 2,
            "kappa": d(po - pe, 1 - pe), "sensibilidad": d(tp, tp + fn),
            "especificidad": d(tn, tn + fp), "vpp": d(tp, tp + fp),
            "vpn": d(tn, tn + fn)}


def auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def todas(y, pred, score=None):
    m = desde_conteos(*conteos(y, pred))
    m["auc"] = auc(y, score if score is not None else pred)
    return m


def boot(y, pred, score, idx):
    out = {k: [] for k in ["exactitud", "f1_macro", "kappa", "auc",
                           "sensibilidad", "especificidad", "vpp", "vpn"]}
    for i in idx:
        m = todas(y[i], pred[i], None if score is None else score[i])
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in out.items()}


def mcnemar(y, a, b):
    ok_a, ok_b = a == y, b == y
    n10 = int((ok_a & ~ok_b).sum())     # a acierta, b falla
    n01 = int((~ok_a & ok_b).sum())     # a falla, b acierta
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"a_acierta_b_falla": n10, "a_falla_b_acierta": n01, "p_exacto": float(p)}


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(m, k, d=3):
    return f"{f(m[k], d)} [{f(m['ic'][k][0], d)}–{f(m['ic'][k][1], d)}]"


def p_fmt(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# ── calibración ──────────────────────────────────────────────────────────────
def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    m = LogisticRegression(C=1e9, max_iter=10000).fit(z[:, None], y)
    pend, inter = float(m.coef_[0, 0]), float(m.intercept_[0])
    # calibración en lo grande: intercepto con pendiente fija en 1 (offset)
    citl = float(np.log(y.mean() / (1 - y.mean())) - 0)  # valor inicial
    for _ in range(50):                                   # Newton sobre a
        q = 1 / (1 + np.exp(-(z + citl)))
        g, h = (y - q).sum(), (q * (1 - q)).sum()
        citl += g / h
    return {"brier": float(((p - y) ** 2).mean()), "pendiente": pend,
            "intercepto": inter, "citl": citl}


def main():
    df = pd.read_csv(R / "loocv_comparacion_modelos.csv")
    casos = pd.read_csv(ROOT / "data/clinico/casos_etiquetados.csv")
    assert (casos.aplicacion_id.values == df.aplicacion_id.values).all()
    y = df.y_riesgo_medio_o_mas.to_numpy()
    n = len(y)
    rng = np.random.default_rng(SEMILLA)
    idx = [rng.integers(0, n, n) for _ in range(NB)]

    # ── control: reproducimos el LOOCV vigente ──────────────────────────────
    ant = pd.read_csv(R / "loocv_caso_por_caso.csv")
    rep = {"pred_identicas": bool((ant.prediccion.values == df.pred_SVM_c.values).all()),
           "max_dif_margen": float(np.abs(ant.margen.values - df.margen_SVM_c.values).max())}

    # ── 1. seis combinaciones + referencias ─────────────────────────────────
    modelos = ["SVM_a", "LR_a", "SVM_b", "LR_b", "SVM_c", "LR_c"]
    res = {}
    for nom in ["reglas"] + modelos + ["SVM_c_mas_BETO"]:
        pred = df[f"pred_{nom}"].to_numpy()
        sc = None if nom == "reglas" else df[f"margen_{nom}"].to_numpy()
        m = todas(y, pred, sc)
        m["conteos"] = dict(zip(["tp", "fp", "tn", "fn"], conteos(y, pred)))
        m["ic"] = boot(y, pred, sc, idx)
        res[nom] = m
    mc = {}
    for nom in modelos + ["SVM_c_mas_BETO"]:
        mc[nom] = {"vs_reglas": mcnemar(y, df[f"pred_{nom}"].values, df.pred_reglas.values),
                   "vs_SVM_c": mcnemar(y, df[f"pred_{nom}"].values, df.pred_SVM_c.values)}

    # ── 2. calibración de Platt anidada ─────────────────────────────────────
    p = df.prob_platt_SVM_c.to_numpy()
    cal = calib(y, p)
    cal_ic = {k: [] for k in cal}
    for i in idx:
        try:
            c = calib(y[i], p[i])
            for k in cal:
                cal_ic[k].append(c[k])
        except Exception:
            pass
    cal["ic"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                 for k, v in cal_ic.items()}
    cal["brier_solo_prevalencia"] = float(y.mean() * (1 - y.mean()))
    cal["n_remuestreos_validos"] = len(cal_ic["brier"])
    orden = np.argsort(p, kind="stable")
    quint = []
    for q, g in enumerate(np.array_split(orden, 5), 1):
        quint.append({"quintil": q, "n": int(len(g)), "p_min": float(p[g].min()),
                      "p_max": float(p[g].max()), "p_media": float(p[g].mean()),
                      "prevalencia_observada": float(y[g].mean()),
                      "n_positivos": int(y[g].sum())})
    # coeficientes del modelo final (los 102 casos, misma búsqueda del script de entrenamiento)
    X = casos[T.ITEMS].to_numpy(float)
    final = T.buscar(T.pipeline_29, {**T.GRID_SVC, **T.GRID_K}).fit(X, y)
    pl = final.best_estimator_
    sel = pl.named_steps["sel"].get_support()
    nombres = [NOMBRES_FEATURES[j] for j in np.where(sel)[0]]
    svc = pl.named_steps["svc"]
    sc_ = pl.named_steps["sc"]
    coefs = None
    if svc.kernel == "linear":
        w = svc.coef_[0]
        escala = sc_.scale_[sel]
        raw = w / escala
        efect = dict(zip(nombres, raw))
        # total = phq_total + gad_total (colineal exacta): su peso se reparte
        if "total" in efect:
            for k in ("phq_total", "gad_total"):
                if k in efect:
                    efect[k] += efect["total"]
        coefs = {"hiperparametros": {k: (v if isinstance(v, str) else float(v))
                                     for k, v in final.best_params_.items()},
                 "intercepto": float(svc.intercept_[0]),
                 "estandarizados": [{"variable": a, "coef": float(b),
                                     "signo": "+" if b > 0 else "−"}
                                    for a, b in zip(nombres, w)],
                 "peso_efectivo_por_punto": {k: float(v) for k, v in efect.items()
                                             if k != "total"}}
    # ── 3. segundo criterio: requiere_derivacion ────────────────────────────
    yd = df.requiere_derivacion.to_numpy()
    pred_svm, mar_svm = df.pred_SVM_c.to_numpy(), df.margen_SVM_c.to_numpy()
    der = {}
    for nom, pr, sc in (("SVM_c", pred_svm, mar_svm), ("reglas", df.pred_reglas.to_numpy(), None)):
        m = todas(yd, pr, sc)
        m["conteos"] = dict(zip(["tp", "fp", "tn", "fn"], conteos(yd, pr)))
        m["ic"] = boot(yd, pr, sc, idx)
        der[nom] = m
    cruce = pd.crosstab(casos.riesgo_clinico, casos.requiere_derivacion)
    der["cruce_nivel_vs_derivacion"] = {a: {str(c): int(v) for c, v in r.items()}
                                        for a, r in cruce.iterrows()}
    der["kappa_entre_criterios"] = float(desde_conteos(*conteos(yd, y))["kappa"])
    der["n_derivacion"] = int(yd.sum())

    # ── 4. ROC ──────────────────────────────────────────────────────────────
    phq = df.phq_total.to_numpy(float)
    roc = {}
    for nom, s in (("SVM", mar_svm), ("PHQ-A total", phq)):
        a = auc(y, s)
        b = [auc(y[i], s[i]) for i in idx]
        roc[nom] = {"auc": float(a), "ic": [float(np.percentile(b, 2.5)),
                                            float(np.percentile(b, 97.5))]}
    dif = [auc(y[i], mar_svm[i]) - auc(y[i], phq[i]) for i in idx]
    roc["diferencia_SVM_menos_PHQ"] = {
        "delta": roc["SVM"]["auc"] - roc["PHQ-A total"]["auc"],
        "ic": [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))]}
    rg = res["reglas"]
    roc["reglas_punto"] = {"sensibilidad": rg["sensibilidad"],
                           "fpr": 1 - rg["especificidad"]}
    filas = []
    for nom, s in (("SVM (margen LOOCV)", mar_svm), ("PHQ-A total", phq)):
        fpr, tpr, thr = roc_curve(y, s)
        filas += [{"curva": nom, "umbral": (None if np.isinf(t) else float(t)),
                   "fpr": float(a), "tpr": float(b)} for a, b, t in zip(fpr, tpr, thr)]
    filas.append({"curva": "Motor de reglas (punto)", "umbral": None,
                  "fpr": roc["reglas_punto"]["fpr"], "tpr": roc["reglas_punto"]["sensibilidad"]})
    pd.DataFrame(filas).to_csv(R / "roc_puntos.csv", index=False)
    figura(roc)

    # ── 7. ablación BETO: comparación con la corrida previa ─────────────────
    previa = pd.read_csv(R / "ablacion_b_predicciones.csv")
    abl = {"filas_previa": int(len(previa)),
           "mismos_casos_y_etiquetas": bool((previa.aplicacion_id.values == df.aplicacion_id.values).all()
                                           and (previa.y_real.values == y).all()),
           "pred_previa_igual_a_esta_corrida": bool((previa.pred_b.values == df.pred_SVM_c_mas_BETO.values).all()),
           "pred_actual_previa_igual_SVM_c": bool((previa.pred_actual.values == df.pred_SVM_c.values).all()),
           "casos_que_cambian": [
               {"caso": f"CASO-{a}", "y": int(yy), "SVM_c": int(pa), "con_BETO": int(pb)}
               for a, yy, pa, pb in zip(df.aplicacion_id, y, df.pred_SVM_c, df.pred_SVM_c_mas_BETO)
               if pa != pb]}

    out = {"n": n, "n_alerta": int(y.sum()), "reproduccion_loocv_vigente": rep,
           "modelos": res, "mcnemar": mc, "calibracion": cal, "quintiles": quint,
           "coeficientes": coefs, "derivacion": der, "roc": roc, "ablacion": abl}
    (R / "v3_estadisticas.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    tablas(out)
    print("listo")


def figura(roc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(R / "roc_puntos.csv")
    az, na, ne = "#2a78d6", "#eb6834", "#0b0b0b"
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), lw=1, color="#8a8984", zorder=1)
    for nom, col, key in (("SVM (margen LOOCV)", az, "SVM"),
                          ("PHQ-A total", na, "PHQ-A total")):
        s = d[d.curva == nom]
        r = roc[key]
        ax.step(s.fpr, s.tpr, where="post", lw=2, color=col, zorder=3,
                label=f"{nom}: AUC {r['auc']:.3f} (IC 95 % {r['ic'][0]:.3f}–{r['ic'][1]:.3f})")
    rp = roc["reglas_punto"]
    ax.plot(rp["fpr"], rp["sensibilidad"], marker="D", ms=9, color=ne,
            mec="white", mew=2, ls="none", zorder=4,
            label=f"Motor de reglas: sens. {rp['sensibilidad']:.3f}, 1 − espec. {rp['fpr']:.3f}")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("1 − especificidad", color="#52514e")
    ax.set_ylabel("Sensibilidad", color="#52514e")
    ax.grid(color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=1,
              edgecolor="#e6e5e1")
    ax.set_title("Criterio: riesgo clínico ≥ MEDIO (n = 102)", fontsize=10,
                 color="#0b0b0b", loc="left")
    fig.tight_layout()
    fig.savefig(R / "roc_svm_phq.png", dpi=300, facecolor="white")


def tablas(o):
    L = []
    nom = {"reglas": "Motor de reglas", "SVM_a": "SVM (a) 3 var.", "LR_a": "Reg. logística (a) 3 var.",
           "SVM_b": "SVM (b) 16 crudas", "LR_b": "Reg. logística (b) 16 crudas",
           "SVM_c": "SVM (c) 29 var. [actual]", "LR_c": "Reg. logística (c) 29 var.",
           "SVM_c_mas_BETO": "SVM 29 + 2 BETO"}
    L += ["## T1", "", "| Modelo | Exactitud | F1-macro | κ | AUC | Sensibilidad | Especificidad | VPP | VPN |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in o["modelos"].items():
        L.append(f"| {nom[k]} | " + " | ".join(ci(v, c) for c in
                 ["exactitud", "f1_macro", "kappa", "auc", "sensibilidad", "especificidad", "vpp", "vpn"]) + " |")
    L += ["", "## T1b", "", "| Modelo | VP | FP | VN | FN |", "|---|---|---|---|---|"]
    for k, v in o["modelos"].items():
        c = v["conteos"]
        L.append(f"| {nom[k]} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} |")
    L += ["", "## T1c", "", "| Modelo | vs reglas: gana / pierde | p exacto | vs SVM (c): gana / pierde | p exacto |",
          "|---|---|---|---|---|"]
    for k, v in o["mcnemar"].items():
        a, b = v["vs_reglas"], v["vs_SVM_c"]
        L.append(f"| {nom[k]} | {a['a_acierta_b_falla']} / {a['a_falla_b_acierta']} | {p_fmt(a['p_exacto'])} | "
                 f"{b['a_acierta_b_falla']} / {b['a_falla_b_acierta']} | {p_fmt(b['p_exacto'])} |")
    c = o["calibracion"]
    L += ["", "## T2", "", "| Medida | Valor | IC 95 % (bootstrap) |", "|---|---|---|"]
    for k, lab in (("brier", "Brier"), ("pendiente", "Pendiente de calibración"),
                   ("intercepto", "Intercepto (con la pendiente estimada)"),
                   ("citl", "Intercepto con pendiente fija en 1 (calibración en lo grande)")):
        L.append(f"| {lab} | {f(c[k])} | {f(c['ic'][k][0])} – {f(c['ic'][k][1])} |")
    L += ["", "## T2q", "", "| Quintil | n | p mín–máx | p media predicha | Prevalencia observada | Positivos |",
          "|---|---|---|---|---|---|"]
    for q in o["quintiles"]:
        L.append(f"| {q['quintil']} | {q['n']} | {f(q['p_min'])}–{f(q['p_max'])} | {f(q['p_media'])} | "
                 f"{f(q['prevalencia_observada'])} | {q['n_positivos']} |")
    if o["coeficientes"]:
        L += ["", "## T2c", "", "| Variable | Coef. estandarizado | Signo | Peso efectivo por punto (unidades originales) |",
              "|---|---|---|---|"]
        ef = o["coeficientes"]["peso_efectivo_por_punto"]
        for r in o["coeficientes"]["estandarizados"]:
            L.append(f"| `{r['variable']}` | {r['coef']:+.4f} | {r['signo']} | "
                     f"{f(ef.get(r['variable']), 4) if r['variable'] in ef else '(repartido en phq_total y gad_total)'} |")
    d = o["derivacion"]
    L += ["", "## T3", "", "| Predictor vs requiere_derivacion | Exactitud | F1-macro | κ | AUC | Sensibilidad | Especificidad | VPP | VPN |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, lab in (("SVM_c", "SVM actual"), ("reglas", "Motor de reglas")):
        v = d[k]
        L.append(f"| {lab} | " + " | ".join(ci(v, c) for c in
                 ["exactitud", "f1_macro", "kappa", "auc", "sensibilidad", "especificidad", "vpp", "vpn"]) + " |")
    L += ["", "## T3b", "", "| Nivel de la psicóloga | Sin derivación | Con derivación |", "|---|---|---|"]
    for nv in ("SIN_RIESGO", "BAJO", "MEDIO", "ALTO", "CRITICO"):
        r = d["cruce_nivel_vs_derivacion"].get(nv, {})
        L.append(f"| {nv} | {r.get('False', 0)} | {r.get('True', 0)} |")
    r = o["roc"]
    L += ["", "## T4", "", "| Curva | AUC | IC 95 % |", "|---|---|---|"]
    for k in ("SVM", "PHQ-A total"):
        L.append(f"| {k} | {f(r[k]['auc'])} | {f(r[k]['ic'][0])}–{f(r[k]['ic'][1])} |")
    dd = r["diferencia_SVM_menos_PHQ"]
    L.append(f"| Δ AUC (SVM − PHQ-A, bootstrap pareado) | {dd['delta']:+.3f} | {dd['ic'][0]:+.3f} a {dd['ic'][1]:+.3f} |")
    (R / "v3_tablas.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
