"""Análisis v3, paso 2: lecturas de SOLO LECTURA de producción (puntos 5 y 6).

  5. Banco de frases: texto de cada estímulo (instrumento, no respuesta de
     estudiante), nº de veces con crisis, nº de detecciones y score medio de
     depresión de BETO, sobre los casos de `casos_etiquetados.csv`.
  6. Encuesta de satisfacción: α de Cronbach de los 5 ítems y conteos de los
     comentarios por categoría temática (clasificación por reglas de palabras).

Garantías de privacidad: la sesión es `readonly`; el campo `respuesta` de las
frases y el campo `comentario` se leen en memoria y NO se imprimen ni se
guardan. Solo salen conteos, métricas y el texto de los estímulos.

Uso (credenciales por entorno, nunca por argumento):
    export DB_SERVER=... DB_NAME=... DB_USER=... DB_PASS=...
    venv/bin/python scripts/analisis_v3_produccion.py \
        --casos data/clinico/casos_etiquetados.csv --salida reports/v3_produccion.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FECHA_ENCUESTA = "2026-09-29"
ITEMS_ENCUESTA = ["facilidad_uso", "claridad", "utilidad", "fluidez",
                  "satisfaccion_general"]

# Reglas de palabras (sobre texto sin tildes y en minúsculas). Prioridad en
# este orden cuando un comentario toca varias categorías.
CATEGORIAS = [
    ("problemas técnicos", r"\b(error|errores|falla|fallo|fallas|lento|lenta|"
     r"demora|demoro|se cayo|cayo|no cargo|no carga|no abre|no funciona|bug|"
     r"internet|conexion|se trabo|trabo|colgo|pantalla|tardo|tarda|"
     r"no me dejo|no dejaba)\b"),
    ("sugerencias", r"\b(sugiero|sugerencia|sugerir|deberia|deberian|podria|"
     r"podrian|mejorar|mejoraria|mejor si|agregar|agreguen|agregaria|"
     r"añadir|anadir|falta|faltan|faltaria|quisiera|me gustaria|ojala|"
     r"recomiendo|cambiar|cambiaria|mas preguntas|menos preguntas|"
     r"pondria|poner)\b"),
    ("facilidad", r"\b(facil|faciles|sencillo|sencilla|simple|simples|rapido|"
     r"rapida|practico|practica|intuitivo|intuitiva|claro|clara|claras|"
     r"entendible|comprensible|comodo|comoda|ordenado|amigable|"
     r"entender|entendi)\b"),
    ("emoción", r"\b(senti|siento|sentir|emocion|emociones|emocional|"
     r"tranquilo|tranquila|feliz|triste|bien|mal|alivio|aliviado|"
     r"desahogar|desahogo|desahogarme|ayuda|ayudo|ayuda|gracias|"
     r"gusto|me gusto|nervios|nervioso|nerviosa|confianza|comodidad|"
     r"escuchado|escuchada|agradable|lindo|bonito|genial|excelente|"
     r"increible|interesante|miedo|ansiedad|llorar)\b"),
]


def norm(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def clasificar(texto: str) -> tuple[str, int]:
    """(categoría principal, nº de categorías que tocó)."""
    t = norm(texto or "").strip()
    toca = [n for n, rx in CATEGORIAS if re.search(rx, t)]
    return (toca[0] if toca else "otros"), len(toca)


def cronbach(M: np.ndarray) -> float:
    k = M.shape[1]
    return float(k / (k - 1) * (1 - M.var(0, ddof=1).sum()
                                / M.sum(1).var(ddof=1)))


def conectar():
    import psycopg2
    faltan = [v for v in ("DB_SERVER", "DB_NAME", "DB_USER", "DB_PASS")
              if not os.environ.get(v)]
    if faltan:
        sys.exit(f"Faltan variables de entorno: {', '.join(faltan)}")
    c = psycopg2.connect(
        host=f"{os.environ['DB_SERVER']}.postgres.database.azure.com",
        port=5432, dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"], sslmode="require", connect_timeout=20)
    c.set_session(readonly=True, autocommit=False)
    return c


def frases(cur, ids: list[int]) -> dict:
    cur.execute("select id, resultado_json from aplicacion_cuestionario "
                "where id = any(%s)", (ids,))
    textos: dict[int, set] = defaultdict(set)
    ag = defaultdict(lambda: {"n": 0, "crisis": 0, "detecciones": 0,
                              "frases_con_deteccion": 0, "dep": [],
                              "casos_crisis": set()})
    escalas = Counter()
    for aid, rj in cur.fetchall():
        r = json.loads(rj) if rj else {}
        for f in r.get("frases", []):
            n = int(f["numero"])
            textos[n].add(f.get("pregunta", ""))          # estímulo, no respuesta
            a = ag[n]
            a["n"] += 1
            a["crisis"] += bool(f.get("crisis"))
            if f.get("crisis"):
                a["casos_crisis"].add(int(aid))
            det = f.get("detectadas") or []
            a["detecciones"] += len(det)
            a["frases_con_deteccion"] += bool(det)
            sc = f.get("scores") or {}
            escalas.update(sc.keys())
            if "depresion" in sc:
                a["dep"].append(float(sc["depresion"]))
    out = []
    for n in sorted(ag):
        a = ag[n]
        out.append({"numero": n, "texto_estimulo": sorted(textos[n]),
                    "n_respuestas": a["n"], "n_crisis": a["crisis"],
                    "n_detecciones": a["detecciones"],
                    "n_frases_con_deteccion": a["frases_con_deteccion"],
                    "casos_con_crisis": sorted(f"CASO-{c}" for c in a["casos_crisis"]),
                    "score_depresion_medio": float(np.mean(a["dep"])) if a["dep"] else None,
                    "n_scores_depresion": len(a["dep"])})
    return {"frases": out, "claves_scores": dict(escalas)}


def encuesta(cur) -> dict:
    cur.execute(f"select {', '.join(ITEMS_ENCUESTA)}, comentario from "
                "satisfaction_surveys where \"timestamp\"::date = %s",
                (FECHA_ENCUESTA,))
    filas = cur.fetchall()
    df = pd.DataFrame(filas, columns=ITEMS_ENCUESTA + ["comentario"])
    completos = df.dropna(subset=ITEMS_ENCUESTA)
    M = completos[ITEMS_ENCUESTA].to_numpy(float)
    # IC 95 % del α por bootstrap de respondientes (10 000, semilla fija)
    rng = np.random.default_rng(20261013)
    bs = []
    for _ in range(10000):
        s = M[rng.integers(0, len(M), len(M))]
        if (s.sum(1).var(ddof=1) > 0):
            bs.append(cronbach(s))
    alfa_sin = {it: cronbach(np.delete(M, j, 1))
                for j, it in enumerate(ITEMS_ENCUESTA)}
    com = df.comentario.dropna().map(lambda t: t.strip())
    com = com[com != ""]
    cats, multi = Counter(), 0
    for t in com:
        c, k = clasificar(t)
        cats[c] += 1
        multi += k >= 2
    return {"fecha": FECHA_ENCUESTA, "n_respuestas": int(len(df)),
            "n_completos": int(len(completos)),
            "alfa_cronbach": cronbach(M),
            "alfa_ic95_bootstrap": [float(np.percentile(bs, 2.5)),
                                    float(np.percentile(bs, 97.5))],
            "alfa_sin_item": alfa_sin,
            "correlacion_media_entre_items": float(
                np.corrcoef(M.T)[np.triu_indices(5, 1)].mean()),
            "n_comentarios": int(len(com)),
            "comentarios_por_categoria": dict(cats),
            "comentarios_que_tocan_2_o_mas_categorias": int(multi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args()
    ids = [int(i) for i in pd.read_csv(a.casos).aplicacion_id]
    conn = conectar()
    cur = conn.cursor()
    res = {"n_casos": len(ids), "banco_frases": frases(cur, ids),
           "encuesta": encuesta(cur)}
    conn.rollback()
    conn.close()
    a.salida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("guardado", a.salida)


if __name__ == "__main__":
    main()
