"""Análisis v3, paso 4: arma datos-setiembre-v3.md a partir de los JSON/CSV de
`reports/`. Las tablas salen de `reports/v3_tablas.md` y `reports/v3_produccion.json`;
la prosa vive en `PLANTILLA`. No toca producción ni imprime texto de estudiantes.

Uso:  venv/bin/python scripts/analisis_v3_informe.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def secciones() -> dict[str, str]:
    t = (R / "v3_tablas.md").read_text()
    partes = re.split(r"^## (\S+)\n", t, flags=re.M)
    return {partes[i]: partes[i + 1].strip() for i in range(1, len(partes), 2)}


def tabla_frases(prod: dict) -> str:
    L = ["| Nº | Estímulo (literal) | Respuestas | Crisis | Detecciones | Score medio de depresión |",
         "|---|---|---|---|---|---|"]
    tot = [0, 0, 0]
    for f in prod["banco_frases"]["frases"]:
        L.append(f"| {f['numero']} | {f['texto_estimulo'][0]} | {f['n_respuestas']} | "
                 f"{f['n_crisis']} | {f['n_detecciones']} | {f['score_depresion_medio']:.3f} |")
        tot[0] += f["n_respuestas"]; tot[1] += f["n_crisis"]; tot[2] += f["n_detecciones"]
    L.append(f"| **Total** | | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | |")
    return "\n".join(L)


def main():
    S = secciones()
    prod = json.loads((R / "v3_produccion.json").read_text())
    e = prod["encuesta"]
    c = e["comentarios_por_categoria"]
    plantilla = (ROOT / "scripts" / "analisis_v3_plantilla.md").read_text()
    reemplazos = {f"{{{{{k}}}}}": v for k, v in S.items()}
    reemplazos["{{FRASES}}"] = tabla_frases(prod)
    reemplazos["{{ALFA}}"] = f"{e['alfa_cronbach']:.3f}"
    reemplazos["{{ALFA_IC}}"] = f"{e['alfa_ic95_bootstrap'][0]:.3f}–{e['alfa_ic95_bootstrap'][1]:.3f}"
    reemplazos["{{ALFA_SIN}}"] = " · ".join(f"sin `{k}` {v:.3f}" for k, v in e["alfa_sin_item"].items())
    reemplazos["{{RMEDIA}}"] = f"{e['correlacion_media_entre_items']:.3f}"
    reemplazos["{{CAT}}"] = "\n".join(
        f"| {k} | {c.get(k, 0)} |" for k in
        ["facilidad", "sugerencias", "emoción", "problemas técnicos", "otros"])
    out = plantilla
    for k, v in reemplazos.items():
        out = out.replace(k, v)
    faltan = re.findall(r"\{\{[^}]+\}\}", out)
    assert not faltan, faltan
    (ROOT / "datos-setiembre-v3.md").write_text(out)
    print("escrito datos-setiembre-v3.md")


if __name__ == "__main__":
    main()
