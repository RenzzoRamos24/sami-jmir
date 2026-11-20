# Sami

Code for the platform and analyses reported in the JMIR Mental Health manuscript
on Sami, a screening system for depressive and anxiety symptoms in secondary-school
students (rule engine on PHQ-A and GAD-7, BETO zero-shot analysis of sentence
completions, and an SVM trained against blind clinical judgement).

**This repository contains no participant data.** The 102 labelled cases are
available on request from the authors; they are not public because they are
mental-health data of minors protected under Peruvian Law 29733.

## Tags

| Tag | Content |
|---|---|
| `sesion-2026-09-29` | Platform code as deployed during the 29/09/2026 session (development commit `00af339`). |
| `jmir-v1` | Adds the analysis scripts used in the manuscript, this README and the licence. |

The development repository is private. Relative to commit `00af339`, the first
tag removes non-code files (management documents, designs) and two document
generators that identify the institution, and replaces the default admin
password and production URL with placeholders. No program logic was changed.

## Layout

```
app/        Backend (FastAPI + SQLAlchemy)
  services/evaluator_service.py   Rule engine: scale cut-offs, crisis flags, composite risk
  services/nlp_service.py         BETO zero-shot (4 categories; thresholds 0.40 / 0.55)
  services/svm_features.py        Feature construction for the clinical SVM (jmir-v1)
project/    Frontend (Vue 3 + Vite)
seeds/      Fixed instrument bank (PHQ-A, GAD-7, ... and 40 sentence stems)
scripts/    Training, evaluation and utility scripts
models/     Pretrained artefacts shipped with the platform
```

## Running the platform

```bash
python3.12 -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env            # set ADMIN_EMAIL, ADMIN_PASSWORD, JWT_SECRET
venv/bin/python -c "from app.database import engine, Base; import app.models; Base.metadata.create_all(bind=engine)"
sqlite3 mental_health.db < seeds/banco_instrumentos.sql
venv/bin/python -m scripts.seed_admin
venv/bin/uvicorn app.main:app --port 8000
cd project && npm install && npm run dev
```

## Reproducing the analyses (tag `jmir-v1`)

The scripts expect the labelled cases at `data/clinico/casos_etiquetados.csv`
(one row per case: case code, the 16 PHQ-A and GAD-7 item scores, the rule-engine
output and the evaluator's label). That file is not distributed; see above.

```bash
venv/bin/pip install -r requirements-analisis.txt

# Clinical SVM: nested leave-one-out CV (seed 20261013, inner 5-fold stratified)
PYTHONPATH=. venv/bin/python scripts/train_svm_clinico.py --casos data/clinico/casos_etiquetados.csv

# Additional analyses (model comparison, bootstrap CIs, McNemar, calibration)
PYTHONPATH=. venv/bin/python scripts/analisis_v3_loocv.py --casos data/clinico/casos_etiquetados.csv
PYTHONPATH=. venv/bin/python scripts/analisis_v3_estadisticas.py   # reads data/clinico/casos_etiquetados.csv
PYTHONPATH=. venv/bin/python scripts/analisis_v3_informe.py
```

`scripts/exportar_casos_etiquetados.py` and `scripts/analisis_v3_produccion.py`
read the production database (read-only) and need `DB_SERVER`, `DB_NAME`,
`DB_USER` and `DB_PASS` in the environment. Outputs are written to `reports/`
(git-ignored) and contain case codes and numbers only, never free text.

## Licence

MIT. See `LICENSE`.
