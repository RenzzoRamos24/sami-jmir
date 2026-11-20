# Cohorte de septiembre 2026 — v3

Análisis adicionales para el paper (JMIR). Generado por `scripts/analisis_v3_*.py`
a partir de `data/clinico/casos_etiquetados.csv` (102 casos, 55 en alerta / 47 fuera)
y de dos lecturas de producción.

**Salvaguardas cumplidas**

- Producción: sesión PostgreSQL `readonly`, solo `SELECT`. No se escribió ni se desplegó nada.
- Firewall: **no se abrió ninguna regla**; la conexión funcionó con las reglas existentes, así que no hay nada que borrar ni verificar.
- Este archivo contiene solo conteos, métricas, códigos de caso y el texto literal de los 10 estímulos (parte del instrumento). No hay nombres ni texto libre de estudiantes. Los comentarios de la encuesta y el campo `respuesta` de las frases se procesaron en memoria y no se imprimieron ni se guardaron.
- Lo que no se pudo verificar está marcado **desconocido**.

**Protocolo común (puntos 1, 2, 3, 4, 7).** LOOCV anidado, la misma grilla de `train_svm_clinico.py`
(C ∈ {0.1, 0.3, 1, 3, 10, 30}; gamma ∈ {scale, 0.01, 0.03, 0.1}; kernel ∈ {rbf, linear}; k ∈ {4, 6, 8, 12, 16, 20}
solo donde hay `SelectKBest`), semilla 20261013, CV interna estratificada de 5 particiones, puntuación f1-macro,
`class_weight="balanced"`. Se importa del script de entrenamiento para que no pueda divergir.
Etiqueta: `riesgo_clinico >= MEDIO`. IC 95 %: bootstrap de 10 000 remuestreos de casos (percentiles), con
**los mismos índices** para todos los modelos. AUC calculado con el margen (SVM / logística) fuera de muestra;
para el motor de reglas, que es binario, equivale a (sensibilidad + especificidad) / 2.

**Control de reproducción:** el SVM (c) de esta corrida da las mismas 102 predicciones que
`reports/loocv_caso_por_caso.csv` (diferencia máxima de margen 5e-5, que es el redondeo a 4 decimales del archivo anterior).
O sea, el flujo es el mismo que produjo el F1-macro 0.9313.

---

## 1. Variables fijadas de antemano — SVM y regresión logística

Conjuntos: **(a)** `phq_total`, `gad_total`, ítem 9 ≥ 1 · **(b)** los 16 ítems crudos · **(c)** las 29 actuales
(con `SelectKBest` dentro del fold, como el modelo vigente).
Regresión logística: `StandardScaler` + L2, `C` en la misma grilla; en (c) también `k`.
Predicciones caso por caso: `reports/loocv_comparacion_modelos.csv` (102 filas, solo códigos de caso y números).

Cada celda: valor [IC 95 %].

{{T1}}

Matrices de confusión (positivo = alerta):

{{T1b}}

### McNemar exacto (binomial bilateral)

«Gana» = el modelo de la fila acierta y el comparador falla; «pierde» = al revés.

{{T1c}}

### Lectura

1. **Tu referencia preliminar para (a) se confirma a medias.** Exactitud 0.882 y κ 0.766 coinciden
   exactamente con la **regresión logística** (a) (0.882 / 0.766), pero no con el SVM (a), que da
   **0.922 / κ 0.843 / AUC 0.940**. Tu AUC 0.944 tampoco coincide con la logística (a) (**0.934**); 0.944 es el AUC
   del SVM (c). Con qué modelo y con cuántas variables salió el cálculo preliminar (3 o solo los 2 totales): **desconocido**.
   Si en el paper dice «solo con los totales», la cifra correcta depende de qué modelo se nombre.
2. **El conjunto de 29 no se distingue del de 3 con este n.** SVM (a) vs SVM (c): 1 gana / 2 pierden, p = 1.000.
   Con 3 variables el SVM acierta 94 de 102 y con 29 acierta 95. (c) tiene un falso positivo menos (1 vs 2) y la misma sensibilidad (49 de 55).
   Los IC de todas las métricas se solapan casi por completo.
3. **Los 16 ítems crudos rinden peor** (SVM 0.843, logística 0.853), al nivel del motor de reglas (7 / 8 y 7 / 7, p = 1.000),
   y significativamente peor que (c) en McNemar (SVM (b) vs (c): 0 / 9, p = 0.004; logística (b): 1 / 9, p = 0.021).
   Lo que aporta valor son las variables resumen, no los ítems sueltos.
4. **El SVM (c) vs reglas** reproduce lo ya reportado: 9 / 1, p = 0.021. El SVM (a) vs reglas no llega a significación (10 / 3, p = 0.092).
5. **Sin corrección por comparaciones múltiples.** Hay 13 comparaciones no triviales; con Bonferroni el umbral es 0.0038
   y ningún p lo cumple (el menor, 0.0039, queda apenas por encima). Reportarlo como exploratorio.
6. La logística iguala o queda cerca del SVM en (a) y (c) (AUC 0.934 y 0.942 vs 0.940 y 0.944): el kernel no es lo que
   hace la diferencia; sí lo es el conjunto de variables.
7. El sesgo de segundo orden de siempre se mantiene: las candidatas y la grilla se definieron viendo estos 102 casos.

---

## 2. Calibración y coeficientes del SVM final

### Calibración de Platt anidada

Dentro de cada una de las 102 vueltas: se elige el modelo con la CV interna (grilla completa), se obtienen los márgenes
fuera-de-muestra de esa CV interna sobre los 101 de entrenamiento, se ajusta la sigmoide de Platt (regresión logística
sobre el margen, sin regularización) y se aplica al margen del caso retenido. El caso evaluado no participa en nada de esto.
Pendiente e intercepto: regresión logística de la etiqueta sobre logit(p). IC por bootstrap de 10 000 remuestreos
de los 102 pares (p, y).

{{T2}}

Referencias: Brier de un predictor que solo conoce la prevalencia = **0.248**. Pendiente ideal = 1, intercepto ideal = 0.

Tabla de calibración por quintiles (orden por p, empates en orden de caso):

{{T2q}}

Lectura:

- El Brier (0.077) es casi un tercio del de la prevalencia sola. La pendiente puntual (0.914) y los interceptos (−0.085 / −0.039)
  están cerca de lo ideal, **pero los IC son muy anchos** (pendiente 0.587–2.388): con n = 102 no se puede afirmar buena calibración, solo
  que no hay evidencia de mala.
- Los quintiles 4 y 5 están saturados (p ≈ 0.93–1.000) y el quintil 5 son 20 de 20 positivos. **El corte entre los quintiles 4 y 5 es arbitrario**
  porque hay muchos casos empatados en p = 1.000 (el orden entre empatados es el del caso).
- La incertidumbre real está en los quintiles 1–3. El quintil 2 predice 0.202 y observa 0.095 (2 de 21); el 1 predice 0.035 y observa 0.095 (2 de 21),
  es decir, **hay 4 positivos en los 42 casos de menor probabilidad**; el 3 predice 0.539 y observa 0.600.
- Limitación del método: el calibrador se ajusta con márgenes de modelos entrenados con el 80 % de los 101, y se aplica al margen de un modelo
  entrenado con los 101. Es el procedimiento estándar pero no es exactamente la misma escala.

### Coeficientes del modelo final (ajustado con los 102)

Hiperparámetros elegidos: `kernel = linear`, `C = 3`, `k = 6` (coinciden con lo documentado en `CLAUDE.md`).
Coeficientes sobre las variables estandarizadas (`StandardScaler` dentro del pipeline), tomados de `SVC.coef_`; intercepto = 1.384.
Signo positivo = empuja hacia «alerta».

{{T2c}}

**Advertencia: los signos individuales no son interpretables por separado.** `total = phq_total + gad_total` exactamente (colinealidad perfecta),
y `phq_cognitiva` es una subsuma de los ítems de `phq_total`. El SVM lineal reparte el peso entre columnas redundantes. La última columna colapsa `total`
dentro de `phq_total` y `gad_total` y expresa el peso por punto en unidades originales. Aun así queda un peso levemente negativo para `gad_total`
(−0.103) y casi nulo para `phq_cognitiva` (−0.026): la señal de ansiedad entra por `gad_n_ge2` (cuántos ítems ≥ 2, +0.899 por ítem) y no por la suma.
`crisis` (ítem 9 ≥ 1) es la variable con mayor peso por unidad (+4.052).
No se calcularon IC ni estabilidad de los coeficientes (un SVM no tiene error estándar; habría que hacer bootstrap): **no calculado**.

---

## 3. Segundo criterio: `requiere_derivacion = Sí` (53 / 102)

Se usan **las mismas predicciones y márgenes del LOOCV** del SVM (c); no se reentrenó nada contra este criterio.
El motor de reglas va de referencia.

{{T3}}

Cruce de los dos criterios de la psicóloga:

{{T3b}}

- El SVM mantiene un buen desempeño contra este criterio (exactitud 0.892, κ 0.784, AUC 0.921), algo por debajo del que tiene contra su propia etiqueta, y sigue por encima de las reglas.
- **Este no es un criterio independiente.** Viene de la misma psicóloga, y la derivación coincide con `riesgo ≥ MEDIO` en 98 de 102 casos (κ entre ambos criterios = 0.921;
  las 4 discrepancias son 1 BAJO derivado y 3 MEDIO no derivados). Sirve como análisis de sensibilidad de la etiqueta, no como segunda validación. El mismo límite de un evaluador único (`SAMI-PSI-02` no etiquetó) sigue aplicando.

---

## 4. Curvas ROC

Figura: `reports/roc_svm_phq.png` (300 dpi, 1 800 × 1 800 px). Puntos de las curvas: `reports/roc_puntos.csv`
(columnas `curva, umbral, fpr, tpr`; incluye el punto del motor de reglas). Criterio: `riesgo_clinico >= MEDIO`, n = 102.

{{T4}}

Punto del motor de reglas: sensibilidad 0.782, 1 − especificidad 0.064.

- El punto de las reglas queda **por debajo de la curva del SVM** (con igual 1 − especificidad el SVM llega a ≈0.9 de sensibilidad, leído de la figura) y por encima de la curva del PHQ-A total en ese mismo punto (≈0.65).
- La diferencia de AUC SVM − PHQ-A total es +0.087 con IC 95 % (bootstrap pareado) que no cruza cero (+0.031 a +0.154).
- El AUC del SVM se calcula con los márgenes de 102 modelos distintos (uno por vuelta), no con un único modelo; es la práctica habitual en LOOCV, pero esos márgenes no están estrictamente en la misma escala.
  El PHQ-A total no tiene parámetros ajustados.

---

## 5. Banco de frases (10 estímulos usados en la cohorte)

Los estímulos son parte del instrumento (adaptación SSCT), no respuestas. «Crisis» = BETO clasifica la respuesta con depresión dominante **y** score ≥ 0.55.
«Detección» = una categoría clínica domina **y** supera 0.40 (como mucho una por frase, por eso detecciones = frases con detección). Score de depresión: media sobre todas las respuestas a ese estímulo.
Cada estímulo tiene un único texto en todos los casos.

{{FRASES}}

- El total de crisis (16) y de frases (1 005) coincide con lo reportado en v2.
- **El estímulo 22 («Cuando estoy triste yo…») concentra 7 de las 16 crisis (43.8 %) y tiene el mayor score medio de depresión (0.245).** Una explicación posible es que el estímulo nombra la tristeza y empuja a BETO hacia depresión; **no está verificado** (no se comparó con estímulos neutros ni con la clasificación de la psicóloga, `etiqueta_frase` sigue en 0).
- Los estímulos 25 («Cuando algo me preocupa…») y 13 («Cuando tengo un examen…») generan más detecciones (29 y 26) pero casi sin crisis (1 cada uno). Qué categoría domina en esas detecciones **no se verificó** en este análisis (el score medio de depresión de ambos, 0.233 y 0.187, no es el más alto).
- El nº de respuestas por estímulo es menor que 102 en varios (17: 98, 1: 100, etc.) porque algunos alumnos no completaron las 10 frases (ver v2 §4).

---

## 6. Encuesta de satisfacción (2026-09-29, 97 respuestas)

### α de Cronbach de los 5 ítems

Ítems: `facilidad_uso`, `claridad`, `utilidad`, `fluidez`, `satisfaccion_general`. 97 respuestas completas (ninguna con ítems vacíos).

| | |
|---|---|
| **α de Cronbach** | **{{ALFA}}** |
| IC 95 % (bootstrap de respondientes, 10 000) | {{ALFA_IC}} |
| Correlación media entre ítems | {{RMEDIA}} |
| α si se elimina cada ítem | {{ALFA_SIN}} |

Quitar cualquier ítem baja el α (ninguno sobra). Advertencias: el IC es ancho (n = 97); las respuestas están muy cargadas al 5 (medianas de 5), así que el α está limitado por el efecto techo;
y estos 5 ítems no son una escala validada ni coinciden con los 5 de la lista del revisor (ver v2 §2). El α describe la consistencia interna de este cuestionario, no su validez.

### Comentarios (29 de 97) por categoría temática

Conteos solamente. Categoría única por comentario.

| Categoría | Comentarios |
|---|---|
{{CAT}}
| **Total** | **29** |

**Método y límites, importante:** la clasificación es **automática, por reglas de palabras clave** (`scripts/analisis_v3_produccion.py`, con prioridad problemas técnicos > sugerencias > facilidad > emoción > otros).
**Yo no leí los comentarios** y la precisión de las reglas **no está verificada: desconocido**. Ningún comentario tocó dos categorías a la vez. De los 17 en «otros», 10 tienen 3 palabras o menos
(típicamente respuestas breves) y 7 tienen 4 o más: esos 7 son los que las reglas no supieron ubicar. «Problemas técnicos = 0» significa que las reglas no encontraron ninguno, no que se haya verificado que no los hay.
Antes de citar estos conteos en el paper, una persona del equipo debería revisar los 29 (unos minutos); si se prefiere, se puede reclasificar a mano y el script solo cambia de entrada.

---

## 7. Ablación 6(b): las 29 + 2 variables de BETO — resultado completo

Variables añadidas: `beto_max_depresion` y `beto_n_detectadas` (de `reports/beto_features_102.csv`). 31 variables, misma grilla, misma semilla, mismo LOOCV anidado.
**Sí había terminado:** `reports/ablacion_b_predicciones.csv` (102 filas, fechado el 2026-10-21) existe. Como el script que lo generó no está en el repositorio (**desconocido**),
repetí la ablación con el protocolo anterior y obtuve **las mismas 102 predicciones** que ese archivo.

| Métrica | SVM 29 (actual) | SVM 29 + 2 BETO |
|---|---|---|
| Exactitud | 0.931 | 0.931 |
| F1-macro | 0.931 | 0.931 |
| κ | 0.863 | 0.863 |
| AUC | 0.944 | 0.944 |
| Sensibilidad / especificidad | 0.891 / 0.979 | 0.891 / 0.979 |
| VP / FP / VN / FN | 49 / 1 / 46 / 6 | 49 / 1 / 46 / 6 |

(IC 95 % en la tabla del punto 1; son idénticos.)

- **Ninguno de los 102 casos cambia de predicción.** McNemar exacto: 0 / 0 discordancias, p = 1.000 (ver la tabla del punto 1).
  Agregar BETO no mejora ni empeora nada, y deja iguales los 6 falsos negativos y el único falso positivo.
- No se verificó si `SelectKBest` llegó a elegir alguna de las dos variables de BETO en algún fold: **desconocido** (se guardaron los hiperparámetros, no las variables elegidas de esa corrida).
  El resultado final es idéntico de todos modos.
- Para el paper: el F1-macro del SVM se obtiene sin información de BETO y añadirla no cambia ninguna decisión del SVM. Lo anterior no dice nada sobre BETO en las decisiones del *sistema* (ver v2 §4: aporta 8 banderas de crisis que el ítem 9 no activa).

---

## 8. Reproducibilidad

Commits locales, **sin push** (ver la sección de commits al final de la respuesta del chat, con los hashes).

| Qué | Dónde |
|---|---|
| LOOCV de las 6 combinaciones + Platt + ablación | `scripts/analisis_v3_loocv.py` |
| Lecturas de producción (puntos 5 y 6) | `scripts/analisis_v3_produccion.py` |
| Estadísticas, calibración, ROC | `scripts/analisis_v3_estadisticas.py` |
| Este informe | `scripts/analisis_v3_informe.py` + `scripts/analisis_v3_plantilla.md` |
| Salidas (no versionadas: `reports/` está en `.gitignore`) | `reports/loocv_comparacion_modelos.csv`, `reports/roc_svm_phq.png`, `reports/roc_puntos.csv`, `reports/v3_*.json` |

Orden de ejecución: `analisis_v3_loocv.py` → `analisis_v3_estadisticas.py` → (con `DB_*` en el entorno) `analisis_v3_produccion.py` → `analisis_v3_informe.py`.
Se instaló `matplotlib` en el `venv` para la figura (no está en `requirements`: **desconocido** si hay que añadirlo).
