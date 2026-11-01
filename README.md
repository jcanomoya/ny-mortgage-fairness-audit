# HMDA New York 2024: Sesgo, Equidad y Explicabilidad en la Concesión de Hipotecas

> Proyecto final de la asignatura **Aprendizaje Avanzado**  
> Clasificación binaria de solicitudes hipotecarias con auditoría de equidad y mitigación de sesgo demográfico.

**Autores:** Izan Cuesta Corbí · Dennis García Solera · Marcos Segurado Llopis · Jesús Cano Moya

---

## Descripción del Proyecto

Este proyecto aborda la predicción automatizada de la **aprobación o denegación de solicitudes hipotecarias** en el estado de Nueva York (2024), utilizando datos reales del registro federal HMDA (*Home Mortgage Disclosure Act*). El eje central no es únicamente el rendimiento predictivo, sino también la relevancia de una auditoría de equidad. Este ciclo completo se compone de:

1. **Exploración y limpieza** del dataset con análisis de sesgo demográfico
2. **Modelo baseline** (Regresión Logística) como referencia de rendimiento y fairness
3. **Modelos de ensamble** (Random Forest, Gradient Boosting, XGBoost, LightGBM) con análisis SHAP
4. **Mitigación de sesgo** mediante Fairlearn (*Equalized Odds*) sobre el modelo ganador

---

## Estructura del Repositorio

```
AA-PROYECTO-HMDA-NY/
├── 📁 data/                                # Carpeta ignorada por tamaño (archivos .csv)
├── 📁 models/                              # Carpeta ignorada por tamaño (modelos .pkl)
│   ├── gradient_boosting.pkl
│   ├── lightgbm.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── 📁 notebooks/
│   ├── 01_EDA_Limpieza.ipynb               # Exploración, detección de sesgo y preprocesamiento
│   ├── 02_Modelo_Baseline.ipynb            # Regresión Logística + métricas de fairness base
│   ├── 03_Modelo_Sesgado_XAI.ipynb         # Modelos de ensamble + análisis SHAP
│   └── 04_Modelo_Mitigado_XAI.ipynb        # Mitigación con Fairlearn + SHAP post-mitigación
├── 📁 outputs/
│   └── images/                             # Gráficos exportados
├── 📁 report/
    ├── report.tex
    └── report.pdf
├── 📄 .gitignore
├── 📄 hmda-presentacion.html              # Archivo para exposición
├── 📄 README.md
└── 📄 requirements.txt
```

Las carpetas `data/` y `models/` **no han sido incluidas en el repositorio por limitaciones de tamaño de GitHub.**

Sin embargo, **el proyecto es completamente reproducible:**

* Ejecutando el Notebook 1, se generan todos los archivos de la carpeta `data/`. Incluidos `train.csv` y `test.csv` (únicos necesarios para los notebooks 2, 3 y 4).

* Ejecutando el Notebook 3, se almacenan todos los modelos de ensamble entrenados (uno por estrategia).


Para disponer del dataset original, la mejor opción es descargarlo desde la fuente oficial:  
🔗 [HMDA Data Browser — New York 2024 (CFPB)](https://ffiec.cfpb.gov/data-browser/data/2024?category=states&items=NY)
 
---
 
## Dataset
 
| Atributo | Detalle |
|---|---|
| **Fuente** | Consumer Financial Protection Bureau (CFPB) — HMDA 2024 |
| **Cobertura** | Todas las solicitudes hipotecarias en Nueva York durante 2024 |
| **Tamaño original** | 383.577 instancias, 99 columnas |
| **Tamaño tras filtrado** | ~283.000 instancias (solo acciones 1 y 3) |
| **Target** | `action_taken`: `1` = concedida, `0` = denegada |
| **Desbalance de clases** | ~74% aprobadas / ~26% denegadas |
| **Atributos sensibles** | `derived_race`, `derived_sex`, `derived_ethnicity` |
 
El dataset incluye características del préstamo (importe, tipo, LTV), del solicitante (ingresos, ratio deuda) y datos censales del barrio (porcentaje de minorías, renta relativa del área).
 
---
 
## Pipeline del Proyecto
 
### Notebook 1 — EDA & Limpieza
- Exploración inicial: tipos de datos, valores faltantes, distribuciones
- Detección visual de sesgo en atributos sensibles (raza, sexo, etnia, edad)
- Filtrado de instancias y columnas, eliminación de *data leakage*
- Imputación con `KNNImputer`, escalado con `RobustScaler`
- Codificación: `OrdinalEncoder` para variables ordinales, `OneHotEncoder` para nominales
- Partición estratificada train/test (`random_state=42`)

### Notebook 2 — Modelo Baseline
- Modelo: Regresión Logística con `class_weight='balanced'`
- Métricas globales de rendimiento
- Fairness baseline
- Establece la referencia de rendimiento y equidad para los modelos posteriores

### Notebook 3 — Modelos de Ensamble + XAI
- Comparativa con GridSearchCV sobre: Random Forest, Gradient Boosting, XGBoost, LightGBM
- Análisis SHAP: importancia global de features y auditoría de atributos sensibles
- Cuantificación del sesgo inherente del modelo ganador sin restricciones

### Notebook 4 — Mitigación de Sesgo + XAI
- `ExponentiatedGradient` (Fairlearn) con restricción `EqualizedOdds`
- Atributo sensible principal: `derived_race_White`
- Trade-off rendimiento/equidad
- SHAP post-mitigación: confirma la neutralización del peso de variables demográficas
---
 
## Tecnologías y Librerías
 
| Categoría | Herramientas |
|---|---|
| **Lenguaje** | Python 3.x |
| **Manipulación de datos** | `pandas`, `numpy` |
| **Visualización** | `matplotlib`, `seaborn` |
| **ML & Preprocesamiento** | `scikit-learn` |
| **Modelos de ensamble** | `scikit-learn`, `xgboost`, `lightgbm` |
| **Explicabilidad (XAI)** | `shap` |
| **Fairness & Mitigación** | `fairlearn` |
| **Reproducibilidad** | *`RANDOM_STATE = 42`* |
 
---
 
## Ejecución
 
1. **Clonar el repositorio**
    ```bash
    git clone https://github.com/jcm204/aa-proyecto-hmda-ny.git
    cd aa-proyecto-hmda-ny
    ```
 
2. **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```
 
3. **Descargar el dataset** desde el enlace oficial y colocarlo en `data/state_NY.csv`:  
    🔗 https://ffiec.cfpb.gov/data-browser/data/2024?category=states&items=NY

4. **Ejecutar el Notebook 1 completo.** Genera `data/train.csv` y `data/test.csv`. Sin este paso previo, los demás notebooks no pueden ejecutarse.

5. **Ejecutar los notebooks restantes en orden:**
    ```
    02_Modelo_Baseline.ipynb  →  03_Modelo_Sesgado_XAI.ipynb  →  04_Modelo_Mitigado_XAI.ipynb
    ```