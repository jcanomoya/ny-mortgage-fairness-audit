# HMDA New York 2024: Algorithmic Fairness Audit for Mortgage Credit Risk

## Business Objective

Automated credit decisioning is only as valuable as it is defensible. Lenders increasingly rely on machine learning to assess mortgage applications at scale, but a model that predicts defaults or approvals accurately can still systematically disadvantage protected groups, exposing the institution to regulatory action, reputational damage and unfair outcomes for applicants.

This project builds an **automated mortgage approval model for New York State (2024)** and, critically, treats fairness as a first-class requirement rather than an afterthought. Using real federal HMDA (*Home Mortgage Disclosure Act*) data, it delivers the full cycle a bank's risk and compliance teams need: predict, audit, explain, and mitigate. The goal is a credit-risk assessment that supports **anti-discrimination compliance** (such as ECOA and fair lending requirements) without discarding predictive performance.

---

## Strategic Value & Architecture

### Why this matters for a lender

* **Regulatory defensibility:** Every model is audited against sensitive attributes (`derived_race`, `derived_sex`, `derived_ethnicity`), producing evidence of how decisions differ across demographic groups.
* **Explainability on demand:** SHAP analysis shows which features drive decisions, and whether demographic variables (or close proxies) carry weight in the model's reasoning.
* **Measured performance/equity trade-off:** Instead of assuming fairness comes for free, the project quantifies exactly what predictive performance is traded for a fairer model, which is the question a risk committee will ask.
* **Reproducible end to end:** Fixed seeds and a fully scripted pipeline make results auditable and repeatable.

### Four-stage pipeline

| Stage | Notebook | Purpose |
|---|---|---|
| **1. Data audit & preparation** | `01_EDA_Limpieza.ipynb` | Explore the data, detect demographic bias, remove data leakage, impute, scale, encode and split |
| **2. Baseline** | `02_Modelo_Baseline.ipynb` | Logistic Regression as the reference point for both performance and fairness |
| **3. Biased model + explainability** | `03_Modelo_Sesgado_XAI.ipynb` | Tuned ensemble models, SHAP analysis, and quantification of the unconstrained model's inherent bias |
| **4. Bias mitigation** | `04_Modelo_Mitigado_XAI.ipynb` | Fairness-constrained retraining of the winning model and post-mitigation SHAP verification |

**Stage details**

* **`01_EDA_Limpieza.ipynb`**
  * Initial profiling: data types, missing values, distributions
  * Visual detection of bias across sensitive attributes (race, sex, ethnicity, age)
  * Row and column filtering, plus removal of *data leakage*
  * Imputation with `KNNImputer`, scaling with `RobustScaler`
  * Encoding: `OrdinalEncoder` for ordinal variables, `OneHotEncoder` for nominal ones
  * Stratified train/test split (`random_state=42`)
* **`02_Modelo_Baseline.ipynb`**
  * Logistic Regression with `class_weight='balanced'`
  * Global performance metrics and baseline fairness metrics
* **`03_Modelo_Sesgado_XAI.ipynb`**
  * `GridSearchCV` comparison across Random Forest, Gradient Boosting, XGBoost and LightGBM
  * SHAP global feature importance and audit of sensitive attributes
  * Quantification of the winning model's bias before any intervention
* **`04_Modelo_Mitigado_XAI.ipynb`**
  * `ExponentiatedGradient` (Fairlearn) with an `EqualizedOdds` constraint
  * Primary sensitive attribute: `derived_race_White`
  * Performance vs. fairness trade-off analysis
  * Post-mitigation SHAP to confirm demographic variables no longer drive predictions

### Dataset

| Attribute | Detail |
|---|---|
| **Source** | Consumer Financial Protection Bureau (CFPB), HMDA 2024 |
| **Coverage** | All mortgage applications in New York State during 2024 |
| **Original size** | 383,577 records, 99 columns |
| **Size after filtering** | ~283,000 records (only action codes 1 and 3) |
| **Target** | `action_taken`: `1` = approved, `0` = denied |
| **Class imbalance** | ~74% approved / ~26% denied |
| **Sensitive attributes** | `derived_race`, `derived_sex`, `derived_ethnicity` |

Features cover loan characteristics (amount, type, LTV), applicant profile (income, debt ratio) and census-tract context (minority population share, relative area income).

---

## Key Results & Metrics

Every model is evaluated on two axes: **predictive performance** and **fairness**. Evaluation uses a stratified 80/20 split (226,665 training / 56,667 test applications, 80 features). The racial gap is the approval-rate difference between White and Black or African American applicants.

| Model | Accuracy | F1 | ROC-AUC | Racial Approval Gap | TPR Gap (White − Black) | Notes |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7120 | 0.7874 | 0.7697 | 34.56 pp | 32.31 pp | Reference point (`02_Modelo_Baseline.ipynb`) |
| LightGBM, unconstrained | 0.8621 | 0.9116 | 0.8868 | 18.39 pp | 5.20 pp | Winner of the comparison (`03_Modelo_Sesgado_XAI.ipynb`) |
| LightGBM, bias-mitigated | **0.8622** | **0.9117** | 0.7803 | **13.71 pp** | **1.67 pp** | `ExponentiatedGradient` + `EqualizedOdds` (`04_Modelo_Mitigado_XAI.ipynb`) |

**What the results show**

* **Bias is structural, not model-specific.** Even the simple linear baseline reproduces a 34.56 pp racial approval gap, reflecting historical inequality in the data.
* **LightGBM was selected for performance and efficiency.** It matches Gradient Boosting on ROC-AUC (0.8868 vs. 0.8871) while training in 1.90 s instead of ~320 s, which matters for the iterative retraining that mitigation requires. Its lower gap versus the baseline (18.39 pp) is a side effect of better overall fit, not of any fairness treatment, and SHAP confirms the model actively uses race.
* **Mitigation narrows the gap at almost no cost to decisions.** The racial gap falls from 18.39 pp to 13.71 pp: approval for Black applicants rises from 66.86% to 70.42% while White approval moves from 85.25% to 84.13%. Accuracy and F1 are essentially unchanged, and the confusion matrices are nearly identical (false negatives 1,484 → 1,455; false positives 6,329 → 6,354).
* **The real cost is ranking power.** ROC-AUC drops from 0.8868 to 0.7803 (−10.65 pp), so the trade-off lands on the model's ability to rank applicants by probability, not on its binary decisions.
* **Demographic influence shrinks, financial logic stays intact.** SHAP importance of `derived_race_White` falls from ~0.10 (rank 8) to ~0.03, while `debt_to_income_ratio`, `preapproval`, `loan_purpose` and `aus_1` keep their dominance.

**Limitations (stated openly)**

* The fairness constraint targets a single binary attribute (`derived_race_White`). The ethnicity gap stays essentially unchanged (~13 pp) and sex disparities barely move, so mitigating one dimension does not generalize to others.
* A residual racial gap of 13.71 pp remains; algorithmic mitigation reduces bias but does not eliminate it, and should sit within a broader human-led audit process.
* HMDA records bank decisions, not true creditworthiness, so models learn to replicate historical decisions, including their bias.
* `ExponentiatedGradient` returns a randomized classifier, which introduces variability to account for in any real deployment.

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Data manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **ML & preprocessing** | `scikit-learn` |
| **Ensemble models** | `scikit-learn`, `xgboost`, `lightgbm` |
| **Explainability (XAI)** | `shap` |
| **Fairness & mitigation** | `fairlearn` |
| **Reproducibility** | `RANDOM_STATE = 42` |

---

## Repository Structure

```text
AA-PROYECTO-HMDA-NY/
├── 📁 data/                                # Git-ignored due to size (.csv files)
├── 📁 models/                              # Git-ignored due to size (.pkl models)
│   ├── gradient_boosting.pkl
│   ├── lightgbm.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── 📁 notebooks/
│   ├── 01_EDA_Limpieza.ipynb               # Exploration, bias detection and preprocessing
│   ├── 02_Modelo_Baseline.ipynb            # Logistic Regression + baseline fairness metrics
│   ├── 03_Modelo_Sesgado_XAI.ipynb         # Ensemble models + SHAP analysis
│   └── 04_Modelo_Mitigado_XAI.ipynb        # Fairlearn mitigation + post-mitigation SHAP
├── 📁 outputs/
│   └── images/                             # Exported charts
├── 📁 report/
│   ├── report.tex
│   └── report.pdf
├── 📄 .gitignore
├── 📄 hmda-presentacion.html               # Project presentation
├── 📄 README.md
└── 📄 requirements.txt
```

> `data/` and `models/` are not included in the repository because of GitHub size limits. The project is fully reproducible:
>
> * Running `01_EDA_Limpieza.ipynb` generates every file in `data/`, including `train.csv` and `test.csv` (the only ones required by notebooks 02, 03 and 04).
> * Running `03_Modelo_Sesgado_XAI.ipynb` saves all trained ensemble models, one per strategy.

---

## Setup & Execution

1. **Clone the repository**
    ```bash
    git clone https://github.com/jcm204/aa-proyecto-hmda-ny.git
    cd aa-proyecto-hmda-ny
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset** from the official source and place it at `data/state_NY.csv`:
    [HMDA Data Browser: New York 2024 (CFPB)](https://ffiec.cfpb.gov/data-browser/data/2024?category=states&items=NY)

4. **Run `01_EDA_Limpieza.ipynb` in full.** It generates `data/train.csv` and `data/test.csv`. The remaining notebooks cannot run without this step.

5. **Run the remaining notebooks in order:**
    ```text
    02_Modelo_Baseline.ipynb  →  03_Modelo_Sesgado_XAI.ipynb  →  04_Modelo_Mitigado_XAI.ipynb
    ```

---

<sub>**Collaborators:** Izan Cuesta Corbí, Dennis García Solera, Marcos Segurado Llopis, Jesús Cano Moya.</sub>
