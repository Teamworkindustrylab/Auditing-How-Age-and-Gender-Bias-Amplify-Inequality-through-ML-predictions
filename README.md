# Auditing How Age and Gender Bias Amplify Inequality through ML Predictions

Do machine learning classifiers merely reflect demographic inequality, or do they make it worse? This project measures **bias amplification** in developer reputation data, identifies its feature-level drivers via SHAP, and tests whether fairness mitigation strategies can bring predictions within a proposed compliance boundary of **|DPD| ≤ 0.10**.

---

## Datasets

| Dataset | N | Target | Sensitive attributes |
|---------|---|--------|----------------------|
| [Stack Overflow Developer Survey 2024](https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024) | ~65,000 | Salary above/below median | Age (primary), Gender |
| [freeCodeCamp 2018 New Coder Survey](https://github.com/freeCodeCamp/2018-new-coder-survey) | ~31,000 | Working as paid developer (yes/no) | Gender (primary), Age |
| [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/20/census+income) | 48,842 | Income above $50K/year | Gender (primary), Age |

> **Note on dataset choice.** An earlier version of this project used the GitHub Open Source Survey 2017 (~5,400 rows). It was replaced with the freeCodeCamp 2018 New Coder Survey because the GH sample was **97% men / 3% women**, leaving only ~150 women in the test split — too few to estimate gender-conditional DPD reliably. The FCC survey is **78% men / 21% women** across ~31K respondents, has the same Open Database License (ODbL), still loads directly from a public GitHub raw URL, and additionally exposes a numeric `age` field that supports intersectional gender × experience analysis.

> **Note on the third dataset.** UCI Adult / Census Income was added alongside FCC rather than instead of it. It contributes a much larger sample (48,842 rows) and a substantially larger raw gender gap in the target rate (30% of men earn >$50K vs. 11% of women) than either SO or FCC show on their own sensitive axis, which gives the project a "high raw bias, large N" reference point next to FCC's "low raw bias, large N" result. It uses the same binary-target architecture as FCC (`above_50k`, mirroring `paid_contributor`), so it required no changes to the DPD/EOD/amplification/mitigation machinery — only a new feature-engineering and bias-amplification class per notebook. A candidate fourth dataset, OPM FedScope, was evaluated and **rejected**: its public release is an aggregated count cube (age and salary binned, cells under 12 employees suppressed) rather than row-level microdata, which is incompatible with this pipeline's per-instance classifiers and SHAP analysis.

---

## SETUP

1.Download SO survey from: https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024
                        1.1. "Download Full Data Set" -> unzip
                        1.2. use survey_results_public.csv  (65,439 rows)
                        
Place SO survey at:  data/survey_results_public.csv

2. freeCodeCamp 2018 New Coder Survey loads automatically from the web (raw.githubusercontent.com).

3. UCI Adult / Census Income loads automatically via `pip install ucimlrepo` (`fetch_ucirepo(id=20)`).
   If `ucimlrepo` is unavailable, download `adult.data` + `adult.test` manually from
   https://archive.ics.uci.edu/dataset/20/census+income and place the combined CSV at
   data/adult_raw.csv -- 01_preprocessing.py will use the cached file if it already exists.

4. Run requrements.txt to install dependencies:  pip install -r requirements.txt 


## Pipeline

```
Data -> Bias baseline -> Model training -> Amplification ratio -> SHAP -> Mitigation -> Compliance audit
                         (LR + XGBoost)    DPD_model/DPD_data             Reweighing
                                           ratio > 1 = worse          Threshold calibration
```

---

## Key Metrics

| Metric | Definition |
|--------|-----------|
| **DPD** | ` P(ŷ=1 / g=female) − P(ŷ=1 / g=male) ` |
| **Amplification ratio** | ` DPD_model / DPD_data ` |
| **Proposed threshold** | ` DPD ≤ 0.10  — research-defined, consistent with fairness literature conventions ` |

---

## Install

```bash
Run requrements.txt to install dependencies:  

pip install -r requirements.txt 

```

---

*Data usage subject to [ODbL](https://opendatacommons.org/licenses/odbl/) (Stack Overflow Developer Survey 2024 and freeCodeCamp 2018 New Coder Survey) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (UCI Adult / Census Income).*


