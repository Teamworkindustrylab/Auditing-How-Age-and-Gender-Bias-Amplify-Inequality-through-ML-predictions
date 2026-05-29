# Auditing How Age and Gender Bias Amplify Inequality through ML Predictions

Do machine learning classifiers merely reflect demographic inequality, or do they make it worse? This project measures **bias amplification** in developer reputation data, identifies its feature-level drivers via SHAP, and tests whether fairness mitigation strategies can bring predictions within a proposed compliance boundary of **|DPD| ≤ 0.10**.

---

## Datasets

| Dataset | N | Target | Sensitive attributes |
|---------|---|--------|----------------------|
| [Stack Overflow Developer Survey 2024](https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024) | ~65,000 | Salary above/below median | Gender, Age |
| [GitHub Open Source Survey 2017](https://github.com/github/open-source-survey) | ~5,400 | OSS participation | Gender, Age |

---

## SETUP

1.Download SO survey from: https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024
                        1.1. "Download Full Data Set" -> unzip
                        1.2. use survey_results_public.csv  (65,439 rows)
                        
Place SO survey at:  data/survey_results_public.csv

2. GitHub OSS Survey loads automatically from the web.

3. Run requrements.txt to install dependencies:  pip install -r requirements.txt 


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

*Data usage subject to [ODbL](https://opendatacommons.org/licenses/odbl/) (Stack Overflow) and [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (GitHub OSS).*


