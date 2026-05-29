# Project Findings: How Age and Gender Bias Amplify Inequality through ML Predictions

## Overview

This document summarises the key findings from our audit of two developer datasets — the **Stack Overflow Developer Survey 2024** (SO, ~65,000 respondents) and the **GitHub Open Source Survey 2017** (GH, ~5,400 respondents). We trained Logistic Regression and XGBoost classifiers to predict **high salary** (SO) and **OSS participation** (GH), then measured how much the models amplify pre-existing demographic gaps.

---

## 0. Responsible AI Framework & Tooling

This project is grounded in two complementary responsible AI frameworks:

**AI Fairness 360 (AIF360)** is IBM's open-source fairness toolkit, one of the most widely cited frameworks in the algorithmic fairness literature. It provides standardised implementations of fairness metrics (including Demographic Parity Difference and Equalized Odds Difference) and mitigation algorithms such as reweighing. Using AIF360 means our measurements follow consistent, peer-reviewed definitions — not ad hoc calculations — making our results reproducible and comparable to other published audits.

**Fairlearn** (Microsoft) complements AIF360 with additional post-processing strategies including threshold calibration, and integrates directly with scikit-learn pipelines.

**Why DPD ≤ 0.10 as the compliance threshold?**
The Demographic Parity Difference threshold of |DPD| ≤ 0.10 is not arbitrary. It draws from the **four-fifths (80%) rule** in US employment discrimination law (EEOC guidelines), which flags a selection rate below 80% of the majority group's rate as evidence of adverse impact — equivalent to roughly a 0.10–0.20 DPD range depending on base rates. The ≤ 0.10 threshold also appears in recent EU AI Act impact assessments and fairness literature (e.g. Chouldechova 2017; Verma & Rubin 2018) as a practical boundary between tolerable statistical noise and actionable disparity. Choosing a research-grounded, externally anchored threshold rather than a project-internal one is itself a responsible AI practice — it prevents teams from moving the goalposts to make their model look compliant.

---

## 1. Do the datasets already contain bias before any model is trained?

Yes — both datasets show a measurable demographic gap in the *raw data*, before a single model is involved.

| Dataset | Sensitive attribute | Data gap (DPD) |
|---------|---------------------|----------------|
| Stack Overflow | Age group (young vs experienced) | 0.305 |
| GitHub OSS | Gender (man vs woman) | 0.131 |

The SO data gap is large: young developers are already 30 percentage points less likely to earn above the median salary than experienced developers, simply reflecting labour-market inequality. The GH gap is smaller but meaningful — women are about 13 points less likely to be classified as active OSS contributors.

---

## 2. Do ML models make things worse? (Bias Amplification)

The central question is whether models *amplify* these gaps. We measure this with the **amplification ratio** = model DPD / data DPD. A ratio above 1.0 means the model makes pre-existing inequality worse.

### Stack Overflow

| Sensitive definition | Model | Amplification ratio |
|----------------------|-------|---------------------|
| Age group (binary) | Logistic Regression | **2.07** |
| Age group (binary) | XGBoost | **1.86** |
| Age × YearsCodePro | Logistic Regression | **1.85** |
| Age × YearsCodePro | XGBoost | **1.86** |
| Age × YearsCode | Logistic Regression | **1.64** |
| Age × YearsCode | XGBoost | **1.67** |

All ratios are well above 1.0. The Logistic Regression model nearly **doubles** the existing age gap.

### GitHub OSS

| Model | Data gap | Model DPD | Amplification ratio |
|-------|----------|-----------|---------------------|
| Logistic Regression | 0.131 | 0.013 | **0.099** |
| XGBoost | 0.131 | 0.024 | **0.184** |

The GH models show the opposite pattern — both reduce the gender gap relative to raw data. This reflects the smaller sample, weaker signal in features, and the different nature of OSS participation as a target.

**Key takeaway:** Bias amplification is not guaranteed, but where it occurs (SO), it can be severe.

---

## 3. Which features drive biased predictions? (SHAP Analysis)

SHAP (SHapley Additive exPlanations) identifies which features contribute most to each model decision.

### Stack Overflow — top features

| Feature | Mean |SHAP| |
|---------|--------|
| `years_code_pro` | 0.453 |
| `years_code` | 0.401 |
| `ed_level_enc` | 0.146 |

The two experience features dominate by a wide margin. This is the key insight: **the model never uses age directly**, but age is strongly correlated with years of coding experience, so young developers are penalised *indirectly*. This is classic *proxy discrimination* — the kind that standard audits checking only for direct attribute use would miss entirely.

### GitHub OSS — top features

| Feature | Mean |SHAP| |
|---------|--------|
| `pro_experience_yrs` | 0.268 |
| `oss_experience_yrs` | 0.186 |
| `contributor_help_score` | 0.095 |

Experience features dominate again. Community-attitude scores (`find_answers_score`, `receive_help_score`) also appear — these may partially reflect the chilling effect of hostile environments on participation, a known barrier for women in open source.

---

## 4. Mitigation Strategies & Compliance

We tested two strategies from the AIF360/Fairlearn toolkit against the |DPD| ≤ 0.10 threshold:

- **Reweighing** (Kamiran & Calders 2012): adjusts per-sample training weights inversely proportional to group × label frequency, then retrains. A pre-processing approach — the model itself changes.
- **Threshold calibration** (post-processing): per-group binary search for the decision threshold that equalises positive-prediction rates. No retraining needed.

### Stack Overflow (XGBoost, age group S1)

| Strategy | DPD | AUC | Meets threshold? |
|----------|-----|-----|-----------------|
| Baseline | 0.567 | 0.759 | ✗ |
| Reweighing | 0.341 | 0.730 | ✗ |
| Threshold calibration | ~0.000 | 0.759 | ✓ |

Reweighing substantially reduces DPD but cannot achieve compliance alone. Threshold calibration reaches near-zero DPD while preserving AUC — but applying different thresholds to different groups raises questions under equal-treatment principles in some jurisdictions.

### GitHub OSS (XGBoost)

| Strategy | DPD | AUC | Meets threshold? |
|----------|-----|-----|-----------------|
| Baseline | 0.024 | 0.562 | ✓ |
| Reweighing | 0.011 | 0.571 | ✓ |
| Threshold calibration | 0.001 | 0.562 | ✓ |

The GH model already meets compliance before any intervention.

---

## 5. Intersectional Analysis (Notebook 6)

Standard audits evaluate age and gender *separately*. **Intersectional fairness** (Foulds et al. 2020) asks whether belonging to multiple disadvantaged groups creates compounding harm beyond what either attribute alone would predict.

Crucially, the two datasets measure different sensitive attributes: **SO has age but no gender column; GH has gender but no age column.** They are therefore analysed separately — never combined — on the intersectional dimensions each actually contains.

- **SO:** age group × experience bracket → 6 subgroups (young_junior, young_mid, young_senior, experienced_junior, experienced_mid, experienced_senior). The `age_exp_pro` column was already constructed in NB2.
- **GH:** gender × experience bracket → 6 subgroups (man_junior, man_mid, man_senior, woman_junior, woman_mid, woman_senior). Experience bracket is derived here from `pro_experience_yrs`.

We compute the maximum pairwise DPD across all subgroup pairs and subtract the worst single-attribute DPD from NB3 to isolate the **compounding gap**. A positive compounding gap means intersectionality reveals harm that single-attribute audits would miss.

Results are in `outputs/so_intersectional_results.csv` and `outputs/gh_intersectional_results.csv`.

---

## 6. Regulatory Context: EU AI Act

Our threshold and methods have direct regulatory relevance. Under the **EU AI Act (Regulation 2024/1689)**, AI systems used in employment contexts are classified as high-risk (Annex III, category 4) and must:

- Document training data composition and known biases (Art. 10)
- Implement technical accuracy and robustness measures (Art. 15)
- Provide transparency reporting to affected individuals (Art. 13)
- Support human oversight mechanisms (Art. 14)

Our SO XGBoost model fails the |DPD| ≤ 0.10 threshold under all three sensitive definitions without mitigation, and still fails under reweighing alone. Only threshold calibration achieves compliance — highlighting the tension between regulatory requirements and the practical limits of fairness interventions.

---

## 7. Summary

| Finding | SO | GH |
|---------|----|----|
| Pre-existing data bias | High (DPD 0.31) | Moderate (DPD 0.13) |
| Model amplification | **Severe** (ratio 1.6–2.1) | None (ratio < 1) |
| Dominant proxy features | Years of experience | Years of experience |
| Mitigation achieves compliance | Only threshold calibration | All strategies |
| Intersectional dimension | Age × Experience | Gender × Experience |

**Open questions for further work:**
1. Would including age directly as a feature reduce or increase amplification?
2. How sensitive are results to the 80/20 train-test split?
3. Does the intersectional gap vary across different salary quantile thresholds?
4. How would counterfactual fairness (Kusner et al. 2017) compare to the strategies tested?