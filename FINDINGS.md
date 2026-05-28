# Project Findings: How Age and Gender Bias Amplify Inequality through ML Predictions

## Overview

This document summarises the key findings from our audit of two developer datasets — the **Stack Overflow Developer Survey 2024** (SO, ~65,000 respondents) and the **GitHub Open Source Survey 2017** (GH, ~5,400 respondents). We trained Logistic Regression and XGBoost classifiers to predict **high salary** (SO) and **OSS participation** (GH), then measured how much the models amplify pre-existing demographic gaps.

---

## 1. Do the datasets already contain bias before any model is trained?

Yes — both datasets show a measurable demographic gap in the *raw data*, before a single model is involved.

| Dataset | Sensitive attribute | Data gap (DPD) |
|---------|---------------------|----------------|
| Stack Overflow | Age group (young vs experienced) | 0.305 |
| GitHub OSS | Gender (man vs woman) | 0.131 |

The SO data gap is particularly large: young developers are already 30 percentage points less likely to earn above the median salary than experienced developers, simply reflecting labour-market inequality. The GH gap is smaller but still meaningful — women are about 13 points less likely to be classified as active OSS contributors.

---

## 2. Do ML models make things worse? (Bias Amplification)

The central research question is whether models *amplify* these gaps — i.e., whether the gap in predictions is larger than the gap in the raw data. We measure this with the **amplification ratio** (model DPD / data DPD). A ratio above 1.0 means the model makes things worse.

### Stack Overflow

| Sensitive definition | Model | Amplification ratio |
|----------------------|-------|---------------------|
| Age group (binary) | Logistic Regression | **2.07** |
| Age group (binary) | XGBoost | **1.86** |
| Age × YearsCodePro | Logistic Regression | **1.85** |
| Age × YearsCodePro | XGBoost | **1.86** |
| Age × YearsCode | Logistic Regression | **1.64** |
| Age × YearsCode | XGBoost | **1.67** |

All amplification ratios are well above 1.0. The Logistic Regression model nearly **doubles** the existing age gap. This means that even if the labour market is already unfair to young developers, the ML predictor makes the disparity significantly worse.

### GitHub OSS

| Model | Data gap | Model DPD | Amplification ratio |
|-------|----------|-----------|---------------------|
| Logistic Regression | 0.131 | 0.013 | **0.099** |
| XGBoost | 0.131 | 0.024 | **0.184** |

Interestingly, the GH models show the *opposite* pattern — the amplification ratio is below 1.0, meaning both models actually *reduce* the observed gender gap relative to raw data. This may reflect the smaller dataset size, weaker signal in features, or genuine differences in how gender is correlated with the predictive features.

**Key takeaway:** Bias amplification is not guaranteed — it depends heavily on the dataset, the features available, and the model type. But where it occurs (SO), it can be severe.

---

## 3. Which features drive biased predictions? (SHAP Analysis)

We used SHAP (SHapley Additive exPlanations) to identify which features contribute most to model decisions.

### Stack Overflow — top features by mean absolute SHAP value

| Feature | Mean |SHAP| |
|---------|--------|
| `years_code_pro` | 0.453 |
| `years_code` | 0.401 |
| `ed_level_enc` | 0.146 |
| `is_student` | 0.102 |
| `is_remote` | 0.090 |
| `devtype_developer_full_stack` | 0.069 |

The two experience features dominate by a large margin. This is a key insight: **the model is not directly using age as an input**, but age is strongly correlated with years of coding experience. The model learns to penalise young developers indirectly, through features that serve as proxies for age. This is a classic example of *proxy discrimination*, which standard fairness audits that only look at direct attribute use would miss.

### GitHub OSS — top features by mean absolute SHAP value

| Feature | Mean |SHAP| |
|---------|--------|
| `pro_experience_yrs` | 0.268 |
| `oss_experience_yrs` | 0.186 |
| `contributor_help_score` | 0.095 |
| `find_answers_score` | 0.082 |
| `receive_help_score` | 0.077 |

Again, experience features dominate — but here both professional and OSS-specific experience matter. The community-attitude scores (`find_answers_score`, `receive_help_score`, `contributor_help_score`) also appear, which may partially reflect the chilling effect of hostile environments on participation — a known barrier for women in open-source communities.

---

## 4. Can fairness mitigation strategies bring models into compliance?

We tested two mitigation strategies against a proposed compliance threshold of **|DPD| ≤ 0.10**:

- **Reweighing** (Kamiran & Calders 2012): adjust sample weights during training to equalise group representation.
- **Threshold calibration** (post-processing): apply different decision thresholds per group to equalise positive-prediction rates.

### Stack Overflow (XGBoost, age group S1)

| Strategy | DPD | AUC | Meets threshold? |
|----------|-----|-----|-----------------|
| Baseline | 0.567 | 0.759 | ✗ |
| Reweighing | 0.341 | 0.730 | ✗ |
| Threshold calibration | ~0.000 | 0.759 | ✓ |

Reweighing reduces the DPD substantially but still cannot bring the SO model into compliance on its own. Threshold calibration achieves near-zero DPD while preserving AUC — but at the cost of applying different decision boundaries to different groups, which raises its own fairness and legal questions under some regulatory frameworks.

### GitHub OSS (XGBoost)

| Strategy | DPD | AUC | Meets threshold? |
|----------|-----|-----|-----------------|
| Baseline | 0.024 | 0.562 | ✓ |
| Reweighing | 0.011 | 0.571 | ✓ |
| Threshold calibration | 0.001 | 0.562 | ✓ |

The GH model already meets the compliance threshold without any intervention. All three strategies keep DPD well below 0.10.

---

## 5. Intersectional Analysis (Notebook 6)

Standard fairness audits evaluate age and gender *separately*. However, **intersectional fairness** asks whether belonging to *both* a disfavoured age group *and* a disfavoured gender creates compounded disadvantage beyond what either attribute alone would predict.

We define four subgroups: `young_woman`, `young_man`, `experienced_woman`, `experienced_man`, and compute the maximum pairwise DPD across them.

This analysis (see `06_intersectional_analysis.py`) addresses the question:

> Is a young woman doubly penalised — more than a young man and more than an older woman?

Preliminary results and interpretation are in `outputs/so_intersectional_results.csv` and `outputs/gh_intersectional_results.csv`. If the **intersectional DPD exceeds the worst single-attribute DPD** from NB3, this constitutes evidence of compounding harm — a finding with direct relevance to EU AI Act Article 10 requirements around bias in training data, and to the broader discussion of algorithmic accountability for multiply-marginalised groups.

---

## 6. Regulatory Context: EU AI Act

Our compliance threshold (|DPD| ≤ 0.10) is not arbitrary. Under the **EU AI Act (Regulation 2024/1689)**, high-risk AI systems in employment and HR contexts (Annex III, category 4) must undergo mandatory conformity assessments including bias audits. The Act requires:

- Documentation of training data composition and known biases (Art. 10).
- Technical accuracy and robustness measures (Art. 15).
- Transparency reporting to affected individuals (Art. 13).

Our XGBoost model on Stack Overflow data fails the proposed |DPD| ≤ 0.10 threshold under all three sensitive definitions without mitigation, and still fails under reweighing. Only threshold calibration achieves compliance — but calibration-based approaches may conflict with equal-treatment requirements in some jurisdictions.

---

## 7. Summary and Open Questions

| Finding | SO | GH |
|---------|----|----|
| Pre-existing data bias | High (DPD 0.31) | Moderate (DPD 0.13) |
| Model amplification | **Severe** (ratio 1.6–2.1) | None (ratio < 1) |
| Dominant proxy features | Years of experience | Years of experience |
| Mitigation achieves compliance | Only threshold calibration | All strategies |
| Intersectional compounding | To be confirmed (NB6) | To be confirmed (NB6) |

**Open questions for further work:**

1. Would including age directly as a feature reduce or increase bias amplification?
2. How sensitive are the amplification results to the 80/20 train/test split?
3. Does the intersectional gap vary across different salary quantile thresholds (not just above/below median)?
4. How would a counterfactual fairness approach (Kusner et al. 2017) compare to the reweighing and threshold strategies tested here?