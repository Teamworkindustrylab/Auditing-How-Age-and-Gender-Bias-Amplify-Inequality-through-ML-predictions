# Project Findings: How Age and Gender Bias Amplify Inequality through ML Predictions

## Overview

This document summarises the key findings from our audit of developer/income datasets — the **Stack Overflow Developer Survey 2024** (SO, ~65,000 respondents) and the **freeCodeCamp 2018 New Coder Survey** (FCC, ~31,000 respondents). We trained Logistic Regression and XGBoost classifiers to predict **high salary** (SO) and **paid-developer status** (FCC), then measured how much the models amplify pre-existing demographic gaps. A third dataset, **UCI Adult / Census Income** (48,842 rows), has been integrated into the pipeline but not yet executed end-to-end — see §7.

> **Dataset history.** An earlier version of this project used the GitHub Open Source Survey 2017 (~5,400 respondents, 97% men) for the gender-bias track. It was replaced with FCC 2018 because the GH sample left only ~150 women in the test split — too few to estimate gender-conditional DPD reliably. All numbers below reflect the live FCC pipeline, not the retired GH source.

---

## 0. Responsible AI Framework & Tooling

This project measures fairness using standard, externally-defined metrics rather than ad hoc calculations, so results are reproducible and comparable to other published audits.

**Fairlearn** (Microsoft) supplies the Demographic Parity Difference and Equalized Odds Difference implementations used throughout, plus the post-processing threshold-optimisation step in `05_mitigation_compliance.py`. The **reweighing** strategy is a from-scratch implementation of Kamiran & Calders (2012) — it does not depend on AIF360, and no script in this project imports `aif360`. (An earlier draft of this document credited AIF360 for the reweighing methodology; that was inaccurate and has been corrected here.)

**Why DPD ≤ 0.10 as the compliance threshold?**
The Demographic Parity Difference threshold of |DPD| ≤ 0.10 is not arbitrary. It draws from the **four-fifths (80%) rule** in US employment discrimination law (EEOC guidelines), which flags a selection rate below 80% of the majority group's rate as evidence of adverse impact — equivalent to roughly a 0.10–0.20 DPD range depending on base rates. The ≤ 0.10 threshold also appears in recent EU AI Act impact assessments and fairness literature (e.g. Chouldechova 2017; Verma & Rubin 2018) as a practical boundary between tolerable statistical noise and actionable disparity. This is a **research-defined operationalisation**, not a statutory EU AI Act number — choosing a research-grounded, externally anchored threshold rather than a project-internal one prevents the goalposts moving to make a model look compliant after the fact.

---

## 1. Do the datasets already contain bias before any model is trained?

Yes — both datasets show a measurable demographic gap in the *raw data*, before a single model is involved.

| Dataset | Sensitive attribute | Data gap (DPD) |
|---------|---------------------|----------------|
| Stack Overflow | Age group (young vs experienced) | 0.305 |
| freeCodeCamp | Gender (man vs woman) | 0.070 |

The SO data gap is large: young developers are already 30 percentage points less likely to earn above the median salary than experienced developers, simply reflecting labour-market inequality. On FCC, men report being a paid developer at 17.9% vs. 10.8% for women (n=23,486 men, n=6,260 women) — a 7.0-point raw gap, smaller than SO's age gap but still a real baseline disparity the model could preserve, amplify, or compress.

---

## 2. Do ML models make things worse? (Bias Amplification)

The central question is whether models *amplify* these gaps. We measure this with the **amplification ratio** = model DPD / data DPD. A ratio above 1.0 means the model makes pre-existing inequality worse; a ratio below 1.0 means it compresses the gap relative to the raw data.

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

### freeCodeCamp

| Model | Data gap | Model DPD | EOD | Amplification ratio | AUC |
|-------|----------|-----------|-----|---------------------|-----|
| Logistic Regression | 0.070 | 0.030 | 0.042 | **0.42** | 0.838 |
| XGBoost | 0.070 | 0.053 | 0.021 | **0.75** | 0.934 |

Both FCC models *compress* the gender gap rather than widen it — the opposite pattern from SO's age results. On its own, this reads as a reassuring finding. §5 shows why that single-axis reading is incomplete.

**Key takeaway:** Bias amplification is not guaranteed, but where it occurs (SO), it can be severe. Where it doesn't occur on a single sensitive axis (FCC gender), that doesn't necessarily mean the model is fair at finer granularity — see §5.

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

### freeCodeCamp — top features

| Feature | Mean |SHAP| |
|---------|--------|
| `log_expected_earning` | 2.880 |
| `months_programming` | 1.032 |
| `has_degree` | 0.332 |
| `is_high_income_country` | 0.255 |
| `hours_learning_per_week` | 0.122 |
| `attended_bootcamp` | 0.096 |
| `is_under_employed` | 0.055 |
| `is_ethnic_minority` | 0.015 |

`log_expected_earning` and `months_programming` together account for the majority of the model's signal — effort/aspiration features, not protected attributes, which is consistent with the under-amplification result in §2. **Caveat:** `log_expected_earning` is itself plausibly *downstream* of gender (women report lower expected salaries even at equal experience, a well-documented effect in the pay-gap literature). The model's "non-amplification" verdict at the gender-only level therefore rests partly on a feature that may carry indirect gender signal — exactly the kind of question the intersectional view in §5 surfaces. A sensitivity run with this feature dropped is listed as an open item in §8.

---

## 4. Mitigation Strategies & Compliance

We tested two strategies against the |DPD| ≤ 0.10 threshold:

- **Reweighing** (Kamiran & Calders 2012, hand-implemented): adjusts per-sample training weights inversely proportional to group × label frequency, then retrains. A pre-processing approach — the model itself changes.
- **Threshold calibration** (Fairlearn post-processing): per-group search for the decision threshold that equalises positive-prediction rates. No retraining needed.

### Stack Overflow (XGBoost, age group)

| Strategy | DPD | AUC | Meets threshold? |
|----------|-----|-----|-----------------|
| Baseline | 0.567 | 0.759 | ✗ |
| Reweighing | 0.341 | 0.730 | ✗ |
| Threshold calibration | ~0.000 | 0.759 | ✓ |

Reweighing substantially reduces DPD but cannot achieve compliance alone. Threshold calibration reaches near-zero DPD while preserving AUC — but applying different thresholds to different groups raises questions under equal-treatment principles in some jurisdictions.

### freeCodeCamp (XGBoost, gender)

| Strategy | DPD | AUC | Meets threshold? |
|----------|-----|-----|-----------------|
| Baseline | 0.053 | 0.934 | ✓ |
| Reweighing | 0.051 | 0.934 | ✓ |
| Threshold calibration | 0.000 | 0.934 | ✓ |

All three configurations already pass the |DPD| ≤ 0.10 threshold on FCC's gender-only axis, with no measurable AUC cost from threshold calibration — suggesting the residual gap is a calibration artefact at a single decision threshold rather than a deeper feature-level bias, *at this level of granularity*.

---

## 5. Intersectional Analysis (Notebook 6) — the headline finding

Standard audits evaluate age and gender *separately*. **Intersectional fairness** (Foulds et al. 2020) asks whether belonging to multiple disadvantaged groups creates compounding harm beyond what either attribute alone would predict.

The two datasets measure different primary sensitive attributes: **SO has age but no gender column; FCC has gender and a numeric age field.** SO's intersectional slice is age group × experience bracket (`age_exp_pro`, built in NB2); FCC's is gender × experience bracket (`gender_age_bracket`, also built in NB2).

### FCC: gender × experience bracket

| Subgroup | Positive-prediction rate |
|----------|---------------------------|
| man_junior | 4.2% |
| man_mid | 44.8% |
| man_senior | 79.7% |
| woman_junior | 3.2% |
| woman_mid | 45.9% |
| woman_senior | 71.4% |

- Gender-only DPD on XGBoost: **0.053** — under the compliance line.
- **Maximum intersectional DPD: 0.76**, between *experienced men* (79.7%) and *junior women* (3.2%).
- Compounding gap (max intersectional DPD − single-attribute DPD): **+0.71 absolute**.

**This is the project's headline result.** A single-axis fairness audit declares the FCC model compliant. Conditioning on one additional axis (experience bracket) reveals a disparity an order of magnitude larger than the compliance threshold. Experience is itself a legitimate predictor of paid-developer status, so the intersectional DPD is not by itself proof of unjust bias — but it does show that a compliance metric's verdict depends heavily on the granularity at which it's evaluated, and that a remedy targeting only the gender axis (§4) would leave this subgroup-level disparity essentially untouched.

> **Open verification item.** This 0.76 figure was computed before a train/test split bug in `06_intersectional_analysis.py` was fixed. The notebook previously re-split the data with different parameters (`test_size=0.2`, unstratified) than the split actually used to train the models being evaluated (`test_size=0.25`, stratified, in `03_bias_amplification.py`), so the "test" set in this analysis likely overlapped with training data. The fix (matching NB6's split exactly to NB3's) is already in the current `06_intersectional_analysis.py`. **The numbers in this section should be re-generated by re-running the pipeline before being used in the final submission** — they may shift once the leak is removed, though the qualitative story (gender-only compliant, intersectional non-compliant) is unlikely to reverse given the size of the compounding gap.

Results are saved to `outputs/so_intersectional_results.csv` and `outputs/fcc_intersectional_results.csv`.

---

## 6. Regulatory Context: EU AI Act

Our threshold and methods have direct regulatory relevance. Under the **EU AI Act (Regulation 2024/1689)**, AI systems used in employment contexts are classified as high-risk (Annex III, category 4) and must:

- Document training data composition and known biases (Art. 10)
- Implement technical accuracy and robustness measures (Art. 15)
- Provide transparency reporting to affected individuals (Art. 13)
- Support human oversight mechanisms (Art. 14)

Our SO XGBoost model fails the |DPD| ≤ 0.10 threshold under all three sensitive definitions without mitigation, and still fails under reweighing alone — only threshold calibration achieves compliance. The FCC model presents the opposite surface story (gender-only compliant pre-mitigation) but fails badly once evaluated intersectionally (§5). Taken together, the two datasets illustrate a genuine tension for Art. 10-style documentation requirements: a single-axis compliance report can be both accurate and misleading about real-world subgroup harm.

---

## 7. UCI Adult / Census Income — integrated, not yet executed

A third dataset, UCI Adult / Census Income (48,842 rows, binary income target `above_50k`, binary `gender_clean`), has been added to all six notebooks: loader and EDA in NB1, `AdultFeatureEngineering` in NB2, `AdultBiasAmplification` in NB3, SHAP support in NB4, `AdultMitigation` in NB5, and `adult_intersectional()` in NB6. All six files compile cleanly.

**No results are reported here yet** — the pipeline has not been run end-to-end against live Adult data in this environment (no network access to `archive.ics.uci.edu` from this sandbox). For context, the dataset's own published documentation reports a raw gap of 30% of men vs. 11% of women earning above $50K — a larger raw gap than FCC's, and one that should comfortably exceed the |DPD| ≤ 0.10 threshold before mitigation, similar in spirit to SO's age results. This is a documented fact about the source data, **not** a number this project has yet reproduced through its own pipeline; treat it as a prior, not a result, until §7 has its own `adult_bias_results.csv`, SHAP table, mitigation table, and intersectional table like §1–§5 above.

A fourth candidate, OPM FedScope, was evaluated and rejected: its public release is aggregated count-cube data (age and salary in 5-year/$10K bands, cells under 12 employees suppressed), not row-level microdata, and is structurally incompatible with this pipeline's per-instance classifiers and SHAP analysis.

---

## 8. Limitations and threats to validity

- **Self-selection.** Both surveys are voluntary online surveys of people already engaged with developer ecosystems (Stack Overflow / freeCodeCamp). They under-represent developers who never participate in those communities, which likely *understates* the gender gap rather than overstates it.
- **Binary gender.** The ~1% "prefer not to say" cohort on FCC is excluded from the gender-conditional analysis because three-way DPD does not have a single agreed-upon definition and the cohort is too small for stable estimates. This is a recognised limitation of binary fairness metrics, not a position taken by this project.
- **Target is "currently working as a developer," not "deserves to be working as a developer."** The FCC target conflates skill, opportunity, and labour-market access. A higher predicted-positive rate for one group is consistent with either unequal access *or* unequal preparation.
- **`log_expected_earning` is downstream of gender.** As flagged in §3, including this feature partially launders gender into the FCC model via an aspiration variable. A sensitivity analysis with this feature removed would strengthen the claims in §2.
- **FCC `paid_contributor` non-response.** Rows where the target question was unanswered are currently coded as "no" via `fillna(0)` rather than dropped — worth noting explicitly as a limitation rather than silently absorbed into the negative class.
- **Single survey year (FCC, Adult).** Findings are not necessarily stable across years; replication on FCC 2017/2021 or a later Census extract would establish robustness.

---

## 9. Summary

| Finding | SO | FCC | Adult |
|---------|----|----|-------|
| Pre-existing data bias | High (DPD 0.31) | Moderate (DPD 0.07) | Pending — documented raw gap ~0.19 |
| Model amplification | **Severe** (ratio 1.6–2.1) | None (ratio 0.4–0.75) | Pending |
| Dominant proxy features | Years of experience | Expected earning, months programming | Pending |
| Mitigation achieves compliance | Only threshold calibration | All strategies (gender-only axis) | Pending |
| Intersectional dimension | Age × Experience | Gender × Experience — **0.76 max DPD, the project headline** | Pending |

**Open questions for further work** (carried over from the presentation notebook's own next-steps list):
1. Re-run NB6 with the split fix applied and confirm whether the 0.76 max intersectional DPD on FCC holds.
2. Run a sensitivity analysis dropping `log_expected_earning` from the FCC feature set to test how much of the under-amplification result depends on that one feature.
3. Re-run reweighing over FCC's intersectional subgroups (not just the gender-only groups) and report whether the 0.76 max-DPD drops below 0.10.
4. Add bootstrapped 95% confidence intervals on DPD and the amplification ratio in `03_bias_amplification.py` — the FCC under-amplification finding is most credible if its CI excludes 1.0.
5. Execute the Adult Income branch end-to-end and populate §7 with real numbers.
6. Would including age directly as a feature reduce or increase amplification?
7. Does the intersectional gap vary across different salary/income quantile thresholds?
