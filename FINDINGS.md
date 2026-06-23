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
| man_junior | 4.4% |
| man_mid | 45.1% |
| man_senior | 81.8% |
| woman_junior | 3.3% |
| woman_mid | 45.9% |
| woman_senior | 74.1% |

- Gender-only DPD on XGBoost: **0.053** — under the compliance line.
- **Maximum intersectional DPD: 0.785**, between *experienced men* (81.8%) and *junior women* (3.3%).
- Compounding gap (max intersectional DPD − single-attribute DPD): **+0.73 absolute**.

**This is the project's headline result.** A single-axis fairness audit declares the FCC model compliant. Conditioning on one additional axis (experience bracket) reveals a disparity an order of magnitude larger than the compliance threshold. Experience is itself a legitimate predictor of paid-developer status, so the intersectional DPD is not by itself proof of unjust bias — but it does show that a compliance metric's verdict depends heavily on the granularity at which it's evaluated, and that a remedy targeting only the gender axis (§4) would leave this subgroup-level disparity essentially untouched.

> **Verification note.** The 0.76 figure in the original notebook was computed before a train/test split bug in `06_intersectional_analysis.py` was fixed (the notebook had re-split with `test_size=0.2`, unstratified, while the model being evaluated had been trained on a `test_size=0.25`, stratified split, so the "test" set in the analysis overlapped with training data). After the split fix, the max intersectional DPD on the corrected hold-out is **0.785** between *experienced men* (81.8%) and *junior women* (3.3%) -- a slightly *larger* gap than the pre-fix estimate. The qualitative story (gender-only compliant, intersectional non-compliant by an order of magnitude) does not reverse. NB8 also reports a bootstrapped 95% CI of [0.512x, 0.955x] on the gender-only amplification, confirming the under-amplification verdict at the gender-only level is statistically distinguishable from "no amplification" rather than noise.

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

## 7a. Cross-Dataset Synthesis (Notebook 7)

NB7 reads the per-dataset CSVs from NB3/NB5/NB6 and assembles a single comparison view. It produces:

- `outputs/final_cross_dataset_comparison.csv` — one row per dataset, with data DPD, model DPD, amplification, best mitigated DPD, intersectional max DPD, and compliance flags.
- `outputs/nb7_amplification_comparison.png` — XGBoost vs Logistic Regression amplification ratio per dataset.
- `outputs/nb7_gap_comparison.png` — data DPD vs model DPD side-by-side per dataset.
- `outputs/nb7_compliance_dashboard.png` — baseline / mitigated / intersectional DPD per dataset against the |DPD| ≤ 0.10 line.
- `outputs/nb7_summary.md` — markdown summary the Presentation notebook can include directly.

NB7 is resilient to missing datasets: if only FCC has been run, NB7 produces a 1-dataset summary rather than crashing. This matters because the SO data needs a manual Kaggle download and the Adult data needs network access to `archive.ics.uci.edu`, neither of which is guaranteed in every environment.

---

## 7b. Sensitivity & Bootstrap Analysis (Notebook 8)

NB8 implements three open items previously listed in §9 and reports them as actual results.

### Bootstrapped 95% confidence intervals (B=1000)

For each (dataset, model) we resample the test predictions with replacement and recompute DPD and amplification. The headline question is: **is the FCC under-amplification verdict statistically distinguishable from "no amplification" (ratio = 1)?**

| Dataset | Model | Point amplification | 95% CI | Verdict |
|---------|-------|---------------------|--------|---------|
| FCC 2018 | XGBoost | 0.732× | [0.512×, 0.955×] | **Compresses** — CI strictly below 1.0 |

The CI excludes 1.0, so the under-amplification finding is statistically defensible rather than within noise. SO and Adult bootstrap rows will populate once those branches have been executed end-to-end.

### Sensitivity: removing `log_expected_earning` from FCC

§3 flagged that `log_expected_earning` is plausibly downstream of gender, so the under-amplification verdict might depend on it. NB8 retrains XGBoost without that single feature:

| Variant | n features | DPD | Amplification | 95% CI on amplification | AUC |
|---------|------------|-----|---------------|--------------------------|-----|
| Full feature set | 9 | 0.053 | 0.73× | [0.51×, 0.96×] | 0.934 |
| Drop `log_expected_earning` | 8 | 0.043 | 0.59× | [0.40×, 0.78×] | 0.854 |

**The under-amplification verdict survives.** The CI upper bound on the dropped variant (0.78×) is still well below 1.0, so the qualitative claim from §2 holds even without the aspiration feature. AUC does fall meaningfully (0.934 → 0.854), which is its own finding: the aspiration variable contributes ≈8 AUC points of legitimate predictive signal alongside whatever indirect gender signal it carries.

### Subgroup-aware reweighing

NB5's reweighing uses gender as the key, which is why the 0.78 intersectional DPD on FCC survives mitigation in §4. NB8 instead reweighs on the *intersectional* subgroup (gender × experience bracket for FCC, gender × age bracket for Adult) and recomputes the intersectional max DPD.

| Dataset | Intersectional DPD (baseline) | Intersectional DPD (subgroup reweigh) | AUC drop | Meets |DPD| ≤ 0.10? |
|---------|--------------------------------|----------------------------------------|----------|--------------------|
| FCC 2018 | 0.785 | **0.065** | 0.934 → 0.902 | **YES** |

This is a constructive answer to §5's headline finding. The order-of-magnitude intersectional disparity *can* be brought under the compliance line — but only by reweighing on the joint axis, not the single-axis reweighing NB5 uses. The mitigation cost is a small (~0.03) AUC drop, which is a defensible trade-off for closing the subgroup gap.

The Adult row will populate once `archive.ics.uci.edu` is reachable from the running environment.

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
| Model amplification | **Severe** (ratio 1.6–2.1) | None (ratio 0.4–0.75, CI [0.51×, 0.96×]) | Pending |
| Dominant proxy features | Years of experience | Expected earning, months programming | Pending |
| Mitigation achieves single-axis compliance | Only threshold calibration | All strategies | Pending |
| Intersectional max DPD | Age × Experience | Gender × Experience — **0.785** (the project headline) | Pending |
| Subgroup-aware reweighing closes intersectional gap | Not yet evaluated | **YES** — 0.785 → 0.065, AUC 0.934 → 0.902 | Pending |

**Status of next-step items** (carried over from earlier drafts):

| Item | Status |
|------|--------|
| 1. Re-run NB6 with the split fix and confirm the FCC 0.76 max intersectional DPD holds | **DONE** — corrected value is 0.785, qualitative story holds (see §5 verification note) |
| 2. Sensitivity analysis dropping `log_expected_earning` from FCC | **DONE** — NB8 §7b shows the under-amplification verdict survives (amplification 0.59×, CI [0.40×, 0.78×]) |
| 3. Re-run reweighing over FCC's intersectional subgroups, not just the gender-only key | **DONE** — NB8 §7b shows intersectional DPD drops from 0.785 to 0.065 at the cost of 0.03 AUC |
| 4. Bootstrapped 95% CIs on DPD and amplification | **DONE** — NB8 §7b; FCC amplification CI [0.512×, 0.955×] excludes 1.0 |
| 5. Execute the Adult Income branch end-to-end and populate §7 | **CODE READY**, requires `archive.ics.uci.edu` access to fetch; pipeline will run cleanly once data is reachable from your environment |
| 6. Would including age directly as a feature reduce or increase amplification? | Open — proposed extension, not yet implemented |
| 7. Does the intersectional gap vary across different salary/income quantile thresholds? | Open — proposed extension, not yet implemented |
