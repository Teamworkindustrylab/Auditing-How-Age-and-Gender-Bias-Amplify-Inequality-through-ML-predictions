"""
  config.py -- Shared constants for the bias-amplification audit pipeline.

  Every notebook (NB1-NB8) imports from here instead of copy-pasting.
  Changing a feature list or threshold in ONE place now propagates
  everywhere automatically.

  Contents
  --------
  - Feature column lists  (SO, FCC, Adult)
  - Sensitive-attribute definitions (SO multi-group)
  - DPD compliance threshold
  - Shared colour palette
  - _classify_age()  -- single canonical implementation used by NB1 & NB2
"""

import re
import numpy as np

# ---------------------------------------------------------------------------
# FEATURE COLUMN LISTS
# ---------------------------------------------------------------------------

SO_BASE_FEATURES: list[str] = [
    "ed_level_enc",
    "is_employed",
    "is_remote",
    "is_student",
    "years_code",
    "years_code_pro",
]

FCC_FEATURE_COLS: list[str] = [
    "months_programming",
    "hours_learning_per_week",
    "num_learning_resources",
    "attended_bootcamp",
    "is_under_employed",
    "is_ethnic_minority",
    "has_degree",
    "is_high_income_country",
    "log_expected_earning",
]

ADULT_BASE_FEATURES: list[str] = [
    "education_num",
    "hours_per_week",
    "log_capital_gain",
    "log_capital_loss",
    "is_married",
    "is_government_employee",
    "is_self_employed",
    "is_us",
]

# ---------------------------------------------------------------------------
# SENSITIVE-ATTRIBUTE DEFINITIONS  (Stack Overflow multi-group)
# ---------------------------------------------------------------------------

SENSITIVE_DEFS: dict[str, str] = {
    "S1_age_group":     "age_group",
    "S2_age_exp_pro":   "age_exp_pro",
    "S3_age_exp_total": "age_exp_total",
}

# ---------------------------------------------------------------------------
# COMPLIANCE THRESHOLD
# ---------------------------------------------------------------------------

DPD_THRESHOLD: float = 0.10  # |DPD| <= 0.10, consistent with fairness literature

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (used in all notebooks for consistent charts)
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "so":    "#f48024",   # Stack Overflow orange
    "fcc":   "#0a0a23",   # freeCodeCamp navy
    "adult": "#2e7d32",   # UCI Adult green
    "ok":    "#27ae60",   # compliance pass
    "bad":   "#e74c3c",   # compliance fail
    "warn":  "#f39c12",   # within 2x of threshold
    "bg":    "#fafafa",   # chart background
}

# ---------------------------------------------------------------------------
# SHARED AGE CLASSIFIER
# ---------------------------------------------------------------------------

def classify_age(age_val) -> str:
    """
    Canonical age classifier used by BOTH NB1 (EDA) and NB2 (feature
    engineering).  Returns 'young' or 'experienced' — no spaces, no
    brackets, consistent with the downstream group-matching logic in
    NB3-NB8.

    Handles:
      - SO ordinal bands  e.g. '25-34 years old', '35-44 years old'
      - FCC numeric ages  e.g. '27', 29.0
      - Missing / NaN / 'NA' / empty string  -> defaults to 'experienced'

    Threshold: age < 35 -> 'young',  age >= 35 -> 'experienced'
    """
    s = str(age_val).strip().lower()

    # Missing / sentinel values
    if not s or s in ("nan", "na", "none", ""):
        return "experienced"

    # Explicit under-18 strings (SO has "Under 18 years old")
    if "under 18" in s or "< 18" in s:
        return "young"

    # Numeric age (FCC stores raw integers e.g. "27")
    try:
        n = float(s)
        if 5 <= n <= 120:                  # guard against bad data entries
            return "young" if n < 35 else "experienced"
    except ValueError:
        pass

    # Ordinal band — extract all integers and use the UPPER bound
    # so "25-34 years old" -> upper=34 -> 'young'
    #    "35-44 years old" -> upper=44 -> 'experienced'
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return "experienced"
    return "young" if max(nums) <= 34 else "experienced"
