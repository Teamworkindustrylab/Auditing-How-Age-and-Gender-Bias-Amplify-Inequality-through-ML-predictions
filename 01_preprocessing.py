"""

  NOTEBOOK 1 -- DATA INGESTION & EXPLORATORY DATA ANALYSIS

---- >> preprocessing.py contains data loading, feature engineering, bias baseline

Datasets:
  Dataset 1: Stack Overflow Developer Survey 2024
  Dataset 2: freeCodeCamp New Coder Survey 2018  (REPLACEMENT FOR GH OSS 2017)

  NOTE: Dataset 2 change-log  (week of 2026-06-09)

  The GitHub Open Source Survey 2017 was replaced because its gender
  distribution was 97 % men / 3 % women, far too skewed to support a
  meaningful audit of gender-conditional outcomes: with ~150 women in
  the test split, every DPD estimate carried huge variance and any
  "mitigation" result was statistical noise.

  The freeCodeCamp 2018 New Coder Survey is a much better fit:
    - Same evaluation criteria apply unchanged (DPD, EOD, amplification
      ratio, intersectional max-gap, |DPD| <= 0.10 compliance threshold).
    - Same dataset family (a developer-focused community survey), so
      the project's "developer reputation" narrative is preserved.
    - Same loading pattern (single CSV on raw.githubusercontent.com).
    - 31,226 respondents -- ~6x the size of the GH OSS sample.
    - Gender distribution 78 % men / 21 % women (clean), which leaves
      ~1,300 women in the test split and makes every group-conditional
      statistic reliable.
    - Includes age, so we can also evaluate gender x age intersectional
      analysis on dataset 2 (which the GH OSS dataset could not support).
    - Open Database License (ODbL) -- same licence as Stack Overflow.


The notebook is organized as follows:
  - Path constants
  - reweigh_samples()
  - StackOverflowPipeline        (unchanged)
  - FCCSurveyPipeline            (replaces GitHubOSSSurveyPipeline)


Other parts of the project (model training, SHAP, mitigation, plotting) live in
separate modules and import from here.


SETUP

1. Download SO survey from: https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024
                        1.1. "Download Full Data Set" -> unzip
                        1.2. use survey_results_public.csv  (65,439 rows)

   Place SO survey at:  data/survey_results_public.csv

2. freeCodeCamp 2018 New Coder Survey loads automatically from the web.

3. Run requirements.txt to install dependencies: pip install -r requirements.txt


Outputs (saved to data/raw_eda/)

  so_eda_summary.csv          -- per-column null-rate, dtype, n_unique
  so_age_salary_dist.png      -- salary distribution by age group
  so_devtype_dist.png         -- DevType frequency chart
  so_years_dist.png           -- YearsCode / YearsCodePro distributions
  fcc_eda_summary.csv
  fcc_gender_paid_dist.png    -- working-as-developer rate by gender
=============================================================================
"""

import csv
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import PALETTE, classify_age   # shared constants & age classifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# Paths

SO_PATH    = "data/survey_results_public.csv"
FCC_URL    = ("https://raw.githubusercontent.com/freeCodeCamp/"
              "2018-new-coder-survey/master/raw-data/2018-new-coder-survey.csv")
FCC_PATH   = "data/fcc_raw.csv"
ADULT_PATH = "data/adult_raw.csv"
OUT_DIR    = "data/raw_eda"
os.makedirs(OUT_DIR, exist_ok=True)

# PALETTE imported from config.py


# SECTION 1 -- STACK OVERFLOW 2024  (UNCHANGED FROM PREVIOUS VERSION)

# 1.1  Parsing raw CSV
"""
The SO CSV wraps every data row in an outer quote. Some rows contain
unescaped inner quotes that break standard parsers, so we read it with
csv.reader and extract the columns we need by positional index.

Columns used in this project:
    row[0] -> inner CSV -> Age[2], Employment[3], RemoteWork[4],
                         EdLevel[7], YearsCode[?], YearsCodePro[?], DevType[?]
    row[-2] -> ConvertedCompYearly
"""
def _parse_so(path: str) -> pd.DataFrame:
    print(f"[SO] Parsing {path} ...")

    with open(path, "r", encoding="latin-1", errors="replace") as f:
        raw_header = f.readline().strip()

    try:
        header_cols = list(csv.reader([raw_header]))[0]
    except Exception:
        header_cols = raw_header.split(",")

    def _find(name):
        try:
            return header_cols.index(name)
        except ValueError:
            return None

    idx = {
        "Age":                 _find("Age"),
        "Employment":          _find("Employment"),
        "RemoteWork":          _find("RemoteWork"),
        "EdLevel":             _find("EdLevel"),
        "YearsCode":           _find("YearsCode"),
        "YearsCodePro":        _find("YearsCodePro"),
        "DevType":             _find("DevType"),
        "ConvertedCompYearly": _find("ConvertedCompYearly"),
    }
    print(f"[SO] Column positions detected: {idx}")

    records = []
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if not row:
                continue
            try:
                inner = list(csv.reader([row[0]]))[0] if row[0] else []
            except Exception:
                inner = []

            def _g(lst, i, fallback=""):
                try:    return lst[i] if i is not None else fallback
                except: return fallback

            comp = ""
            try:    comp = row[-2]
            except: pass

            records.append({
                "Age":                 _g(inner, idx["Age"]),
                "Employment":          _g(inner, idx["Employment"]),
                "RemoteWork":          _g(inner, idx["RemoteWork"]),
                "EdLevel":             _g(inner, idx["EdLevel"]),
                "YearsCode":           _g(inner, idx["YearsCode"]),
                "YearsCodePro":        _g(inner, idx["YearsCodePro"]),
                "DevType":             _g(inner, idx["DevType"]),
                "ConvertedCompYearly": comp,
            })

    df = pd.DataFrame(records)
    print(f"[SO]   Rows parsed: {len(df):,}")
    return df


def load_so(path: str = SO_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[SO] File not found: {path}\n"
            "  Download from https://survey.stackoverflow.co/2024/\n"
            "  -> 'Download Full Data Set' -> unzip\n"
            "  -> place survey_results_public.csv at data/survey_results_public.csv"
        )
    return _parse_so(path)


# 1.2 EDA

def eda_so(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("SO 2024 -- EDA")
    print("="*60)
    print(f"\n  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    null_rates = (df.replace("", np.nan).isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        "dtype":     df.dtypes,
        "n_unique":  df.nunique(),
        "null_pct":  null_rates,
        "sample":    df.iloc[0],
    })
    summary.to_csv(f"{OUT_DIR}/so_eda_summary.csv")
    print(f"\n  Null rates (%):\n{null_rates.to_string()}")

    comp = pd.to_numeric(
        df["ConvertedCompYearly"].replace({"NA": np.nan, "": np.nan}),
        errors="coerce"
    )
    valid_comp = comp.dropna()
    print(f"\n  Salary rows with valid value : {len(valid_comp):,} "
          f"({len(valid_comp)/len(df)*100:.1f}%)")
    print(f"  Median salary               : ${valid_comp.median():,.0f}")
    print(f"  Salary range (p1-p99)       : "
          f"${valid_comp.quantile(.01):,.0f} - ${valid_comp.quantile(.99):,.0f}")

    print(f"\n  Age value counts:\n{df['Age'].value_counts().head(10).to_string()}")
    print(f"\n  YearsCode value counts:\n"
          f"{df['YearsCode'].value_counts().head(10).to_string()}")
    print(f"\n  YearsCodePro value counts:\n"
          f"{df['YearsCodePro'].value_counts().head(10).to_string()}")

    devtype_all = (
        df["DevType"].dropna()
          .str.split(";")
          .explode()
          .str.strip()
          .value_counts()
    )
    print(f"\n  DevType (top 15):\n{devtype_all.head(15).to_string()}")
    return summary


# 1.3 SO plots for age/salary, DevType, and years of coding experience.

# Age classification is provided by config.classify_age (shared with NB2).
# The local alias below keeps the rest of this file unchanged.
_classify_age = classify_age


def plot_so_salary_by_age(df: pd.DataFrame):
    comp = pd.to_numeric(
        df["ConvertedCompYearly"].replace({"NA": np.nan, "": np.nan}),
        errors="coerce"
    )
    age_group = df["Age"].apply(_classify_age)
    plot_df = pd.DataFrame({"salary": comp, "age_group": age_group}).dropna()
    plot_df = plot_df[(plot_df["salary"] >= plot_df["salary"].quantile(.01)) &
                      (plot_df["salary"] <= plot_df["salary"].quantile(.99))]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    for grp, color in [("young", PALETTE["so"]),
                       ("experienced", "#1a5276")]:
        sub = plot_df[plot_df["age_group"] == grp]["salary"]
        if len(sub):
            ax.hist(sub, bins=40, color=color, alpha=0.6, label=f"{grp} (n={len(sub):,})")
    ax.set_xlabel("Annual compensation (USD)")
    ax.set_ylabel("Count")
    ax.set_title("SO 2024 -- Salary distribution by age group", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = f"{OUT_DIR}/so_age_salary_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SO]   Saved -> {out}")


def plot_so_devtype(df: pd.DataFrame):
    devtype_all = (
        df["DevType"].dropna()
          .str.split(";")
          .explode()
          .str.strip()
          .value_counts()
          .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=PALETTE["bg"])
    devtype_all.sort_values().plot.barh(ax=ax, color=PALETTE["so"], alpha=0.85)
    ax.set_xlabel("Count")
    ax.set_title("SO 2024 -- Top 15 DevTypes", fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/so_devtype_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SO]   Saved -> {out}")


def plot_so_years(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("SO 2024 -- Coding Experience Distributions", fontweight="bold")

    order = ["Less than 1 year", "1 to 2 years", "2 to 3 years",
             "3 to 5 years", "6 to 8 years", "9 to 11 years",
             "12 to 14 years", "15 to 17 years", "18 to 20 years",
             "21 to 23 years", "24 to 26 years", "27 to 29 years",
             "30 to 32 years", "33 to 35 years", "36 to 38 years",
             "39 to 41 years", "42 to 44 years", "45 to 47 years",
             "48 to 50 years", "More than 50 years"]

    for ax, col, title in [
        (axes[0], "YearsCode",    "Years Coding (total)"),
        (axes[1], "YearsCodePro", "Years Coding (professional)"),
    ]:
        vc = df[col].replace({"NA": np.nan, "": np.nan}).dropna().value_counts()
        vc_ordered = pd.Series(
            {k: vc.get(k, 0) for k in order if k in vc.index or vc.get(k, 0) > 0}
        )
        vc_ordered.plot.bar(ax=ax, color=PALETTE["so"], alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)

    plt.tight_layout()
    out = f"{OUT_DIR}/so_years_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SO]   Saved -> {out}")


# SECTION 2 -- freeCodeCamp NEW CODER SURVEY 2018 
# -----------------------------------------------------------------------
# Loaded from the official freeCodeCamp GitHub repository as a single
# CSV. 31,226 respondents, 136 columns, Open Database License (ODbL).

GENDER_COL          = "What's your gender?"
AGE_COL             = "How old are you?"
DEVELOPER_COL       = "Are you already working as a software developer?"
EMPLOYMENT_COL      = "Regarding employment status, are you currently..."
MONTHS_PROG_COL     = "About how many months have you been programming for?"
HOURS_LEARN_COL     = "About how many hours do you spend learning each week?"
EXPECTED_EARN_COL   = ("About how much money do you expect to earn per year "
                       "at your first developer job (in US Dollars)?")
COUNTRY_COL         = "Which country do you currently live in?"
ETHNIC_MINORITY_COL = "Are you an ethnic minority in your country?"
UNDER_EMPLOYED_COL  = "Do you consider yourself under-employed?"
BOOTCAMP_COL        = "Have you attended a full-time coding bootcamp?"
DEGREE_COL          = "What's the highest degree or level of school you have completed?"


def load_fcc() -> pd.DataFrame:
    """
    Load the FCC 2018 New Coder Survey, caching to data/fcc_raw.csv
    so subsequent runs don't re-download.
    """
    if os.path.exists(FCC_PATH):
        print(f"\n[FCC] Loading from cache: {FCC_PATH}")
        return pd.read_csv(FCC_PATH, low_memory=False)

    print(f"[FCC] Downloading from raw.githubusercontent.com ...")
    try:
        df = pd.read_csv(FCC_URL, low_memory=False)
        os.makedirs("data", exist_ok=True)
        df.to_csv(FCC_PATH, index=False)
        print(f"[FCC] Cached -> {FCC_PATH}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"[FCC] Download failed: {e}\n"
            f"  URL: {FCC_URL}\n"
            f"  Check your network connection or place the CSV manually at {FCC_PATH}"
        )


def eda_fcc(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("FCC 2018 -- EDA")
    print("="*60)
    print(f"\n  Shape : {df.shape}")

    # Show columns we will actually use rather than dumping all 136
    used_cols = [GENDER_COL, AGE_COL, DEVELOPER_COL, EMPLOYMENT_COL,
                 MONTHS_PROG_COL, HOURS_LEARN_COL, EXPECTED_EARN_COL,
                 COUNTRY_COL, ETHNIC_MINORITY_COL, UNDER_EMPLOYED_COL,
                 BOOTCAMP_COL, DEGREE_COL]
    used_cols = [c for c in used_cols if c in df.columns]
    sub = df[used_cols]

    null_rates = (sub.replace("", np.nan).isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        "dtype":    sub.dtypes,
        "n_unique": sub.nunique(),
        "null_pct": null_rates,
    })
    summary.to_csv(f"{OUT_DIR}/fcc_eda_summary.csv")
    print(f"\n  Null rates for used columns (%):\n{null_rates.to_string()}")

    print(f"\n  Gender distribution:\n"
          f"{df[GENDER_COL].value_counts(dropna=False).to_string()}")
    print(f"\n  Gender share (%):\n"
          f"{(df[GENDER_COL].value_counts(normalize=True) * 100).round(2).to_string()}")

    print(f"\n  Target: 'Are you already working as a software developer?'")
    print(df[DEVELOPER_COL].value_counts(dropna=False).to_string())

    age = pd.to_numeric(df[AGE_COL], errors="coerce")
    age = age[(age >= 14) & (age <= 80)]
    print(f"\n  Age stats (filtered 14-80): "
          f"n={len(age):,}, mean={age.mean():.1f}, median={age.median():.0f}")

    months = pd.to_numeric(df[MONTHS_PROG_COL], errors="coerce")
    print(f"  Months programming: median={months.median():.0f}, "
          f"p90={months.quantile(.9):.0f}")

    print(f"\n  Top 10 countries:\n"
          f"{df[COUNTRY_COL].value_counts().head(10).to_string()}")

    return summary


def plot_fcc_gender_paid(df: pd.DataFrame):
    """
    Two-panel chart:
      Left  - gender distribution
      Right - working-as-developer rate by gender.
    """
    def _gender(g):
        g = str(g).strip()
        if g in ("Male", "Man"):     return "man"
        if g in ("Female", "Woman"): return "woman"
        return "other"

    tmp = df.copy()
    tmp["gender_clean"] = tmp[GENDER_COL].apply(_gender)
    tmp["is_dev"] = pd.to_numeric(tmp[DEVELOPER_COL], errors="coerce").fillna(0).astype(int)

    # Drop "other" for the rate plot since downstream analysis is binary man/woman
    tmp_binary = tmp[tmp["gender_clean"].isin(["man", "woman"])]
    stats = tmp_binary.groupby("gender_clean")["is_dev"].agg(["mean", "count"])
    print(f"\n[FCC]   Working-as-developer rates by gender:\n{stats.to_string()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle("FCC 2018 -- Gender Distribution & Working-as-Developer Rate",
                 fontweight="bold")

    tmp["gender_clean"].value_counts().plot.bar(
        ax=axes[0], color=PALETTE["fcc"], alpha=0.85
    )
    axes[0].set_title("Gender distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", labelrotation=0)
    for i, v in enumerate(tmp["gender_clean"].value_counts()):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

    stats["mean"].plot.bar(ax=axes[1], color=PALETTE["fcc"], alpha=0.85)
    axes[1].set_title("Working-as-developer rate by gender")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, max(0.3, stats["mean"].max() * 1.4))
    axes[1].tick_params(axis="x", labelrotation=0)
    for i, v in enumerate(stats["mean"]):
        axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10,
                     fontweight="bold")

    plt.tight_layout()
    out = f"{OUT_DIR}/fcc_gender_paid_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FCC]   Saved -> {out}")


# SECTION 3 -- UCI ADULT / CENSUS INCOME (NEW)
# -----------------------------------------------------------------------
# 48,842 respondents, 14 attributes, donated by Barry Becker from the
# 1994 Census database (UCI id=20, the combined train+test "Census
# Income" release -- larger than the classic 32,561-row "Adult" id=2).
# Binary target: income >$50K/yr. Binary sensitive attribute: sex.
# Loaded via a chain of fallbacks so the rest of the pipeline doesn't
# get stuck if any one source is down:
#   1. Local cache file (data/adult_raw.csv)
#   2. ucimlrepo package -> archive.ics.uci.edu
#   3. sklearn fetch_openml -> openml.org
#   4. Stable raw GitHub mirror (jbrownlee/Datasets)
# Whichever path wins, the columns are normalised the same way and the
# result is cached so subsequent runs skip the network entirely.

# Standard 15-column Adult schema, used when the source doesn't ship headers
_ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]

_ADULT_GITHUB_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
)


def _normalise_adult(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten column names to snake_case and clean the target column.
    The target column comes through as "income", ">50K", or "class"
    depending on the source, and the official test split has a trailing
    period (">50K.") which we strip here so downstream string matches
    don't silently drop half the positive class.
    """
    df = df.copy()
    df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                  for c in df.columns]

    # Find the target column. Order matters: an EXACT-name match wins
    # first (covers github / openml which ship the column named "income").
    # If that's absent, fall back to value-based detection (look for the
    # ">50K" / "<=50K" pattern in the last column). The earlier version
    # of this function matched any column containing "income" OR "class"
    # OR "50", which accidentally renamed "workclass" to "income" and
    # produced an "income.1" duplicate for the real target.
    if "income" in df.columns:
        target_col = "income"
    elif "class" in df.columns:
        target_col = "class"
    else:
        # Last-column fallback: scan for >50K / <=50K markers
        last = df.columns[-1]
        sample = df[last].astype(str).str.strip().str.rstrip(".")
        if sample.isin(["<=50K", ">50K"]).any():
            target_col = last
        else:
            target_col = last  # accept it, but warn
            print(f"  [ADULT] Warning: couldn't confidently identify target "
                  f"column; using last column '{last}'.")

    df[target_col] = df[target_col].astype(str).str.strip().str.rstrip(".")
    if target_col != "income":
        df = df.rename(columns={target_col: "income"})
    return df


def _try_load_adult_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("  [ADULT] ucimlrepo not installed -- skipping that path.")
        return None
    print("  [ADULT] Trying ucimlrepo (id=20) ...")
    try:
        census_income = fetch_ucirepo(id=20)
        df = pd.concat(
            [census_income.data.features, census_income.data.targets], axis=1
        )
        return df
    except Exception as e:
        print(f"  [ADULT] ucimlrepo path failed ({type(e).__name__}).")
        return None


def _try_load_adult_openml() -> pd.DataFrame | None:
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        return None
    print("  [ADULT] Trying sklearn fetch_openml('adult') ...")
    try:
        bundle = fetch_openml(name="adult", version=2, as_frame=True,
                               parser="auto")
        df = pd.concat([bundle.data, bundle.target.rename("income")], axis=1)
        return df
    except Exception as e:
        print(f"  [ADULT] openml path failed ({type(e).__name__}).")
        return None


def _try_load_adult_github() -> pd.DataFrame | None:
    print(f"  [ADULT] Trying GitHub mirror ({_ADULT_GITHUB_URL}) ...")
    try:
        df = pd.read_csv(_ADULT_GITHUB_URL, header=None, names=_ADULT_COLS,
                         low_memory=False, na_values="?",
                         skipinitialspace=True)
        return df
    except Exception as e:
        print(f"  [ADULT] github path failed ({type(e).__name__}).")
        return None


def load_adult() -> pd.DataFrame:
    """Load UCI Adult / Census Income with multi-source fallback. Caches
    the result to data/adult_raw.csv so subsequent runs skip the network."""
    if os.path.exists(ADULT_PATH):
        print(f"\n[ADULT] Loading from cache: {ADULT_PATH}")
        return pd.read_csv(ADULT_PATH, low_memory=False)

    print("\n[ADULT] No cache found; trying remote sources ...")
    for attempt in (_try_load_adult_ucimlrepo,
                    _try_load_adult_openml,
                    _try_load_adult_github):
        df = attempt()
        if df is not None and len(df) > 0:
            df = _normalise_adult(df)
            os.makedirs("data", exist_ok=True)
            df.to_csv(ADULT_PATH, index=False)
            print(f"[ADULT] Cached -> {ADULT_PATH}  "
                  f"({len(df):,} rows, {df.shape[1]} columns)")
            return df

    raise RuntimeError(
        "[ADULT] All load paths failed. Options:\n"
        "  (a) pip install ucimlrepo  and re-run NB1\n"
        "  (b) confirm network access to openml.org or "
        "raw.githubusercontent.com\n"
        "  (c) manually download adult.data + adult.test from "
        "https://archive.ics.uci.edu/dataset/20/census+income\n"
        f"      and place the combined CSV at {ADULT_PATH}"
    )


def eda_adult(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("ADULT / CENSUS INCOME -- EDA")
    print("="*60)
    print(f"\n  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # "?" is this dataset's missing-value sentinel on workclass,
    # occupation, native_country (per the UCI variable table).
    null_rates = (df.replace("?", np.nan).isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        "dtype":    df.dtypes,
        "n_unique": df.nunique(),
        "null_pct": null_rates,
    })
    summary.to_csv(f"{OUT_DIR}/adult_eda_summary.csv")
    print(f"\n  Null rates (%, '?' treated as missing):\n"
          f"{null_rates.to_string()}")

    print(f"\n  Sex distribution:\n"
          f"{df['sex'].value_counts(dropna=False).to_string()}")
    print(f"\n  Sex share (%):\n"
          f"{(df['sex'].value_counts(normalize=True) * 100).round(2).to_string()}")

    print(f"\n  Income target distribution:\n"
          f"{df['income'].value_counts(dropna=False).to_string()}")

    age = pd.to_numeric(df["age"], errors="coerce")
    print(f"\n  Age stats: n={len(age):,}, mean={age.mean():.1f}, "
          f"median={age.median():.0f}, range={age.min():.0f}-{age.max():.0f}")

    print(f"\n  Top 10 occupations:\n"
          f"{df['occupation'].value_counts().head(10).to_string()}")
    print(f"\n  Marital status:\n"
          f"{df['marital_status'].value_counts().to_string()}")
    print(f"\n  Relationship (NOTE: excluded from model features in NB2 -- "
          f"see comment there):\n"
          f"{df['relationship'].value_counts().to_string()}")

    return summary


def plot_adult_sex_income(df: pd.DataFrame):
    """
    Two-panel chart, parallel to plot_fcc_gender_paid:
      Left  - gender distribution
      Right - >$50K rate by gender
    """
    tmp = df.copy()
    tmp["gender_clean"] = (
        tmp["sex"].astype(str).str.strip().map({"Male": "man", "Female": "woman"})
    )
    tmp["above_50k"] = (tmp["income"].astype(str).str.strip() == ">50K").astype(int)
    tmp = tmp[tmp["gender_clean"].isin(["man", "woman"])]

    stats = tmp.groupby("gender_clean")["above_50k"].agg(["mean", "count"])
    print(f"\n[ADULT]   >$50K rate by gender:\n{stats.to_string()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle("UCI Adult/Census Income -- Gender Distribution & >$50K Rate",
                 fontweight="bold")

    tmp["gender_clean"].value_counts().plot.bar(
        ax=axes[0], color=PALETTE["adult"], alpha=0.85
    )
    axes[0].set_title("Gender distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", labelrotation=0)
    for i, v in enumerate(tmp["gender_clean"].value_counts()):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

    stats["mean"].plot.bar(ax=axes[1], color=PALETTE["adult"], alpha=0.85)
    axes[1].set_title(">$50K rate by gender")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, max(0.4, stats["mean"].max() * 1.4))
    axes[1].tick_params(axis="x", labelrotation=0)
    for i, v in enumerate(stats["mean"]):
        axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10,
                     fontweight="bold")
    gap = stats["mean"].max() - stats["mean"].min()
    axes[1].text(0.98, 0.97, f"max gap = {gap:.3f}",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=9, color="red",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()
    out = f"{OUT_DIR}/adult_gender_income_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ADULT]   Saved -> {out}")


# MAIN

if __name__ == "__main__":
    print("\n" + "#"*65)
    print("  NOTEBOOK 1 -- DATA INGESTION & EDA")
    print("#"*65)

    os.makedirs("data", exist_ok=True)

    # Stack Overflow
    try:
        so_raw = load_so(SO_PATH)
        so_raw.to_csv("data/so_raw.csv", index=False)
        print(f"\n  [SO] Raw data saved -> data/so_raw.csv  "
              f"({len(so_raw):,} rows, {len(so_raw.columns)} columns)")
        eda_so(so_raw)
        for fn, label in [
            (plot_so_salary_by_age, "salary-by-age plot"),
            (plot_so_devtype,       "DevType plot"),
            (plot_so_years,         "YearsCode plot"),
        ]:
            try:
                fn(so_raw)
            except Exception as e:
                print(f"  [SO] Warning: {label} failed -- {e}")
    except FileNotFoundError as e:
        print(e)
        print("  [SO] Skipping SO section; please download the CSV and re-run.")

    # freeCodeCamp 2018
    fcc_raw = load_fcc()
    fcc_raw.to_csv("data/fcc_raw.csv", index=False)
    print(f"\n  [FCC] Raw data saved -> data/fcc_raw.csv  "
          f"({len(fcc_raw):,} rows, {len(fcc_raw.columns)} columns)")

    eda_fcc(fcc_raw)

    try:
        plot_fcc_gender_paid(fcc_raw)
    except Exception as e:
        print(f"  [FCC] Warning: gender/paid plot failed -- {e}")

    # UCI Adult / Census Income
    try:
        adult_raw = load_adult()
        print(f"\n  [ADULT] Raw data ready -> {ADULT_PATH}  "
              f"({len(adult_raw):,} rows, {len(adult_raw.columns)} columns)")
        eda_adult(adult_raw)
        try:
            plot_adult_sex_income(adult_raw)
        except Exception as e:
            print(f"  [ADULT] Warning: gender/income plot failed -- {e}")
    except (RuntimeError, ConnectionError, Exception) as e:
        # fetch_ucirepo raises a plain ConnectionError when archive.ics.uci.edu
        # isn't reachable (offline, sandboxed, or DNS-blocked). We previously
        # only caught RuntimeError which let the ConnectionError abort NB1.
        # Broad catch here is intentional: NB1's other branches have already
        # written their CSVs, so an Adult failure shouldn't kill the run.
        print(f"  [ADULT] Skipping -- {type(e).__name__}: {e}")
        print("  [ADULT] If you want Adult, install ucimlrepo and ensure "
              "https://archive.ics.uci.edu is reachable, then re-run NB1.")

    print("\n  NOTEBOOK 1 COMPLETE")
    print("   data/so_raw.csv")
    print("   data/fcc_raw.csv")
    print("   data/adult_raw.csv")
