"""

  NOTEBOOK 1 — DATA INGESTION & EXPLORATORY DATA ANALYSIS

---- >> preprocessing.py contains data loading, feature engineering, bias baseline

Datasets:
  Dataset 1: Stack Overflow Developer Survey 2024
  Dataset 2: GitHub Open Source Survey 2017
 

Covers:
  - Path constants
  - reweigh_samples()          
  - StackOverflowPipeline        
  - GitHubOSSSurveyPipeline      
                                        

Downstream consumers (model training, SHAP, mitigation, plotting) live in
separate modules and import from here.


SETUP

1.Download SO survey from: https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024
                        1.1. "Download Full Data Set" -> unzip
                        1.2. use survey_results_public.csv  (65,439 rows)
                        
Place SO survey at:  data/survey_results_public.csv

2. GitHub OSS Survey loads automatically from the web.

3. Run requrements.txt to install dependencies:  pip install -r requirements.txt 


Outputs (saved to data/raw_eda/)
  
  so_eda_summary.csv          — per-column null-rate, dtype, n_unique
  so_age_salary_dist.png      — salary distribution by age group
  so_devtype_dist.png         — DevType frequency chart
  so_years_dist.png           — YearsCode / YearsCodePro distributions
  gh_eda_summary.csv
  gh_gender_paid_dist.png     — paid-contributor rate by gender
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

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────

SO_PATH  = "data/survey_results_public.csv"
GH_URL   = ("https://raw.githubusercontent.com/github/"
             "open-source-survey/master/data/survey_data.csv")
GH_PATH  = "data/gh_oss_survey.csv"
OUT_DIR  = "data/raw_eda"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "so":  "#f48024",
    "gh":  "#24292e",
    "bg":  "#fafafa",
}


# SECTION 1 — STACK OVERFLOW 2024

# ── 1.1  Parsing raw CSV ────────────────────────────────────────────────────────
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
    """
    So we need to parse SO survey CSV and return a DataFrame with the columns we care about.
    
    Because the inner-row quoting is inconsistent, we first read the header
    to detect column positions, then extract values positionally.
    """
    print(f"[SO] Parsing {path} …")

    # Step 1: read raw header to find column positions
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        raw_header = f.readline().strip()

    # The header row itself is a plain comma-separated line
    try:
        header_cols = list(csv.reader([raw_header]))[0]
    except Exception:
        header_cols = raw_header.split(",")

    def _find(name):
        """Return index of column name in header, or None."""
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

    # parsing each data row
    records = []
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if not row:
                continue
            # row[0] is the outer-quoted chunk
            try:
                inner = list(csv.reader([row[0]]))[0] if row[0] else []
            except Exception:
                inner = []

            def _g(lst, i, fallback=""):
                try:    return lst[i] if i is not None else fallback
                except: return fallback

            # ConvertedCompYearly is always at row[-2] (not inside row[0])
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
            "  → 'Download Full Data Set' → unzip\n"
            "  → place survey_results_public.csv at data/so_survey_2024.csv"
        )
    return _parse_so(path)


# ── 1.2  EDA ──────────────────────────────────────────────────────────────────

def eda_so(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("SO 2024 — EDA")
    print("="*60)

    # Basic shape
    print(f"\n  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Null rates
    null_rates = (df.replace("", np.nan).isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        "dtype":     df.dtypes,
        "n_unique":  df.nunique(),
        "null_pct":  null_rates,
        "sample":    df.iloc[0],
    })
    summary.to_csv(f"{OUT_DIR}/so_eda_summary.csv")
    print(f"\n  Null rates (%):\n{null_rates.to_string()}")

    # Salary distribution
    comp = pd.to_numeric(
        df["ConvertedCompYearly"].replace({"NA": np.nan, "": np.nan}),
        errors="coerce"
    )
    valid_comp = comp.dropna()
    print(f"\n  Salary rows with valid value : {len(valid_comp):,} "
          f"({len(valid_comp)/len(df)*100:.1f}%)")
    print(f"  Median salary               : ${valid_comp.median():,.0f}")
    print(f"  Salary range (p1–p99)       : "
          f"${valid_comp.quantile(.01):,.0f} – ${valid_comp.quantile(.99):,.0f}")

    # Age value counts
    print(f"\n  Age value counts:\n{df['Age'].value_counts().head(10).to_string()}")

    # YearsCode value counts
    print(f"\n  YearsCode value counts:\n"
          f"{df['YearsCode'].value_counts().head(10).to_string()}")
    print(f"\n  YearsCodePro value counts:\n"
          f"{df['YearsCodePro'].value_counts().head(10).to_string()}")

    # DevType (multi-select, semicolon-separated)
    devtype_all = (
        df["DevType"].dropna()
          .str.split(";")
          .explode()
          .str.strip()
          .value_counts()
    )
    print(f"\n  DevType (top 15):\n{devtype_all.head(15).to_string()}")

    return summary


# ── 1.3  Plots ────────────────────────────────────────────────────────────────

def _classify_age(age_val: str) -> str:
    """
    Classify an Age string as "young (< 35)" or "experienced (35+)".

    Handles the two formats seen across SO survey exports:
      • "18-24 years old"  / "25-34 years old"  
      • "18 - 24 years old"               
      • "Under 18 years old"
      
    Strategy: extract the *upper* bound of the range and threshold at 35.
    Anything whose upper bound is <= 34 is "young", everything else "experienced".
    
    """
    import re
    s = str(age_val).strip().lower()

    if not s or s in ("nan", "na", ""):
        return "experienced (35+)"   # safest imputation for missing

    # "under 18" -> upper bound 17
    if "under 18" in s or "< 18" in s:
        return "young (< 35)"

    # Extract all numbers from the string, e.g. "18-24" -> [18, 24]
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return "experienced (35+)"

    upper = max(nums)   # use the upper end of the age band
    return "young (< 35)" if upper <= 34 else "experienced (35+)"


def plot_so_salary_by_age(df: pd.DataFrame):
    """Salary distribution split by young vs experienced."""
    comp = pd.to_numeric(
        df["ConvertedCompYearly"].replace({"NA": np.nan, "": np.nan}),
        errors="coerce"
    )

    # Debug: show what Age values actually exist in the data
    age_counts = df["Age"].value_counts()
    print(f"\n[SO]   Age unique values in data:\n{age_counts.to_string()}")

    age_group = df["Age"].apply(_classify_age)
    group_counts = age_group.value_counts()
    print(f"\n[SO]   Age group counts after classification:\n{group_counts.to_string()}")

    plot_df = pd.DataFrame({"salary": comp, "age_group": age_group})
    plot_df  = plot_df.dropna()
    lo = plot_df["salary"].quantile(0.01)
    hi = plot_df["salary"].quantile(0.99)
    plot_df  = plot_df[(plot_df["salary"] >= lo) & (plot_df["salary"] <= hi)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("SO 2024 — Salary Distribution by Age Group", fontweight="bold")

    # KDE
    for grp, color in [("young (< 35)", "#f48024"), ("experienced (35+)", "#1a5276")]:
        sub = plot_df[plot_df["age_group"] == grp]["salary"]
        axes[0].hist(sub / 1000, bins=60, alpha=0.55, label=grp, color=color,
                     density=True)
    axes[0].set_xlabel("Annual Salary (USD thousands)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Salary distribution (KDE)")
    axes[0].legend()

    # Box
    groups = ["young (< 35)", "experienced (35+)"]
    data   = [plot_df[plot_df["age_group"] == g]["salary"].values / 1000
              for g in groups]
    bp = axes[1].boxplot(data, labels=groups, patch_artist=True,
                         medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], ["#f48024", "#1a5276"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Annual Salary (USD thousands)")
    axes[1].set_title("Box plot comparison")

    plt.tight_layout()
    out = f"{OUT_DIR}/so_age_salary_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SO]   Saved → {out}")


def plot_so_devtype(df: pd.DataFrame):
    """Top DevType categories."""
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
    ax.set_title("SO 2024 — Top 15 DevType categories", fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/so_devtype_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SO]   Saved → {out}")


def plot_so_years(df: pd.DataFrame):
    """YearsCode and YearsCodePro distributions."""
    # These are ordinal strings; we show value counts as bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("SO 2024 — Coding Experience Distributions", fontweight="bold")

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
        # Keep only known ordinal values, sort by order
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
    print(f"[SO]   Saved → {out}")



# SECTION 2 — GITHUB OSS SURVEY 2017


def load_gh() -> pd.DataFrame:
    if os.path.exists(GH_PATH):
        print(f"\n[GH] Loading from cache: {GH_PATH}")
        return pd.read_csv(GH_PATH, low_memory=False)

    print(f"[GH] Downloading from GitHub …")
    try:
        df = pd.read_csv(GH_URL, low_memory=False)
        os.makedirs("data", exist_ok=True)
        df.to_csv(GH_PATH, index=False)
        print(f"[GH] Cached → {GH_PATH}")
        return df
    except Exception as e:
        print(f"[GH] Download failed ({e}) — using synthetic stand-in.")
        return _synthetic_gh()


def _synthetic_gh(n: int = 5_441) -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    gs   = rng.choice(["Man", "Woman", "Other"],
                      p=[0.91, 0.03, 0.06], size=n)
    opts = ["< 1 year", "1-2 years", "3-5 years", "6-10 years", "11+ years"]
    pro  = rng.choice(opts, p=[0.05, 0.12, 0.25, 0.30, 0.28], size=n)
    oss  = rng.choice(opts, p=[0.10, 0.18, 0.28, 0.26, 0.18], size=n)
    fo   = ["Never", "Rarely", "Sometimes", "Often", "Always"]
    fa   = np.where(gs == "Woman",
                    rng.choice(fo, p=[0.05, 0.20, 0.35, 0.28, 0.12], size=n),
                    rng.choice(fo, p=[0.02, 0.10, 0.30, 0.35, 0.23], size=n))
    rh   = np.where(gs == "Woman",
                    rng.choice(fo, p=[0.06, 0.22, 0.33, 0.26, 0.13], size=n),
                    rng.choice(fo, p=[0.03, 0.12, 0.28, 0.35, 0.22], size=n))
    neg  = np.where(gs == "Woman",
                    rng.binomial(1, 0.55, n), rng.binomial(1, 0.20, n))
    har  = np.where(gs == "Woman",
                    rng.binomial(1, 0.25, n), rng.binomial(1, 0.08, n))
    base = (0.15
            + 0.04 * pd.Categorical(pro, categories=opts).codes
            + 0.03 * pd.Categorical(oss, categories=opts).codes)
    pen  = np.where(gs == "Woman", -0.15, 0.0)
    paid = rng.binomial(1, np.clip(base + pen, 0.05, 0.90))
    return pd.DataFrame({
        "GENDER":                  gs,
        "PROFESSIONAL.EXPERIENCE": pro,
        "CONTRIBUTING.TO.OSS":     oss,
        "EMPLOYMENT.STATUS":       np.where(paid == 1,
                                            "Employed full-time",
                                            "Not employed"),
        "FIND.ANSWERS":            fa,
        "RECEIVE.HELP":            rh,
        "CONTRIBUTOR.HELP":        rng.choice(fo, size=n),
        "FIND.MAINTAINER":         rng.choice(fo, size=n),
        "NEGATIVE.EXPERIENCES":    neg.astype(str),
        "HARASSMENT.RECEIVED":     har.astype(str),
        "LOCATION":                rng.choice(
            ["United States", "Germany", "United Kingdom",
             "India", "Canada", "Other"], size=n),
    })


def eda_gh(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("GH OSS 2017 — EDA")
    print("="*60)
    print(f"\n  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    null_rates = (df.replace("", np.nan).isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        "dtype":    df.dtypes,
        "n_unique": df.nunique(),
        "null_pct": null_rates,
    })
    summary.to_csv(f"{OUT_DIR}/gh_eda_summary.csv")
    print(f"\n  Null rates (%):\n{null_rates.to_string()}")

    gender_col = "GENDER" if "GENDER" in df.columns else None
    if gender_col:
        print(f"\n  Gender value counts:\n"
              f"{df[gender_col].value_counts().to_string()}")

    emp_col = "EMPLOYMENT.STATUS" if "EMPLOYMENT.STATUS" in df.columns else None
    if emp_col:
        print(f"\n  Employment value counts:\n"
              f"{df[emp_col].value_counts().to_string()}")

    return summary


def plot_gh_gender_paid(df: pd.DataFrame):
    """Paid-contributor rate by gender."""
    gender_col = "GENDER" if "GENDER" in df.columns else None
    emp_col    = "EMPLOYMENT.STATUS" if "EMPLOYMENT.STATUS" in df.columns else None
    if not gender_col or not emp_col:
        print("[GH]   Skipping gender/paid plot — columns not found.")
        return

    paid_kw = ["Employed full-time", "Employed part-time",
               "Self-employed", "Freelance", "Contractor"]
    tmp = df.copy()
    tmp["paid"] = tmp[emp_col].apply(
        lambda x: 1 if any(k in str(x) for k in paid_kw) else 0
    )

    def _g(g):
        g = str(g).strip()
        if g in ("Man", "Male"):         return "man"
        if g in ("Woman", "Female"):     return "woman"
        return "other"
    tmp["gender_clean"] = tmp[gender_col].apply(_g)

    stats = tmp.groupby("gender_clean")["paid"].agg(["mean", "count"])
    print(f"\n[GH]   Paid-contributor rates:\n{stats.to_string()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle("GH OSS 2017 — Gender Distribution & Paid-Contributor Rate",
                 fontweight="bold")

    # Counts
    tmp["gender_clean"].value_counts().plot.bar(
        ax=axes[0], color=PALETTE["gh"], alpha=0.85
    )
    axes[0].set_title("Gender distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", labelrotation=0)

    # Rates
    stats["mean"].plot.bar(ax=axes[1], color=PALETTE["gh"], alpha=0.85)
    axes[1].set_title("Paid-contributor rate by gender")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", labelrotation=0)
    for i, v in enumerate(stats["mean"]):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    out = f"{OUT_DIR}/gh_gender_paid_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GH]   Saved → {out}")


# MAIN

if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  NOTEBOOK 1 — DATA INGESTION & EDA")
    print("█"*65)

    os.makedirs("data", exist_ok=True)

    #  Stack Overflow
    so_raw = load_so(SO_PATH)

    # Save immediately
    so_raw.to_csv("data/so_raw.csv", index=False)
    print(f"\n  [SO] Raw data saved → data/so_raw.csv  "
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
            print(f"  [SO] Warning: {label} failed — {e}")

    #  GitHub OSS 
    gh_raw = load_gh()

    # Save immediately
    gh_raw.to_csv("data/gh_raw.csv", index=False)
    print(f"\n  [GH] Raw data saved → data/gh_raw.csv  "
          f"({len(gh_raw):,} rows, {len(gh_raw.columns)} columns)")

    eda_gh(gh_raw)

    try:
        plot_gh_gender_paid(gh_raw)
    except Exception as e:
        print(f"  [GH] Warning: gender/paid plot failed — {e}")

    print("\n  NOTEBOOK 1 COMPLETE")
    print("   data/so_raw.csv")
    print("   data/gh_raw.csv")