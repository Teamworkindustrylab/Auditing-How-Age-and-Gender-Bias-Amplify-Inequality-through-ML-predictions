"""

  NOTEBOOK 2 -- FEATURE ENGINEERING & PREPROCESSING

  Inputs  : data/so_raw.csv   (from Notebook 1)
            data/fcc_raw.csv  (from Notebook 1)

  Outputs : data/preprocessed/so_preprocessed.csv
            data/preprocessed/fcc_preprocessed.csv
            data/raw_eda/so_feature_distributions.png
            data/raw_eda/so_sensitive_group_rates.png
            data/raw_eda/fcc_feature_distributions.png

  
  Sensitive-attribute definitions
  
  SO:
    S1  age_group       -- binary: young (<35) vs experienced (35+)
    S2  age_exp_pro     -- age x YearsCodePro bracket (junior/mid/senior)
    S3  age_exp_total   -- age x YearsCode bracket    (junior/mid/senior)

  FCC:
    gender_clean        -- binary: man vs woman   (primary)
    age_group           -- binary: young (<35) vs experienced (35+)   (kept
                           for optional intersectional extension in NB6)

  Targets
  SO  : above_median_salary (binary)
  FCC : paid_contributor    (binary, = 'Are you already working as a software developer?')

"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO    = "data/so_raw.csv"
IN_FCC   = "data/fcc_raw.csv"
IN_ADULT = "data/adult_raw.csv"
OUT      = "data/preprocessed"
PLOTS    = "data/raw_eda"
os.makedirs(OUT,   exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

PALETTE = {"so": "#f48024", "fcc": "#0a0a23", "adult": "#2e7d32", "bg": "#fafafa"}

TOP_DEVTYPES = [
    "Developer, full-stack",
    "Developer, back-end",
    "Developer, front-end",
    "Developer, mobile",
    "Data scientist or machine learning specialist",
    "DevOps specialist",
    "Engineering manager",
    "Data engineer",
    "Developer, desktop or enterprise applications",
    "Security professional",
]


# SHARED UTILITY

def _classify_age(age_val) -> str:
    """
    Fuzzy age classifier -- handles both SO ordinal age bands and
    FCC numeric ages. Thresholds at 35 -> 'young' vs 'experienced'.
    """
    s = str(age_val).strip().lower()
    if not s or s in ("nan", "na", ""):
        return "experienced"
    if "under 18" in s or "< 18" in s:
        return "young"
    # Try numeric (FCC has numeric ages e.g. "27")
    try:
        n = float(s)
        # Filter out obvious data-entry errors (e.g. age=250)
        if 5 <= n <= 100:
            return "young" if n < 35 else "experienced"
    except ValueError:
        pass
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return "experienced"
    return "young" if max(nums) <= 34 else "experienced"


# SECTION 1 -- STACK OVERFLOW 2024  (UNCHANGED)

class SOFeatureEngineering:

    YEARS_MAP = {
        "Less than 1 year": 0.5,  "1 to 2 years": 1.5,  "2 to 3 years": 2.5,
        "3 to 5 years": 4.0,      "6 to 8 years": 7.0,   "9 to 11 years": 10.0,
        "12 to 14 years": 13.0,   "15 to 17 years": 16.0, "18 to 20 years": 19.0,
        "21 to 23 years": 22.0,   "24 to 26 years": 25.0, "27 to 29 years": 28.0,
        "30 to 32 years": 31.0,   "33 to 35 years": 34.0, "36 to 38 years": 37.0,
        "39 to 41 years": 40.0,   "42 to 44 years": 43.0, "45 to 47 years": 46.0,
        "48 to 50 years": 49.0,   "More than 50 years": 52.0,
        "NA": np.nan, "": np.nan,
    }

    ED_MAP = {
        "primary": 0, "elementary": 0,
        "secondary": 1, "high school": 1,
        "some college": 2, "without earning": 2,
        "associate": 3, "bachelor": 4, "master": 5,
        "doctoral": 6, "ph.d": 6, "professional": 6,
    }

    def load(self, path: str = IN_SO) -> "SOFeatureEngineering":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[SO FE] {path} not found -- run Notebook 1 first."
            )
        self.df = pd.read_csv(path)
        print(f"[SO FE] Loaded {len(self.df):,} rows from {path}")
        return self

    def _build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ConvertedCompYearly"] = pd.to_numeric(
            df["ConvertedCompYearly"].astype(str).str.strip()
              .replace({"NA": np.nan, "": np.nan}),
            errors="coerce",
        )
        n_before = len(df)
        df = df[df["ConvertedCompYearly"].notna() &
                (df["ConvertedCompYearly"] > 0)].copy()
        print(f"[SO FE] Rows with valid salary: {len(df):,} "
              f"(dropped {n_before - len(df):,})")

        lo = df["ConvertedCompYearly"].quantile(0.01)
        hi = df["ConvertedCompYearly"].quantile(0.99)
        df = df[(df["ConvertedCompYearly"] >= lo) &
                (df["ConvertedCompYearly"] <= hi)].copy()

        median_sal = df["ConvertedCompYearly"].median()
        df["above_median_salary"] = (
            df["ConvertedCompYearly"] >= median_sal
        ).astype(int)
        self.median_sal = median_sal
        print(f"[SO FE] After p1-p99 trim: {len(df):,} rows | "
              f"median salary ${median_sal:,.0f}")
        return df

    def _build_sensitive(self, df: pd.DataFrame) -> pd.DataFrame:
        df["age_group"] = df["Age"].apply(_classify_age)
        ycp = df["YearsCodePro"].map(self.YEARS_MAP)
        ycp = ycp.fillna(pd.to_numeric(df["YearsCodePro"], errors="coerce"))
        df["years_code_pro"] = ycp.fillna(ycp.median())
        yc = df["YearsCode"].map(self.YEARS_MAP)
        yc = yc.fillna(pd.to_numeric(df["YearsCode"], errors="coerce"))
        df["years_code"] = yc.fillna(yc.median())

        def _bracket(yrs):
            if pd.isna(yrs): return "unknown"
            if yrs < 5:      return "junior"
            if yrs < 15:     return "mid"
            return "senior"
        df["exp_pro_bracket"]   = df["years_code_pro"].apply(_bracket)
        df["exp_total_bracket"] = df["years_code"].apply(_bracket)
        df["age_exp_pro"]   = df["age_group"] + "_" + df["exp_pro_bracket"]
        df["age_exp_total"] = df["age_group"] + "_" + df["exp_total_bracket"]
        return df

    def _encode_education(self, df: pd.DataFrame) -> pd.DataFrame:
        def _ed(v):
            v = str(v).lower()
            for kw, score in self.ED_MAP.items():
                if kw in v:
                    return score
            return np.nan
        df["ed_level_enc"] = df["EdLevel"].apply(_ed)
        df["ed_level_enc"] = df["ed_level_enc"].fillna(df["ed_level_enc"].median())
        return df

    def _encode_employment(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_employed"] = (
            df["Employment"].fillna("").str.contains(
                "Employed|contractor|freelancer|self-employed", case=False
            )
        ).astype(int)
        df["is_student"] = (
            df["Employment"].fillna("").str.contains("Student", case=False)
        ).astype(int)
        df["is_remote"] = (
            df["RemoteWork"].fillna("").str.strip() == "Remote"
        ).astype(int)
        return df

    def _encode_devtype(self, df: pd.DataFrame) -> pd.DataFrame:
        df["DevType"] = df["DevType"].fillna("").astype(str)
        slugs = []
        for role in TOP_DEVTYPES:
            slug = ("devtype_"
                    + role.lower()
                         .replace(",", "").replace(" ", "_")
                         .replace("/", "_").replace("-", "_"))
            df[slug] = df["DevType"].str.contains(
                role, case=False, regex=False
            ).astype(int)
            slugs.append(slug)
        self.devtype_cols = slugs
        return df

    def engineer(self) -> "SOFeatureEngineering":
        df = self.df.copy()
        df = self._build_target(df)
        df = self._build_sensitive(df)
        df = self._encode_education(df)
        df = self._encode_employment(df)
        df = self._encode_devtype(df)
        self.df_engineered = df
        print(f"\n[SO FE] Final dataset: {df.shape}")
        return self

    @property
    def feature_cols(self) -> list:
        base = ["ed_level_enc", "is_employed", "is_remote", "is_student",
                "years_code", "years_code_pro"]
        return base + self.devtype_cols

    def save(self, path: str = f"{OUT}/so_preprocessed.csv") -> "SOFeatureEngineering":
        keep = list(dict.fromkeys(
            self.feature_cols
            + ["above_median_salary", "ConvertedCompYearly",
               "age_group", "age_exp_pro", "age_exp_total",
               "exp_pro_bracket", "exp_total_bracket",
               "years_code", "years_code_pro"]
        ))
        keep = [c for c in keep if c in self.df_engineered.columns]
        self.df_engineered[keep].to_csv(path, index=False)
        print(f"[SO FE] Saved -> {path}  "
              f"({len(self.df_engineered):,} rows, {len(keep)} columns)")
        return self


# SECTION 2 -- freeCodeCamp 2018 NEW CODER SURVEY (REPLACES GH OSS 2017)
# ----------------------------------------------------------------------
# Engineered feature set is intentionally similar in spirit to the
# previous GH OSS feature set, but uses fields that FCC actually
# collects:
#
#   GH OSS 2017 feature             ->  FCC 2018 analog
#   ----------------------------------------------------------------
#   pro_experience_yrs              ->  months_programming / 12
#   find_answers_score              ->  num_learning_resources (count
#                                       of distinct online resources
#                                       the respondent reports using)
#   receive_help_score              ->  attended_bootcamp (a binary
#                                       proxy for structured help-seeking)
#   had_negative_exp                ->  is_under_employed
#   had_harassment                  ->  is_ethnic_minority (proxy for
#                                       exposure to systemic disadvantage)
#   is_high_income_country          ->  (same logic, World-Bank-style list)
#   (new)                           ->  has_degree (bachelor's or higher)
#   (new)                           ->  log_expected_earning
#
# Result: 8 model features

class FCCFeatureEngineering:

    FEATURE_COLS = [
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

    HIGH_INCOME = {
        "United States of America", "United States", "Canada",
        "United Kingdom", "Germany", "France", "Italy", "Spain",
        "Netherlands", "Sweden", "Norway", "Denmark", "Finland",
        "Switzerland", "Austria", "Belgium", "Ireland", "Australia",
        "New Zealand", "Japan", "Singapore", "South Korea", "Israel",
    }

    # Learning-resource columns: every FCC binary column that names a
    # specific learning resource. We count them to build the
    # "num_learning_resources" feature, an analog of the OSS-help-
    # seeking ordinal in the previous dataset.
    
    LEARNING_RESOURCE_COLS = [
        "freeCodeCamp", "Coursera", "edX", "Khan Academy", "Udacity",
        "Udemy", "Codecademy", "Treehouse", "Pluralsight",
        "Lynda.com", "Code Wars", "HackerRank", "Stack Overflow",
        "MDN", "W3Schools", "egghead.io", "The Odin Project",
    ]

    def load(self, path: str = IN_FCC) -> "FCCFeatureEngineering":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[FCC FE] {path} not found -- run Notebook 1 first."
            )
        self.df = pd.read_csv(path, low_memory=False)
        print(f"\n[FCC FE] Loaded {len(self.df):,} rows from {path}")
        return self

    def engineer(self) -> "FCCFeatureEngineering":
        df = self.df.copy()

        # Sensitive attribute: gender
        def _gender(g):
            g = str(g).strip()
            if g in ("Male", "Man"):     return "man"
            if g in ("Female", "Woman"): return "woman"
            return "other"
        df["gender_clean"] = df["What's your gender?"].apply(_gender)

        # Sensitive attribute (bonus): age_group
        age_num = pd.to_numeric(df["How old are you?"], errors="coerce")
        age_num = age_num.where((age_num >= 14) & (age_num <= 80))
        df["age_numeric"] = age_num
        df["age_group"]   = age_num.apply(
            lambda v: "experienced" if pd.notna(v) and v >= 35 else "young"
        )

        # Target: are they already working as a software developer?
        dev = pd.to_numeric(
            df["Are you already working as a software developer?"],
            errors="coerce",
        )
        df["paid_contributor"] = dev.fillna(0).astype(int)

        # Feature: months programming (numeric, light winsorization)
        mp = pd.to_numeric(
            df["About how many months have you been programming for?"],
            errors="coerce",
        )
        mp = mp.clip(lower=0, upper=mp.quantile(0.99) if mp.notna().any() else 480)
        df["months_programming"] = mp.fillna(mp.median())

        # Feature: hours learning per week
        hl = pd.to_numeric(
            df["About how many hours do you spend learning each week?"],
            errors="coerce",
        )
        hl = hl.clip(lower=0, upper=hl.quantile(0.99) if hl.notna().any() else 100)
        df["hours_learning_per_week"] = hl.fillna(hl.median())

        # Feature: num distinct learning resources used
        present_resources = [c for c in self.LEARNING_RESOURCE_COLS
                             if c in df.columns]
        if present_resources:
            df["num_learning_resources"] = (
                df[present_resources]
                  .apply(pd.to_numeric, errors="coerce")
                  .fillna(0)
                  .sum(axis=1)
                  .clip(upper=len(present_resources))
            )
        else:
            df["num_learning_resources"] = 0
        print(f"[FCC FE] Learning-resource columns used: {len(present_resources)}")

        # Binary features
        def _to_bin(col):
            return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        df["attended_bootcamp"]  = _to_bin("Have you attended a full-time coding bootcamp?")
        df["is_under_employed"]  = _to_bin("Do you consider yourself under-employed?")
        df["is_ethnic_minority"] = _to_bin("Are you an ethnic minority in your country?")

        # Has bachelor's or higher
        deg = df["What's the highest degree or level of school you have completed?"].fillna("").str.lower()
        df["has_degree"] = deg.str.contains(
            "bachelor|master|doctorate|professional degree|ph", regex=True
        ).astype(int)

        # High-income country
        df["is_high_income_country"] = (
            df["Which country do you currently live in?"].isin(self.HIGH_INCOME)
        ).astype(int)

        # Log expected earning (USD) -- log1p handles zeros, clip outliers first
        ee = pd.to_numeric(
            df["About how much money do you expect to earn per year at your first developer job (in US Dollars)?"],
            errors="coerce",
        )
        ee = ee.clip(lower=0, upper=300_000)
        ee = ee.fillna(ee.median())
        df["log_expected_earning"] = np.log1p(ee)

        # Restrict to binary gender for downstream bias analysis
        df = df[df["gender_clean"].isin(["man", "woman"])].copy()

        self.df_engineered = df
        print(f"[FCC FE] Final shape after filtering to binary gender: {df.shape}")
        print(f"[FCC FE] Gender breakdown:  "
              f"{df['gender_clean'].value_counts().to_dict()}")
        print(f"[FCC FE] Target rate (working as developer):  "
              f"{df['paid_contributor'].mean():.3f}")
        return self

    def save(self, path: str = f"{OUT}/fcc_preprocessed.csv") -> "FCCFeatureEngineering":
        keep = [c for c in self.FEATURE_COLS
                + ["paid_contributor", "gender_clean", "age_group", "age_numeric"]
                if c in self.df_engineered.columns]
        self.df_engineered[keep].to_csv(path, index=False)
        print(f"[FCC FE] Saved -> {path}  "
              f"({len(self.df_engineered):,} rows, {len(keep)} columns)")
        return self

    def plot_feature_distributions(self):
        df = self.df_engineered
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=PALETTE["bg"])
        fig.suptitle("FCC 2018 -- Engineered Feature Distributions",
                     fontweight="bold")

        numeric_feats = ["months_programming", "hours_learning_per_week",
                         "num_learning_resources"]
        for ax, col in zip(axes[0], numeric_feats):
            df[col].hist(bins=30, ax=ax, color=PALETTE["fcc"], alpha=0.8)
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("Count")

        binary_feats = ["attended_bootcamp", "is_under_employed",
                        "is_ethnic_minority"]
        for ax, col in zip(axes[1], binary_feats):
            df[col].value_counts().sort_index().plot.bar(
                ax=ax, color=PALETTE["fcc"], alpha=0.8
            )
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelrotation=0)

        plt.tight_layout()
        out = f"{PLOTS}/fcc_feature_distributions.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FCC FE] Saved -> {out}")

    def plot_sensitive_group_rates(self):
        """Working-as-developer rate by gender, and by age x gender."""
        df = self.df_engineered
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])
        fig.suptitle("FCC 2018 -- Working-as-Developer Rate by Sensitive Group",
                     fontweight="bold")

        # By gender
        rates = (
            df.groupby("gender_clean")["paid_contributor"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "rate", "count": "n"})
        )
        colors = [PALETTE["fcc"] if g == "woman" else "#888888" for g in rates.index]
        rates["rate"].plot.bar(ax=axes[0], color=colors, alpha=0.85)
        axes[0].set_title("By gender")
        axes[0].set_ylabel("Working-as-developer rate")
        axes[0].set_ylim(0, max(0.4, rates["rate"].max() * 1.4))
        axes[0].tick_params(axis="x", labelrotation=0)
        for i, v in enumerate(rates["rate"]):
            axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10,
                         fontweight="bold")
        gap = rates["rate"].max() - rates["rate"].min()
        axes[0].text(0.98, 0.97, f"max gap = {gap:.3f}",
                     transform=axes[0].transAxes, ha="right", va="top",
                     fontsize=9, color="red",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        # By gender x age
        df["gender_age"] = df["gender_clean"] + "_" + df["age_group"]
        rates2 = (
            df.groupby("gender_age")["paid_contributor"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "rate", "count": "n"})
              .sort_values("rate", ascending=False)
        )
        colors2 = [PALETTE["fcc"] if "woman" in g else "#888888"
                   for g in rates2.index]
        rates2["rate"].plot.bar(ax=axes[1], color=colors2, alpha=0.85)
        axes[1].set_title("By gender x age")
        axes[1].set_ylabel("Working-as-developer rate")
        axes[1].set_ylim(0, max(0.4, rates2["rate"].max() * 1.4))
        axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
        for i, v in enumerate(rates2["rate"]):
            axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)

        plt.tight_layout()
        out = f"{PLOTS}/fcc_sensitive_group_rates.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FCC FE] Saved -> {out}")


# SECTION 3 -- UCI ADULT / CENSUS INCOME (NEW)
# ----------------------------------------------------------------------
# Sensitive attribute: gender_clean (man/woman), from `sex`.
# Bonus intersectional attribute: gender_age_bracket (gender x age
# bracket), built here so NB6 can read it straight off the preprocessed
# CSV -- the same pattern SO uses for age_exp_pro, rather than FCC's
# pattern of deriving the bracket inline inside NB6 itself.
# Target: above_50k (income > $50K/yr).
#
# Feature set is deliberately narrower than the full 14 UCI columns:
#   - `relationship` (Husband/Wife/Own-child/...) is EXCLUDED. In this
#     dataset relationship is close to a direct gender proxy -- almost
#     every "Husband" row is male and almost every "Wife" row is female
#     -- so including it wouldn't measure indirect/proxy discrimination
#     the way years_code_pro does for SO; it would just be re-injecting
#     the sensitive attribute under another name. `marital_status` is
#     kept (as the binary `is_married`) because it's a softer signal --
#     correlated with gender but not a near-1:1 encoding of it -- which
#     is a more defensible proxy-discrimination story.
#   - `fnlwgt` is EXCLUDED -- it's a Census sampling/representativeness
#     weight, not a person-level attribute, and is conventionally
#     dropped in essentially all published treatments of this dataset.
#   - `race` and the 41-way `native_country` are EXCLUDED from the base
#     feature set for parity with how FCC keeps geography to a single
#     coarse `is_high_income_country` flag; native_country collapses to
#     `is_us` here in the same spirit. Both fields remain in the saved
#     preprocessed CSV if you want to extend the intersectional analysis
#     to race or geography later.

class AdultFeatureEngineering:

    FEATURE_COLS = [
        "education_num", "hours_per_week",
        "log_capital_gain", "log_capital_loss",
        "is_married", "is_government_employee", "is_self_employed",
        "is_us",
    ]

    TOP_OCCUPATIONS = [
        "Exec-managerial", "Prof-specialty", "Craft-repair",
        "Adm-clerical", "Sales", "Other-service",
        "Machine-op-inspct", "Transport-moving",
        "Handlers-cleaners", "Tech-support",
    ]

    def load(self, path: str = IN_ADULT) -> "AdultFeatureEngineering":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ADULT FE] {path} not found -- run Notebook 1 first."
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().replace("-", "_").replace(" ", "_")
                      for c in df.columns]
        self.df = df
        print(f"\n[ADULT FE] Loaded {len(df):,} rows from {path}")
        return self

    def engineer(self) -> "AdultFeatureEngineering":
        df = self.df.copy()

        # Sensitive attribute: gender. Restrict to binary for the same
        # statistical-power reason FCC does -- this dataset's `sex`
        # field is already binary-coded (Male/Female only), so no rows
        # are dropped here in practice.
        df["gender_clean"] = (
            df["sex"].astype(str).str.strip().map({"Male": "man", "Female": "woman"})
        )
        df = df[df["gender_clean"].isin(["man", "woman"])].copy()

        # Sensitive attribute (bonus): age_group, reusing the SAME
        # shared _classify_age() used for SO's age_group above -- not a
        # separate inline lambda, to avoid the kind of missing-value
        # default drift flagged previously between SO and FCC's age
        # handling. Adult's `age` field has zero missing values per the
        # UCI documentation, so the fallback branch is never exercised
        # here in practice, but using the shared function keeps the
        # logic in one place rather than three.
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = df["age"].apply(_classify_age)

        def _age_bracket(a):
            if pd.isna(a): return "unknown"
            if a < 30:     return "junior"
            if a < 50:     return "mid"
            return "senior"
        df["age_bracket"] = df["age"].apply(_age_bracket)
        df["gender_age_bracket"] = df["gender_clean"] + "_" + df["age_bracket"]

        # Target
        df["above_50k"] = (
            df["income"].astype(str).str.strip().str.rstrip(".") == ">50K"
        ).astype(int)

        # Features
        df["education_num"] = pd.to_numeric(df["education_num"], errors="coerce")
        df["education_num"] = df["education_num"].fillna(df["education_num"].median())

        df["hours_per_week"] = pd.to_numeric(df["hours_per_week"], errors="coerce")
        df["hours_per_week"] = df["hours_per_week"].fillna(df["hours_per_week"].median())

        cg = pd.to_numeric(df["capital_gain"], errors="coerce").fillna(0)
        cl = pd.to_numeric(df["capital_loss"], errors="coerce").fillna(0)
        df["log_capital_gain"] = np.log1p(cg)
        df["log_capital_loss"] = np.log1p(cl)

        ms = df["marital_status"].fillna("").astype(str)
        df["is_married"] = ms.str.contains("Married", case=False).astype(int)

        wc = df["workclass"].fillna("").astype(str)
        df["is_government_employee"] = wc.str.contains("gov", case=False).astype(int)
        df["is_self_employed"]       = wc.str.contains("Self-emp", case=False).astype(int)

        df["is_us"] = (
            df["native_country"].astype(str).str.strip() == "United-States"
        ).astype(int)

        # Occupation dummies (parallel to SO's devtype_ dummies)
        occ = df["occupation"].fillna("").astype(str)
        occ_cols = []
        for role in self.TOP_OCCUPATIONS:
            slug = "occ_" + role.lower().replace("-", "_")
            df[slug] = (occ == role).astype(int)
            occ_cols.append(slug)
        self.occ_cols = occ_cols

        self.df_engineered = df
        print(f"[ADULT FE] Final shape: {df.shape}")
        print(f"[ADULT FE] Gender breakdown: "
              f"{df['gender_clean'].value_counts().to_dict()}")
        print(f"[ADULT FE] Target rate (>$50K): {df['above_50k'].mean():.3f}")
        return self

    @property
    def feature_cols(self) -> list:
        return self.FEATURE_COLS + self.occ_cols

    def save(self, path: str = f"{OUT}/adult_preprocessed.csv") -> "AdultFeatureEngineering":
        keep = list(dict.fromkeys(
            self.feature_cols
            + ["above_50k", "gender_clean", "age", "age_group",
               "age_bracket", "gender_age_bracket",
               "race", "native_country"]
        ))
        keep = [c for c in keep if c in self.df_engineered.columns]
        self.df_engineered[keep].to_csv(path, index=False)
        print(f"[ADULT FE] Saved -> {path}  "
              f"({len(self.df_engineered):,} rows, {len(keep)} columns)")
        return self

    def plot_feature_distributions(self):
        df = self.df_engineered
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=PALETTE["bg"])
        fig.suptitle("UCI Adult -- Engineered Feature Distributions",
                     fontweight="bold")

        numeric_feats = ["education_num", "hours_per_week", "log_capital_gain"]
        for ax, col in zip(axes[0], numeric_feats):
            df[col].hist(bins=30, ax=ax, color=PALETTE["adult"], alpha=0.8)
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("Count")

        binary_feats = ["is_married", "is_government_employee", "is_self_employed"]
        for ax, col in zip(axes[1], binary_feats):
            df[col].value_counts().sort_index().plot.bar(
                ax=ax, color=PALETTE["adult"], alpha=0.8
            )
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelrotation=0)

        plt.tight_layout()
        out = f"{PLOTS}/adult_feature_distributions.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[ADULT FE] Saved -> {out}")

    def plot_sensitive_group_rates(self):
        """>$50K rate by gender, and by gender x age bracket."""
        df = self.df_engineered
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])
        fig.suptitle("UCI Adult -- >$50K Rate by Sensitive Group",
                     fontweight="bold")

        rates = (
            df.groupby("gender_clean")["above_50k"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "rate", "count": "n"})
        )
        colors = [PALETTE["adult"] if g == "woman" else "#888888" for g in rates.index]
        rates["rate"].plot.bar(ax=axes[0], color=colors, alpha=0.85)
        axes[0].set_title("By gender")
        axes[0].set_ylabel(">$50K rate")
        axes[0].set_ylim(0, max(0.4, rates["rate"].max() * 1.4))
        axes[0].tick_params(axis="x", labelrotation=0)
        for i, v in enumerate(rates["rate"]):
            axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10,
                         fontweight="bold")
        gap = rates["rate"].max() - rates["rate"].min()
        axes[0].text(0.98, 0.97, f"max gap = {gap:.3f}",
                     transform=axes[0].transAxes, ha="right", va="top",
                     fontsize=9, color="red",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        rates2 = (
            df.groupby("gender_age_bracket")["above_50k"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "rate", "count": "n"})
              .sort_values("rate", ascending=False)
        )
        colors2 = [PALETTE["adult"] if "woman" in g else "#888888"
                   for g in rates2.index]
        rates2["rate"].plot.bar(ax=axes[1], color=colors2, alpha=0.85)
        axes[1].set_title("By gender x age bracket")
        axes[1].set_ylabel(">$50K rate")
        axes[1].set_ylim(0, max(0.4, rates2["rate"].max() * 1.4))
        axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
        for i, v in enumerate(rates2["rate"]):
            axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)

        plt.tight_layout()
        out = f"{PLOTS}/adult_sensitive_group_rates.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[ADULT FE] Saved -> {out}")


# MAIN

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NOTEBOOK 2 -- FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 65)

    # Stack Overflow (unchanged)
    if os.path.exists(IN_SO):
        so_fe = SOFeatureEngineering().load().engineer()
        so_fe.save()
    else:
        print(f"[SO FE] {IN_SO} not found -- skipping SO section.")

    # FCC (replaces GH OSS)
    fcc_fe = FCCFeatureEngineering().load().engineer()
    fcc_fe.save()
    try:
        fcc_fe.plot_feature_distributions()
        fcc_fe.plot_sensitive_group_rates()
    except Exception as e:
        print(f"[FCC FE] Warning: plot failed -- {e}")

    # UCI Adult / Census Income (new)
    if os.path.exists(IN_ADULT):
        adult_fe = AdultFeatureEngineering().load().engineer()
        adult_fe.save()
        try:
            adult_fe.plot_feature_distributions()
            adult_fe.plot_sensitive_group_rates()
        except Exception as e:
            print(f"[ADULT FE] Warning: plot failed -- {e}")
    else:
        print(f"[ADULT FE] {IN_ADULT} not found -- skipping Adult section.")

    print("\n  NOTEBOOK 2 COMPLETE")
    print("  -> data/preprocessed/so_preprocessed.csv")
    print("  -> data/preprocessed/fcc_preprocessed.csv")
    print("  -> data/preprocessed/adult_preprocessed.csv")
