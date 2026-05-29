"""

  NOTEBOOK 2 -- FEATURE ENGINEERING & PREPROCESSING

  Inputs  : data/so_raw.csv   (from Notebook 1)
            data/gh_raw.csv   (from Notebook 1)

  Outputs : data/preprocessed/so_preprocessed.csv
            data/preprocessed/gh_preprocessed.csv
            data/raw_eda/so_feature_distributions.png
            data/raw_eda/so_sensitive_group_rates.png

  Sensitive-attribute definitions (SO)
  -------------------------------------
  S1  age_group       -- binary: young (<35) vs experienced (35+)
  S2  age_exp_pro     -- age x YearsCodePro bracket (junior/mid/senior)
  S3  age_exp_total   -- age x YearsCode bracket    (junior/mid/senior)
  All three saved; NB3 evaluates them in parallel.

  YearsCode / YearsCodePro : ordinal string -> float midpoint
  DevType                  : multi-select semicolon -> top-N binary flags
  Target                   : above_median_salary (binary)

"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- no display needed
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO = "data/so_raw.csv"
IN_GH = "data/gh_raw.csv"
OUT   = "data/preprocessed"
PLOTS = "data/raw_eda"
os.makedirs(OUT,   exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

PALETTE = {"so": "#f48024", "gh": "#24292e", "bg": "#fafafa"}

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
    Fuzzy age classifier -- extracts the upper bound of the age band
    and thresholds at 35. Handles all SO survey export formats:
      '18-24 years old', '25-34 years old', 'Under 18 years old', etc.
    """
    s = str(age_val).strip().lower()
    if not s or s in ("nan", "na", ""):
        return "experienced"
    if "under 18" in s or "< 18" in s:
        return "young"
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return "experienced"
    return "young" if max(nums) <= 34 else "experienced"


# SECTION 1 -- STACK OVERFLOW 2024

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
        print(f"[SO FE] Columns: {list(self.df.columns)}")
        return self

    # Target
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
              f"(dropped {n_before - len(df):,} with missing/zero salary)")

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

    #  Sensitive attributes 
    def _build_sensitive(self, df: pd.DataFrame) -> pd.DataFrame:
        # S1 -- age only
        df["age_group"] = df["Age"].apply(_classify_age)

        # classification result
        vc = df["age_group"].value_counts()
        print(f"\n[SO FE] age_group distribution:")
        for k, n in vc.items():
            print(f"    {k:20s}  n={n:6,}  ({n/len(df)*100:.1f}%)")

        # YearsCodePro -> float
        ycp = df["YearsCodePro"].map(self.YEARS_MAP)
        ycp = ycp.fillna(pd.to_numeric(df["YearsCodePro"], errors="coerce"))
        median_ycp = ycp.median()
        df["years_code_pro"] = ycp.fillna(median_ycp)

        # YearsCode -> float
        yc = df["YearsCode"].map(self.YEARS_MAP)
        yc = yc.fillna(pd.to_numeric(df["YearsCode"], errors="coerce"))
        median_yc = yc.median()
        df["years_code"] = yc.fillna(median_yc)

        print(f"[SO FE] years_code_pro: median={median_ycp:.1f}, "
              f"nulls filled: {ycp.isna().sum()}")
        print(f"[SO FE] years_code:     median={median_yc:.1f}, "
              f"nulls filled: {yc.isna().sum()}")

        def _bracket(yrs):
            if pd.isna(yrs): return "unknown"
            if yrs < 5:      return "junior"
            if yrs < 15:     return "mid"
            return "senior"

        df["exp_pro_bracket"]   = df["years_code_pro"].apply(_bracket)
        df["exp_total_bracket"] = df["years_code"].apply(_bracket)

        # S2 and S3
        df["age_exp_pro"]   = df["age_group"] + "_" + df["exp_pro_bracket"]
        df["age_exp_total"] = df["age_group"] + "_" + df["exp_total_bracket"]

        for col in ["age_exp_pro", "age_exp_total"]:
            vc = df[col].value_counts()
            print(f"\n[SO FE] {col}:")
            for k, n in vc.items():
                print(f"    {k:35s}  n={n:6,}  ({n/len(df)*100:.1f}%)")

        return df

    # Education 
    def _encode_education(self, df: pd.DataFrame) -> pd.DataFrame:
        def _ed(v):
            v = str(v).lower()
            for kw, score in self.ED_MAP.items():
                if kw in v:
                    return score
            return np.nan
        df["ed_level_enc"] = df["EdLevel"].apply(_ed)
        median_ed = df["ed_level_enc"].median()
        df["ed_level_enc"] = df["ed_level_enc"].fillna(median_ed)
        return df

    #  Employment / remote 
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

    #  DevType flags 
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
        print(f"[SO FE] DevType flags: {slugs}")
        return df

    #  Orchestrate 
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

    # Save 
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

    # Plots 
    def plot_feature_distributions(self):
        df  = self.df_engineered
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=PALETTE["bg"])
        fig.suptitle("SO 2024 -- Engineered Feature Distributions",
                     fontweight="bold")

        for ax, col in zip(axes[0],
                           ["ed_level_enc", "years_code", "years_code_pro"]):
            df[col].dropna().hist(bins=30, ax=ax,
                                  color=PALETTE["so"], alpha=0.8)
            ax.set_title(col)
            ax.set_ylabel("Count")

        for ax, col in zip(axes[1],
                           ["is_employed", "is_remote", "is_student"]):
            df[col].value_counts().sort_index().plot.bar(
                ax=ax, color=PALETTE["so"], alpha=0.8)
            ax.set_title(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelrotation=0)

        plt.tight_layout()
        out = f"{PLOTS}/so_feature_distributions.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SO FE] Saved -> {out}")

    def plot_sensitive_group_rates(self):
        df  = self.df_engineered
        fig = plt.figure(figsize=(18, 6), facecolor=PALETTE["bg"])
        fig.suptitle(
            "SO 2024 -- Above-Median Salary Rate by Sensitive Group Definition",
            fontweight="bold"
        )
        sens_cols = [
            ("age_group",     "S1: Age group only"),
            ("age_exp_pro",   "S2: Age x YearsCodePro"),
            ("age_exp_total", "S3: Age x YearsCode (total)"),
        ]
        for i, (col, title) in enumerate(sens_cols):
            ax = fig.add_subplot(1, 3, i + 1)
            rates = (
                df.groupby(col)["above_median_salary"]
                  .agg(["mean", "count"])
                  .rename(columns={"mean": "rate", "count": "n"})
                  .sort_values("rate", ascending=False)
            )
            colors = [PALETTE["so"] if "young" in str(idx) else "#1a5276"
                      for idx in rates.index]
            rates["rate"].plot.bar(ax=ax, color=colors, alpha=0.85)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_ylabel("Above-median salary rate")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            for j, v in enumerate(rates["rate"]):
                ax.text(j, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
            gap = rates["rate"].max() - rates["rate"].min()
            ax.text(0.98, 0.97, f"max gap = {gap:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        plt.tight_layout()
        out = f"{PLOTS}/so_sensitive_group_rates.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SO FE] Saved -> {out}")


# SECTION 2 -- GITHUB OSS SURVEY 2017

class GHFeatureEngineering:

    FEATURE_COLS = [
        "pro_experience_yrs", "oss_experience_yrs",
        "find_answers_score", "receive_help_score",
        "contributor_help_score", "find_maintainer_score",
        "had_negative_exp", "had_harassment", "is_high_income_country",
    ]

    RENAME = {
        "GENDER": "gender_raw",
        "PROFESSIONAL.EXPERIENCE": "pro_experience",
        "CONTRIBUTING.TO.OSS": "oss_experience",
        "EMPLOYMENT.STATUS": "employment",
        "FIND.ANSWERS": "find_answers",
        "RECEIVE.HELP": "receive_help",
        "CONTRIBUTOR.HELP": "contributor_help",
        "FIND.MAINTAINER": "find_maintainer",
        "NEGATIVE.EXPERIENCES": "negative_exp",
        "HARASSMENT.RECEIVED": "harassment",
        "LOCATION": "location",
    }

    EXP_MAP  = {"< 1 year": 0.5, "1-2 years": 1.5, "3-5 years": 4.0,
                "6-10 years": 8.0, "11+ years": 13.0}
    FREQ_MAP = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
    HIGH_INCOME = {"United States", "Germany", "United Kingdom", "Canada",
                   "Australia", "Switzerland", "Netherlands", "Sweden"}

    def load(self, path: str = IN_GH) -> "GHFeatureEngineering":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[GH FE] {path} not found -- run Notebook 1 first."
            )
        self.df = pd.read_csv(path, low_memory=False)
        print(f"\n[GH FE] Loaded {len(self.df):,} rows from {path}")
        return self

    def engineer(self) -> "GHFeatureEngineering":
        df = self.df.copy()
        df = df.rename(columns={k: v for k, v in self.RENAME.items()
                                 if k in df.columns})

        def _gender(g):
            g = str(g).strip()
            if g in ("Man", "Male", "male", "man"):         return "man"
            if g in ("Woman", "Female", "female", "woman"): return "woman"
            return "other"
        df["gender_clean"] = df["gender_raw"].apply(_gender)

        paid_kw = ["Employed full-time", "Employed part-time",
                   "Self-employed", "Freelance", "Contractor"]
        df["paid_contributor"] = df["employment"].apply(
            lambda x: 1 if any(k in str(x) for k in paid_kw) else 0
        )

        df["pro_experience_yrs"] = (
            df["pro_experience"].map(self.EXP_MAP).fillna(4.0)
            if "pro_experience" in df.columns else 4.0
        )
        df["oss_experience_yrs"] = (
            df["oss_experience"].map(self.EXP_MAP).fillna(4.0)
            if "oss_experience" in df.columns else 4.0
        )

        for src, dst in [("find_answers",     "find_answers_score"),
                         ("receive_help",     "receive_help_score"),
                         ("contributor_help", "contributor_help_score"),
                         ("find_maintainer",  "find_maintainer_score")]:
            df[dst] = (df[src].map(self.FREQ_MAP).fillna(2.0)
                       if src in df.columns else 2.0)

        def _to_bin(v):
            return 1 if any(w in str(v).lower()
                            for w in ["yes", "true", "1", "often", "always"]) else 0

        df["had_negative_exp"] = (df["negative_exp"].apply(_to_bin)
                                  if "negative_exp" in df.columns else 0)
        df["had_harassment"]   = (df["harassment"].apply(_to_bin)
                                  if "harassment"   in df.columns else 0)
        df["is_high_income_country"] = (
            df["location"].isin(self.HIGH_INCOME).astype(int)
            if "location" in df.columns else 0
        )

        df = df[df["gender_clean"].isin(["man", "woman"])].copy()
        self.df_engineered = df
        print(f"[GH FE] Final shape: {df.shape} | "
              f"gender: {df['gender_clean'].value_counts().to_dict()}")
        return self

    def save(self, path: str = f"{OUT}/gh_preprocessed.csv") -> "GHFeatureEngineering":
        keep = [c for c in self.FEATURE_COLS + ["paid_contributor", "gender_clean"]
                if c in self.df_engineered.columns]
        self.df_engineered[keep].to_csv(path, index=False)
        print(f"[GH FE] Saved -> {path}  ({len(self.df_engineered):,} rows)")
        return self


# MAIN


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NOTEBOOK 2 -- FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 65)

    #  Stack Overflow 
    so_fe = SOFeatureEngineering().load().engineer()

    # Save FIRST before any plots
    so_fe.save()

    for fn, label in [
        (so_fe.plot_feature_distributions,  "feature distributions plot"),
        (so_fe.plot_sensitive_group_rates,  "sensitive group rates plot"),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"[SO FE] Warning: {label} failed -- {e}")

    #  GitHub OSS 
    gh_fe = GHFeatureEngineering().load().engineer()
    gh_fe.save()

    print("\n  NOTEBOOK 2 COMPLETE")
    print("  -> data/preprocessed/so_preprocessed.csv")
    print("  -> data/preprocessed/gh_preprocessed.csv")
