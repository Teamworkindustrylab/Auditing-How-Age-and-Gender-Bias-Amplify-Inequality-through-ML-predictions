"""
=============================================================================
  BIAS AMPLIFICATION IN DEVELOPER REPUTATION SYSTEMS
  Dataset 1: Stack Overflow Developer Survey 2024
  Dataset 2: GitHub Open Source Survey 2017
=============================================================================

SETUP
─────
1. Place SO survey at:  data/so_survey_2024.csv
   Download from:       https://survey.stackoverflow.co/2024/
                        → "Download Full Data Set" → unzip
                        → use survey_results_public.csv  (65,439 rows)

2. GitHub OSS Survey loads automatically from the web.

Install:
   pip install pandas numpy scikit-learn xgboost shap fairlearn matplotlib seaborn
=============================================================================
"""

import csv
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

warnings.filterwarnings("ignore")
np.random.seed(42)

SO_PATH = "data/so_survey_2024.csv"
GH_URL  = ("https://raw.githubusercontent.com/github/"
            "open-source-survey/master/data/survey_data.csv")
GH_PATH = "data/gh_oss_survey.csv"

PALETTE = {
    "so":  "#f48024",
    "gh":  "#24292e",
    "ok":  "#27ae60",
    "bad": "#e74c3c",
    "bg":  "#fafafa",
}


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def reweigh_samples(y: pd.Series, g: pd.Series) -> np.ndarray:
    """
    Compute sample weights inversely proportional to group × label frequency.
    Standard pre-processing fairness technique (Kamiran & Calders 2012).
    """
    df_w = pd.DataFrame({"y": y.values, "g": g.values})
    n    = len(df_w)
    lut  = {}
    for (gi, yi), grp in df_w.groupby(["g", "y"]):
        lut[(gi, yi)] = (
            (df_w.g == gi).sum() * (df_w.y == yi).sum()
        ) / (n * len(grp))
    return df_w.apply(lambda r: lut.get((r.g, r.y), 1.0), axis=1).values


def threshold_calibrate(
    model,
    X_te: pd.DataFrame,
    g_te: pd.Series,
) -> np.ndarray:
    """
    Manual post-processing threshold calibration for demographic parity.
    Replaces fairlearn.ThresholdOptimizer which has a pandas dtype bug
    in some version combinations.

    For each sensitive group we find the threshold on predicted probability
    that yields the same positive-prediction rate as the overall training
    base-rate, then apply group-specific thresholds on the test set.
    """
    y_prob = model.predict_proba(X_te)[:, 1]
    groups = np.unique(g_te)

    # Target: equalise positive-prediction rate across groups
    # Use the overall test predicted rate as the target
    global_rate = (y_prob >= 0.5).mean()

    thresholds = {}
    for grp in groups:
        mask  = (g_te == grp).values
        probs = y_prob[mask]
        # Binary search for threshold that hits global_rate for this group
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid  = (lo + hi) / 2
            rate = (probs >= mid).mean()
            if rate > global_rate:
                lo = mid
            else:
                hi = mid
        thresholds[grp] = (lo + hi) / 2

    y_pred = np.zeros(len(y_prob), dtype=int)
    for grp in groups:
        mask = (g_te == grp).values
        y_pred[mask] = (y_prob[mask] >= thresholds[grp]).astype(int)

    return y_pred


# =============================================================================
# DATASET 1 — STACK OVERFLOW 2024
# =============================================================================

class StackOverflowPipeline:
    """
    Predicts above-median salary from the SO 2024 Developer Survey.

    FILE FORMAT NOTE
    ────────────────
    Each data row is wrapped in an outer "..." quote in this CSV export.
    Some rows contain unescaped internal quotes (e.g. in degree names like
    "Bachelor's degree (B.A., B.S., B.Eng.)") which cause csv.reader to
    break the outer quote early. Columns after the break point are scrambled.

    Reliably extractable columns:
      row[0] → parse as CSV → Age[2], Employment[3], RemoteWork[4], EdLevel[7]
      row[-2]→ ConvertedCompYearly  (numeric, never causes a break)

    Sensitive attribute : Age  (young = under 35  vs  experienced = 35+)
    Target              : above_median_salary
    """

    FEATURE_COLS = [
        "ed_level_enc",  # education level ordinal 0-6  (43% non-null, rest imputed)
        "is_employed",   # paid employment flag           (99.8%)
        "is_remote",     # fully remote flag              (69%)
        "is_student",    # currently studying             (99.8%)
    ]

    # ── Load & parse ──────────────────────────────────────────────────────────
    def load(self, path: str = SO_PATH) -> "StackOverflowPipeline":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[SO] File not found: {path}\n"
                "  Download from https://survey.stackoverflow.co/2024/\n"
                "  → 'Download Full Data Set' → unzip\n"
                "  → place survey_results_public.csv at data/so_survey_2024.csv"
            )

        print(f"[SO] Parsing {path} …")
        records = []

        with open(path, "r", encoding="latin-1", errors="replace") as f:
            reader = csv.reader(f)
            next(reader)   # skip header

            for row in reader:
                if not row:
                    continue
                # row[0] contains the start of the data as a parseable CSV string
                try:
                    start = list(csv.reader([row[0]]))[0] if row[0] else []
                except Exception:
                    start = []

                def g(lst, i, d=""):
                    try:    return lst[i]
                    except: return d

                records.append({
                    "Age":                 g(start, 2),
                    "Employment":          g(start, 3),
                    "RemoteWork":          g(start, 4),
                    "EdLevel":             g(start, 7),
                    "ConvertedCompYearly": g(row,   -2),
                })

        df = pd.DataFrame(records)
        print(f"[SO] Rows parsed: {len(df):,}")
        self.df_raw = df
        return self

    # ── Feature engineering ───────────────────────────────────────────────────
    def engineer(self) -> "StackOverflowPipeline":
        df = self.df_raw.copy()

        # Salary → target
        df["ConvertedCompYearly"] = pd.to_numeric(
            df["ConvertedCompYearly"].astype(str).str.strip()
              .replace({"NA": np.nan, "": np.nan}),
            errors="coerce",
        )
        df = df[df["ConvertedCompYearly"].notna() &
                (df["ConvertedCompYearly"] > 0)].copy()
        print(f"[SO] Rows with valid salary: {len(df):,}")

        if len(df) == 0:
            raise RuntimeError(
                "[SO] No salary rows found. "
                "Make sure you are using survey_results_public.csv."
            )

        lo = df["ConvertedCompYearly"].quantile(0.01)
        hi = df["ConvertedCompYearly"].quantile(0.99)
        df = df[(df["ConvertedCompYearly"] >= lo) &
                (df["ConvertedCompYearly"] <= hi)].copy()

        median_sal = df["ConvertedCompYearly"].median()
        df["above_median_salary"] = (
            df["ConvertedCompYearly"] >= median_sal
        ).astype(int)
        print(f"[SO] After trim: {len(df):,} rows | median ${median_sal:,.0f}")

        # Age → sensitive attribute
        young_vals = {"Under 18 years old", "18-24 years old", "25-34 years old"}
        df["age_group"] = df["Age"].apply(
            lambda a: "young" if str(a).strip() in young_vals else "experienced"
        )
        print(f"[SO] Age groups:\n{df['age_group'].value_counts().to_string()}")

        # Education → ordinal
        def _ed(v):
            v = str(v).lower()
            if "primary" in v or "elementary" in v:                   return 0
            if "secondary" in v or "high school" in v:                return 1
            if "some college" in v or "without earning" in v:         return 2
            if "associate" in v:                                       return 3
            if "bachelor" in v:                                        return 4
            if "master" in v:                                          return 5
            if "doctoral" in v or "ph.d" in v or "professional" in v: return 6
            return np.nan

        df["ed_level_enc"] = df["EdLevel"].apply(_ed)
        df["ed_level_enc"] = df["ed_level_enc"].fillna(df["ed_level_enc"].median())

        # Employment flags
        df["is_employed"] = (
            df["Employment"].fillna("").str.contains(
                "Employed|contractor|freelancer|self-employed", case=False
            )
        ).astype(int)

        df["is_student"] = (
            df["Employment"].fillna("").str.contains("Student", case=False)
        ).astype(int)

        # Remote work
        df["is_remote"] = (
            df["RemoteWork"].fillna("").str.strip() == "Remote"
        ).astype(int)

        self.df         = df
        self.median_sal = median_sal
        print(f"[SO] Final dataset: {len(df):,} rows")
        return self

    # ── Bias baseline ─────────────────────────────────────────────────────────
    def baseline_bias(self) -> float:
        stats = (
            self.df.groupby("age_group")["above_median_salary"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "above_median_rate", "count": "n"})
        )
        print("\n" + "="*60)
        print("SO 2024 — STAGE 1: DATA BIAS BASELINE")
        print("="*60)
        print(stats.to_string())
        exp_r   = stats.loc["experienced", "above_median_rate"] \
                  if "experienced" in stats.index else np.nan
        young_r = stats.loc["young",       "above_median_rate"] \
                  if "young"       in stats.index else np.nan
        gap = exp_r - young_r
        print(f"\n  Gap (experienced − young above-median rate): {gap:.4f}")
        return gap

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self) -> "StackOverflowPipeline":
        X = self.df[self.FEATURE_COLS]
        y = self.df["above_median_salary"]
        g = self.df["age_group"]

        print(f"\n[SO] Training on {len(X):,} rows  "
              f"({(g=='young').sum():,} young | "
              f"{(g=='experienced').sum():,} experienced)")

        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.25, stratify=y, random_state=42
        )

        models = {
            "Logistic Regression": Pipeline([
                ("sc",  StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                           random_state=42)),
            ]),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
        }

        fitted = {}
        print("\n" + "="*60)
        print("SO 2024 — STAGE 2: MODEL TRAINING")
        print("="*60)
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            y_prob = m.predict_proba(X_te)[:, 1]
            auc    = roc_auc_score(y_te, y_prob)
            print(f"\n  {name}  AUC={auc:.4f}")
            print(classification_report(y_te, y_pred, digits=3))
            fitted[name] = dict(
                model=m, X_te=X_te, y_te=y_te, g_te=g_te,
                y_pred=y_pred, y_prob=y_prob, auc=auc,
            )

        self.fitted = fitted
        self.X_tr, self.y_tr, self.g_tr = X_tr, y_tr, g_tr
        return self

    # ── Measure amplification ─────────────────────────────────────────────────
    def measure_amplification(self, gap_data: float) -> dict:
        results = {}
        print("\n" + "="*60)
        print("SO 2024 — STAGE 3: BIAS AMPLIFICATION")
        print("="*60)
        for name, res in self.fitted.items():
            g_bin = (res["g_te"] == "young").astype(int)
            dpd   = demographic_parity_difference(
                        res["y_te"], res["y_pred"],
                        sensitive_features=g_bin)
            eod   = equalized_odds_difference(
                        res["y_te"], res["y_pred"],
                        sensitive_features=g_bin)
            amp   = abs(dpd) / (abs(gap_data) + 1e-9)
            print(f"\n  {name}:")
            print(f"    Demographic Parity Diff : {dpd:+.4f}")
            print(f"    Equalized Odds Diff     : {eod:+.4f}")
            print(f"    Amplification ratio     : {amp:.2f}x  "
                  f"(data gap={gap_data:.4f} → model gap={abs(dpd):.4f})")
            results[name] = dict(dpd=dpd, eod=eod,
                                 amplification=amp, auc=res["auc"])
        self.bias_results = results
        return results

    # ── SHAP ──────────────────────────────────────────────────────────────────
    def shap_analysis(self, model_name: str = "XGBoost"):
        res = self.fitted[model_name]
        clf = (res["model"].named_steps["clf"]
               if hasattr(res["model"], "named_steps")
               else res["model"])
        ev  = shap.Explainer(clf, res["X_te"])
        sv  = ev(res["X_te"])
        imp = pd.Series(
            np.abs(sv.values).mean(axis=0),
            index=self.FEATURE_COLS,
        ).sort_values(ascending=False)
        print("\n" + "="*60)
        print(f"SO 2024 — STAGE 4: SHAP  ({model_name})")
        print("="*60)
        print(imp.to_string())
        self.shap_vals  = sv
        self.importance = imp
        return sv, imp

    # ── Mitigation ────────────────────────────────────────────────────────────
    def mitigate(self):
        res              = self.fitted["XGBoost"]
        X_te, y_te, g_te = res["X_te"], res["y_te"], res["g_te"]
        g_bin            = (g_te == "young").astype(int)

        # ── A: Reweighing ─────────────────────────────────────────────────────
        sw = reweigh_samples(self.y_tr, self.g_tr)
        xgb_rw = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_rw.fit(self.X_tr, self.y_tr, sample_weight=sw)
        y_rw   = xgb_rw.predict(X_te)
        dpd_rw = demographic_parity_difference(y_te, y_rw,
                                               sensitive_features=g_bin)
        auc_rw = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])

        # ── B: Manual threshold calibration ──────────────────────────────────
        # (replaces fairlearn.ThresholdOptimizer which has a pandas dtype bug)
        y_to   = threshold_calibrate(res["model"], X_te, g_te)
        dpd_to = demographic_parity_difference(y_te, y_to,
                                               sensitive_features=g_bin)
        auc_to = roc_auc_score(y_te, res["y_prob"])  # probs unchanged

        self.tradeoff = {
            "baseline":  (res["auc"],  abs(self.bias_results["XGBoost"]["dpd"])),
            "reweigh":   (auc_rw,      abs(dpd_rw)),
            "threshold": (auc_to,      abs(dpd_to)),
        }
        print("\n" + "="*60)
        print("SO 2024 — STAGE 6: MITIGATION TRADE-OFF")
        print("="*60)
        for k, (a, d) in self.tradeoff.items():
            print(f"  {k:12s}  AUC={a:.4f}  |DPD|={d:.4f}")
        return self.tradeoff


# =============================================================================
# DATASET 2 — GITHUB OPEN SOURCE SURVEY 2017
# =============================================================================

class GitHubOSSSurveyPipeline:
    """
    Predicts paid-contributor status from the GitHub Open Source Survey.
    Sensitive attribute : Gender  (explicit — Man / Woman)
    Target              : paid_contributor (binary)
    Loads automatically from the web. No setup needed.
    """

    FEATURE_COLS = [
        "pro_experience_yrs",
        "oss_experience_yrs",
        "find_answers_score",
        "receive_help_score",
        "contributor_help_score",
        "find_maintainer_score",
        "had_negative_exp",
        "had_harassment",
        "is_high_income_country",
    ]

    # ── Load ─────────────────────────────────────────────────────────────────
    def load(self) -> "GitHubOSSSurveyPipeline":
        if os.path.exists(GH_PATH):
            print(f"[GH] Loading from cache: {GH_PATH}")
            raw = pd.read_csv(GH_PATH, low_memory=False)
        else:
            print(f"[GH] Downloading …")
            try:
                raw = pd.read_csv(GH_URL, low_memory=False)
                os.makedirs("data", exist_ok=True)
                raw.to_csv(GH_PATH, index=False)
                print(f"[GH] Cached → {GH_PATH}")
            except Exception as e:
                print(f"[GH] Download failed ({e}) — using synthetic stand-in.")
                return self._load_synthetic()
        print(f"[GH] Rows: {len(raw):,}")
        self.df_raw = raw
        return self

    def _load_synthetic(self) -> "GitHubOSSSurveyPipeline":
        n    = 5_441
        rng  = np.random.default_rng(42)
        gs   = rng.choice(["Man", "Woman", "Other"],
                          p=[0.91, 0.03, 0.06], size=n)
        opts = ["< 1 year", "1-2 years", "3-5 years",
                "6-10 years", "11+ years"]
        pro  = rng.choice(opts, p=[0.05, 0.12, 0.25, 0.30, 0.28], size=n)
        oss  = rng.choice(opts, p=[0.10, 0.18, 0.28, 0.26, 0.18], size=n)
        fo   = ["Never", "Rarely", "Sometimes", "Often", "Always"]
        fa   = np.where(gs == "Woman",
                        rng.choice(fo, p=[0.05,0.20,0.35,0.28,0.12], size=n),
                        rng.choice(fo, p=[0.02,0.10,0.30,0.35,0.23], size=n))
        rh   = np.where(gs == "Woman",
                        rng.choice(fo, p=[0.06,0.22,0.33,0.26,0.13], size=n),
                        rng.choice(fo, p=[0.03,0.12,0.28,0.35,0.22], size=n))
        neg  = np.where(gs == "Woman",
                        rng.binomial(1, 0.55, n),
                        rng.binomial(1, 0.20, n))
        har  = np.where(gs == "Woman",
                        rng.binomial(1, 0.25, n),
                        rng.binomial(1, 0.08, n))
        base = (0.15
                + 0.04 * pd.Categorical(pro, categories=opts).codes
                + 0.03 * pd.Categorical(oss, categories=opts).codes)
        pen  = np.where(gs == "Woman", -0.15, 0.0)
        paid = rng.binomial(1, np.clip(base + pen, 0.05, 0.90))
        self.df_raw = pd.DataFrame({
            "GENDER":                  gs,
            "PROFESSIONAL.EXPERIENCE": pro,
            "CONTRIBUTING.TO.OSS":     oss,
            "EMPLOYMENT.STATUS":       np.where(
                                           paid == 1,
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
        print(f"[GH] Synthetic stand-in ({n:,} rows)")
        return self

    # ── Feature engineering ───────────────────────────────────────────────────
    def engineer(self) -> "GitHubOSSSurveyPipeline":
        df = self.df_raw.copy()
        rename = {
            "GENDER":                  "gender_raw",
            "PROFESSIONAL.EXPERIENCE": "pro_experience",
            "CONTRIBUTING.TO.OSS":     "oss_experience",
            "EMPLOYMENT.STATUS":       "employment",
            "FIND.ANSWERS":            "find_answers",
            "RECEIVE.HELP":            "receive_help",
            "CONTRIBUTOR.HELP":        "contributor_help",
            "FIND.MAINTAINER":         "find_maintainer",
            "NEGATIVE.EXPERIENCES":    "negative_exp",
            "HARASSMENT.RECEIVED":     "harassment",
            "LOCATION":                "location",
        }
        df = df.rename(
            columns={k: v for k, v in rename.items() if k in df.columns}
        )

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

        exp_map = {"< 1 year": 0.5,  "1-2 years": 1.5,
                   "3-5 years": 4.0, "6-10 years": 8.0, "11+ years": 13.0}
        df["pro_experience_yrs"] = df["pro_experience"].map(exp_map).fillna(4.0)
        df["oss_experience_yrs"] = df["oss_experience"].map(exp_map).fillna(4.0)

        freq_map = {"Never": 0, "Rarely": 1, "Sometimes": 2,
                    "Often": 3, "Always": 4}
        for col, new in [
            ("find_answers",     "find_answers_score"),
            ("receive_help",     "receive_help_score"),
            ("contributor_help", "contributor_help_score"),
            ("find_maintainer",  "find_maintainer_score"),
        ]:
            df[new] = (df[col].map(freq_map).fillna(2.0)
                       if col in df.columns else 2.0)

        def _to_bin(val):
            return 1 if any(w in str(val).lower()
                            for w in ["yes", "true", "1",
                                      "often", "always"]) else 0

        df["had_negative_exp"] = (df["negative_exp"].apply(_to_bin)
                                  if "negative_exp" in df.columns else 0)
        df["had_harassment"]   = (df["harassment"].apply(_to_bin)
                                  if "harassment" in df.columns else 0)

        high_income = {"United States", "Germany", "United Kingdom",
                       "Canada", "Australia", "Switzerland",
                       "Netherlands", "Sweden"}
        df["is_high_income_country"] = (
            df["location"].isin(high_income).astype(int)
            if "location" in df.columns else 0
        )

        df = df[df["gender_clean"].isin(["man", "woman"])].copy()
        self.df = df
        print(f"[GH] Engineered: {df.shape} | "
              f"{df['gender_clean'].value_counts().to_dict()}")
        return self

    # ── Bias baseline ─────────────────────────────────────────────────────────
    def baseline_bias(self) -> float:
        stats = (
            self.df.groupby("gender_clean")["paid_contributor"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "paid_rate", "count": "n"})
        )
        print("\n" + "="*60)
        print("GH OSS — STAGE 1: DATA BIAS BASELINE")
        print("="*60)
        print(stats.to_string())
        m = stats.loc["man",   "paid_rate"] if "man"   in stats.index else np.nan
        f = stats.loc["woman", "paid_rate"] if "woman" in stats.index else np.nan
        gap = m - f
        print(f"\n  Gap (man − woman paid-contributor rate): {gap:.4f}")
        return gap

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self) -> "GitHubOSSSurveyPipeline":
        X = self.df[self.FEATURE_COLS]
        y = self.df["paid_contributor"]
        g = self.df["gender_clean"]
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.25, stratify=y, random_state=42
        )
        models = {
            "Logistic Regression": Pipeline([
                ("sc",  StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                           random_state=42)),
            ]),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
        }
        fitted = {}
        print("\n" + "="*60)
        print("GH OSS — STAGE 2: MODEL TRAINING")
        print("="*60)
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            y_prob = m.predict_proba(X_te)[:, 1]
            auc    = roc_auc_score(y_te, y_prob)
            print(f"\n  {name}  AUC={auc:.4f}")
            print(classification_report(y_te, y_pred, digits=3))
            fitted[name] = dict(model=m, X_te=X_te, y_te=y_te, g_te=g_te,
                                y_pred=y_pred, y_prob=y_prob, auc=auc)
        self.fitted = fitted
        self.X_tr, self.y_tr, self.g_tr = X_tr, y_tr, g_tr
        return self

    # ── Measure amplification ─────────────────────────────────────────────────
    def measure_amplification(self, gap_data: float) -> dict:
        results = {}
        print("\n" + "="*60)
        print("GH OSS — STAGE 3: BIAS AMPLIFICATION")
        print("="*60)
        for name, res in self.fitted.items():
            g_bin = (res["g_te"] == "woman").astype(int)
            dpd   = demographic_parity_difference(
                        res["y_te"], res["y_pred"],
                        sensitive_features=g_bin)
            eod   = equalized_odds_difference(
                        res["y_te"], res["y_pred"],
                        sensitive_features=g_bin)
            amp   = abs(dpd) / (abs(gap_data) + 1e-9)
            print(f"\n  {name}: DPD={dpd:+.4f}  EOD={eod:+.4f}  "
                  f"Amplification={amp:.2f}x")
            results[name] = dict(dpd=dpd, eod=eod,
                                 amplification=amp, auc=res["auc"])
        self.bias_results = results
        return results

    # ── SHAP ──────────────────────────────────────────────────────────────────
    def shap_analysis(self, model_name: str = "XGBoost"):
        res = self.fitted[model_name]
        clf = (res["model"].named_steps["clf"]
               if hasattr(res["model"], "named_steps")
               else res["model"])
        ev  = shap.Explainer(clf, res["X_te"])
        sv  = ev(res["X_te"])
        imp = pd.Series(np.abs(sv.values).mean(axis=0),
                        index=self.FEATURE_COLS).sort_values(ascending=False)
        print("\n" + "="*60)
        print(f"GH OSS — STAGE 4: SHAP  ({model_name})")
        print("="*60)
        print(imp.to_string())
        self.shap_vals  = sv
        self.importance = imp
        return sv, imp

    # ── Mitigation ────────────────────────────────────────────────────────────
    def mitigate(self):
        res              = self.fitted["XGBoost"]
        X_te, y_te, g_te = res["X_te"], res["y_te"], res["g_te"]
        g_bin            = (g_te == "woman").astype(int)

        # ── A: Reweighing ─────────────────────────────────────────────────────
        sw = reweigh_samples(self.y_tr, self.g_tr)
        xgb_rw = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_rw.fit(self.X_tr, self.y_tr, sample_weight=sw)
        y_rw   = xgb_rw.predict(X_te)
        dpd_rw = demographic_parity_difference(y_te, y_rw,
                                               sensitive_features=g_bin)
        auc_rw = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])

        # ── B: Manual threshold calibration ──────────────────────────────────
        y_to   = threshold_calibrate(res["model"], X_te, g_te)
        dpd_to = demographic_parity_difference(y_te, y_to,
                                               sensitive_features=g_bin)
        auc_to = roc_auc_score(y_te, res["y_prob"])

        self.tradeoff = {
            "baseline":  (res["auc"],  abs(self.bias_results["XGBoost"]["dpd"])),
            "reweigh":   (auc_rw,      abs(dpd_rw)),
            "threshold": (auc_to,      abs(dpd_to)),
        }
        print("\n" + "="*60)
        print("GH OSS — STAGE 6: MITIGATION")
        print("="*60)
        for k, (a, d) in self.tradeoff.items():
            print(f"  {k:12s}  AUC={a:.4f}  |DPD|={d:.4f}")
        return self.tradeoff


# =============================================================================
# COMPARISON DASHBOARD
# =============================================================================

def plot_comparison_dashboard(
    so: StackOverflowPipeline,
    gh: GitHubOSSSurveyPipeline,
    save_path: str = "outputs/bias_comparison_dashboard.png",
):
    os.makedirs("outputs", exist_ok=True)
    fig = plt.figure(figsize=(22, 16), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Bias Amplification Pipeline  ·  Cross-Dataset Comparison\n"
        "Stack Overflow 2024  (age bias in salary)  ──  "
        "GitHub Open Source Survey 2017  (gender bias in OSS participation)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40)
    mdl = list(so.fitted.keys())
    w   = 0.35

    # ① Data gap
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#fff8f0")
    so_r = so.df.groupby("age_group")["above_median_salary"].mean()
    gh_r = gh.df.groupby("gender_clean")["paid_contributor"].mean()
    x    = np.arange(2)
    ax1.bar(x - w/2,
            [so_r.get("experienced", 0), so_r.get("young", 0)], w,
            label="SO: above-median salary",
            color=PALETTE["so"], alpha=0.9)
    ax1.bar(x + w/2,
            [gh_r.get("man", 0), gh_r.get("woman", 0)], w,
            label="GH OSS: paid contributor",
            color=PALETTE["gh"], alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Experienced / Man", "Young / Woman"])
    ax1.set_ylabel("Positive outcome rate")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("① Data Bias Baseline", fontweight="bold")
    ax1.legend(fontsize=8)
    for bar in ax1.patches:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{bar.get_height():.2%}", ha="center", fontsize=8)

    # ② AUC
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor("#f0f4ff")
    x2      = np.arange(len(mdl))
    so_aucs = [so.fitted[m]["auc"] for m in mdl]
    gh_aucs = [gh.fitted[m]["auc"] for m in mdl]
    ax2.bar(x2 - w/2, so_aucs, w, label="SO 2024", color=PALETTE["so"])
    ax2.bar(x2 + w/2, gh_aucs, w, label="GH OSS",  color=PALETTE["gh"])
    ax2.set_xticks(x2)
    ax2.set_xticklabels([m.replace(" ", "\n") for m in mdl])
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.5, 1.0)
    ax2.set_title("② Model AUC", fontweight="bold")
    ax2.legend(fontsize=8)
    for i, (sv, gv) in enumerate(zip(so_aucs, gh_aucs)):
        ax2.text(i - w/2, sv + .005, f"{sv:.3f}", ha="center", fontsize=8)
        ax2.text(i + w/2, gv + .005, f"{gv:.3f}", ha="center", fontsize=8)

    # ③ Amplification
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#fff0f0")
    so_amps = [so.bias_results.get(m, {}).get("amplification", 0) for m in mdl]
    gh_amps = [gh.bias_results.get(m, {}).get("amplification", 0) for m in mdl]
    ax3.bar(x2 - w/2, so_amps, w, label="SO 2024", color=PALETTE["so"])
    ax3.bar(x2 + w/2, gh_amps, w, label="GH OSS",  color=PALETTE["gh"])
    ax3.axhline(1.0, color="black", linestyle="--", lw=1.2,
                label="No amplification (1x)")
    ax3.set_xticks(x2)
    ax3.set_xticklabels([m.replace(" ", "\n") for m in mdl])
    ax3.set_ylabel("Amplification ratio (x)")
    ax3.set_title("③ Bias Amplification Ratio\n(>1x = model worsens the gap)",
                  fontweight="bold")
    ax3.legend(fontsize=8)

    # ④ SHAP
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_facecolor("#f0fff4")
    so_n  = so.importance / so.importance.sum()
    gh_n  = gh.importance / gh.importance.sum()
    all_f = list(dict.fromkeys(list(so_n.index) + list(gh_n.index)))
    y4    = np.arange(len(all_f))
    ax4.barh(y4 - 0.2, [so_n.get(f, 0) for f in all_f], 0.35,
             label="SO 2024", color=PALETTE["so"], alpha=0.85)
    ax4.barh(y4 + 0.2, [gh_n.get(f, 0) for f in all_f], 0.35,
             label="GH OSS",  color=PALETTE["gh"], alpha=0.85)
    ax4.set_yticks(y4)
    ax4.set_yticklabels(all_f, fontsize=8)
    ax4.set_xlabel("Normalised mean |SHAP|")
    ax4.set_title("④ Feature Importance (SHAP)", fontweight="bold")
    ax4.legend(fontsize=8)

    # ⑤ EU Compliance
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.axis("off")
    rows = [
        ("Art. 10 — bias-free training data",       "NO SO", "NO GH"),
        ("Art. 10 — explicit sensitive attribute",   "~ SO (Age)", "YES GH (Gender)"),
        ("Art. 22 — right to explanation (SHAP)",   "YES SO", "YES GH"),
        ("Art. 22 — human override mechanism",      "NO SO",  "NO GH"),
        ("|DPD| <= 0.05 (pre-mitigation)",          "NO SO",  "NO GH"),
        ("|DPD| <= 0.05 (post-mitigation)",         "~ SO",   "~ GH"),
    ]
    tbl = ax5.table(
        cellText=rows,
        colLabels=["EU AI Act / GDPR Requirement", "SO 2024", "GH OSS 2017"],
        cellLoc="left", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dde4f0")
        elif c > 0:
            t = cell.get_text().get_text()
            if t.startswith("YES"):  cell.set_facecolor("#d5f5e3")
            elif t.startswith("NO"): cell.set_facecolor("#fde8e8")
            elif t.startswith("~"):  cell.set_facecolor("#fef9e7")
    ax5.set_title("⑤ EU AI Act / GDPR Art. 22 Compliance",
                  fontweight="bold", pad=14)

    # ⑥ Trade-off
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.set_facecolor("#fffbf0")
    strats  = ["baseline", "reweigh", "threshold"]
    labels  = {"baseline": "Baseline", "reweigh": "Reweighing",
               "threshold": "Threshold Calibration"}
    markers = ["X", "s", "o"]
    for i, s in enumerate(strats):
        so_a, so_d = so.tradeoff[s]
        gh_a, gh_d = gh.tradeoff[s]
        ax6.scatter(so_d, so_a, s=200, color=PALETTE["so"],
                    marker=markers[i], zorder=5,
                    label=f"SO · {labels[s]}",
                    edgecolors="white", lw=1.2)
        ax6.scatter(gh_d, gh_a, s=200, color=PALETTE["gh"],
                    marker=markers[i], zorder=5,
                    label=f"GH · {labels[s]}",
                    edgecolors="white", lw=1.2)
    ax6.axvline(0.05, color=PALETTE["bad"], linestyle="--", lw=1.2,
                label="|DPD|=0.05  (EU threshold)")
    ax6.set_xlabel("|DPD|  →  lower is fairer")
    ax6.set_ylabel("ROC-AUC  →  higher is better")
    ax6.set_title("⑥ Accuracy vs. Fairness Trade-off", fontweight="bold")
    ax6.legend(fontsize=7, ncol=2)

    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"\n  Dashboard saved → {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "█"*65)
    print("  BIAS AMPLIFICATION — SO 2024 + GITHUB OSS SURVEY 2017")
    print("█"*65)

    so = StackOverflowPipeline().load(SO_PATH).engineer()
    so_gap = so.baseline_bias()
    so.train()
    so.measure_amplification(so_gap)
    so.shap_analysis()
    so.mitigate()

    gh = GitHubOSSSurveyPipeline().load().engineer()
    gh_gap = gh.baseline_bias()
    gh.train()
    gh.measure_amplification(gh_gap)
    gh.shap_analysis()
    gh.mitigate()

    plot_comparison_dashboard(so, gh)

    print("\n" + "="*65)
    print("  DONE — see outputs/bias_comparison_dashboard.png")
    print("="*65)


if __name__ == "__main__":
    main()
