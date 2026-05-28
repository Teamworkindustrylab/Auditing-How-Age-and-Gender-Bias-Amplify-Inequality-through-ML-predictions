"""
=============================================================================
  NOTEBOOK 5 -- MITIGATION & COMPLIANCE REPORTING
=============================================================================
  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/gh_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/gh_xgboost.pkl

  Outputs : outputs/nb5_mitigation_tradeoff.png
            outputs/nb5_eu_compliance_table.png
            outputs/so_mitigation_results.csv
            outputs/gh_mitigation_results.csv

  Mitigation strategies
  ----------------------
  A. Reweighing (Kamiran & Calders 2012)
     Per-sample weights inversely proportional to group x label frequency.
     Retrain XGBoost with these weights.

  B. Threshold calibration (post-processing)
     Per-group binary-search for the threshold that equalises positive-
     prediction rate to the global rate. No retraining needed.

  Note on train/test split
  ------------------------
  NB5 re-splits from scratch with the same random_state=42 used in NB3.
  This gives the same test partition, so baseline AUC/DPD match NB3 exactly.
  The reweighed model is retrained fresh here (not saved in NB3).
=============================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score
from xgboost                 import XGBClassifier
from fairlearn.metrics       import demographic_parity_difference

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_GH   = "data/preprocessed/gh_preprocessed.csv"
OUT     = "outputs"
MDL_DIR = "outputs/models"
os.makedirs(OUT, exist_ok=True)

PALETTE = {"so": "#f48024", "gh": "#24292e",
           "ok": "#27ae60", "bad": "#e74c3c", "bg": "#fafafa"}

SO_BASE_FEATURES = [
    "ed_level_enc", "is_employed", "is_remote", "is_student",
    "years_code", "years_code_pro",
]
GH_FEATURE_COLS = [
    "pro_experience_yrs", "oss_experience_yrs",
    "find_answers_score", "receive_help_score",
    "contributor_help_score", "find_maintainer_score",
    "had_negative_exp", "had_harassment", "is_high_income_country",
]
SENSITIVE_DEFS = {
    "S1_age_group":     "age_group",
    "S2_age_exp_pro":   "age_exp_pro",
    "S3_age_exp_total": "age_exp_total",
}
DPD_THRESHOLD = 0.05


# =============================================================================
# SHARED MITIGATION HELPERS
# =============================================================================

def reweigh_samples(y: pd.Series, g: pd.Series) -> np.ndarray:
    """Kamiran & Calders (2012): weight = P(Y)*P(G) / P(Y,G)"""
    df_w = pd.DataFrame({"y": y.values, "g": g.values})
    n    = len(df_w)
    lut  = {}
    for (gi, yi), grp in df_w.groupby(["g", "y"]):
        lut[(gi, yi)] = (
            (df_w.g == gi).sum() * (df_w.y == yi).sum()
        ) / (n * len(grp))
    return df_w.apply(lambda r: lut.get((r.g, r.y), 1.0), axis=1).values


def threshold_calibrate(model, X_te: pd.DataFrame,
                        g_te: pd.Series) -> np.ndarray:
    """
    Post-processing: per-group threshold binary search to equalise
    positive-prediction rate to global predicted rate.
    Works for binary and multi-class sensitive attributes.
    """
    y_prob      = model.predict_proba(X_te)[:, 1]
    global_rate = (y_prob >= 0.5).mean()

    y_pred = np.zeros(len(y_prob), dtype=int)
    for grp in np.unique(g_te):
        mask  = (g_te == grp).values
        probs = y_prob[mask]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (probs >= mid).mean() > global_rate:
                lo = mid
            else:
                hi = mid
        y_pred[mask] = (probs >= (lo + hi) / 2).astype(int)
    return y_pred


def _max_gap(y_pred, g_series) -> float:
    """Max - min positive-prediction rate across all groups."""
    groups = np.unique(g_series)
    rates  = [y_pred[g_series == grp].mean() for grp in groups]
    return float(max(rates) - min(rates))


def _load_xgb(prefix: str):
    for name in [f"{prefix}_xgboost.pkl", f"{prefix}_xgb.pkl"]:
        p = os.path.join(MDL_DIR, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(
        f"No XGBoost model found for '{prefix}' in {MDL_DIR}. "
        f"Run Notebook 3 first."
    )


# =============================================================================
# SECTION 1 -- STACK OVERFLOW 2024
# =============================================================================

class SOMitigation:

    def load(self) -> "SOMitigation":
        if not os.path.exists(IN_SO):
            raise FileNotFoundError(
                f"[SO MIT] {IN_SO} not found -- run Notebook 2 first."
            )
        df = pd.read_csv(IN_SO)
        devtype_cols      = [c for c in df.columns if c.startswith("devtype_")]
        self.feature_cols = [c for c in SO_BASE_FEATURES + devtype_cols
                             if c in df.columns]
        self.df = df
        print(f"[SO MIT] Loaded {len(df):,} rows | "
              f"{len(self.feature_cols)} features")
        return self

    def run(self) -> dict:
        df        = self.df
        X         = df[self.feature_cols]
        y         = df["above_median_salary"]
        tradeoffs = {}

        base_model = _load_xgb("so")

        print("\n" + "=" * 60)
        print("SO 2024 -- MITIGATION (all sensitive definitions)")
        print("=" * 60)

        for key, col in SENSITIVE_DEFS.items():
            if col not in df.columns:
                print(f"  [skip] {col} not in DataFrame")
                continue

            g = df[col]
            X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
                X, y, g, test_size=0.25, stratify=y, random_state=42
            )

            # Baseline (loaded model, no mitigation)
            y_base   = base_model.predict(X_te)
            auc_base = roc_auc_score(y_te, base_model.predict_proba(X_te)[:, 1])
            dpd_base = _max_gap(y_base, g_te.values)

            # A: Reweighing
            sw = reweigh_samples(y_tr, g_tr)
            xgb_rw = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            xgb_rw.fit(X_tr, y_tr, sample_weight=sw)
            y_rw    = xgb_rw.predict(X_te)
            auc_rw  = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])
            dpd_rw  = _max_gap(y_rw, g_te.values)

            # B: Threshold calibration
            y_to   = threshold_calibrate(base_model, X_te, g_te)
            auc_to = roc_auc_score(y_te, base_model.predict_proba(X_te)[:, 1])
            dpd_to = _max_gap(y_to, g_te.values)

            tradeoffs[key] = {
                "baseline":  (auc_base, dpd_base),
                "reweigh":   (auc_rw,   dpd_rw),
                "threshold": (auc_to,   dpd_to),
            }

            print(f"\n  {key}:")
            for strat, (a, d) in tradeoffs[key].items():
                flag = "PASS" if d <= DPD_THRESHOLD else "FAIL"
                print(f"    {strat:12s}  AUC={a:.4f}  |DPD|={d:.4f}  [{flag}]")

        self.tradeoffs = tradeoffs

        rows = []
        for key, res in tradeoffs.items():
            for strat, (auc, dpd) in res.items():
                rows.append({"sensitive_def": key, "strategy": strat,
                             "auc": auc, "dpd": dpd,
                             "meets_threshold": dpd <= DPD_THRESHOLD})
        pd.DataFrame(rows).to_csv(f"{OUT}/so_mitigation_results.csv",
                                  index=False)
        print(f"\n  Saved -> {OUT}/so_mitigation_results.csv")
        return tradeoffs


# =============================================================================
# SECTION 2 -- GITHUB OSS SURVEY 2017
# =============================================================================

class GHMitigation:

    def load(self) -> "GHMitigation":
        if not os.path.exists(IN_GH):
            raise FileNotFoundError(
                f"[GH MIT] {IN_GH} not found -- run Notebook 2 first."
            )
        self.df = pd.read_csv(IN_GH)
        self.feature_cols = [c for c in GH_FEATURE_COLS
                             if c in self.df.columns]
        print(f"\n[GH MIT] Loaded {len(self.df):,} rows")
        return self

    def run(self) -> dict:
        df = self.df
        X  = df[self.feature_cols]
        y  = df["paid_contributor"]
        g  = df["gender_clean"]

        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.25, stratify=y, random_state=42
        )

        base_model = _load_xgb("gh")
        g_bin_te   = (g_te == "woman").astype(int)

        # Baseline
        y_base   = base_model.predict(X_te)
        auc_base = roc_auc_score(y_te, base_model.predict_proba(X_te)[:, 1])
        dpd_base = abs(float(demographic_parity_difference(
            y_te, y_base, sensitive_features=g_bin_te
        )))

        # A: Reweighing
        sw = reweigh_samples(y_tr, g_tr)
        xgb_rw = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_rw.fit(X_tr, y_tr, sample_weight=sw)
        y_rw    = xgb_rw.predict(X_te)
        auc_rw  = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])
        dpd_rw  = abs(float(demographic_parity_difference(
            y_te, y_rw, sensitive_features=g_bin_te
        )))

        # B: Threshold calibration
        y_to   = threshold_calibrate(base_model, X_te, g_te)
        auc_to = roc_auc_score(y_te, base_model.predict_proba(X_te)[:, 1])
        dpd_to = abs(float(demographic_parity_difference(
            y_te, y_to, sensitive_features=g_bin_te
        )))

        self.tradeoff = {
            "baseline":  (auc_base, dpd_base),
            "reweigh":   (auc_rw,   dpd_rw),
            "threshold": (auc_to,   dpd_to),
        }

        print("\n" + "=" * 60)
        print("GH OSS 2017 -- MITIGATION")
        print("=" * 60)
        for strat, (a, d) in self.tradeoff.items():
            flag = "PASS" if d <= DPD_THRESHOLD else "FAIL"
            print(f"  {strat:12s}  AUC={a:.4f}  |DPD|={d:.4f}  [{flag}]")

        rows = [{"strategy": s, "auc": a, "dpd": d,
                 "meets_threshold": d <= DPD_THRESHOLD}
                for s, (a, d) in self.tradeoff.items()]
        pd.DataFrame(rows).to_csv(f"{OUT}/gh_mitigation_results.csv",
                                  index=False)
        print(f"\n  Saved -> {OUT}/gh_mitigation_results.csv")
        return self.tradeoff


# =============================================================================
# PLOTS
# =============================================================================

def plot_mitigation_tradeoff(so_tradeoffs: dict, gh_tradeoff: dict):
    markers = {"baseline": "X", "reweigh": "s", "threshold": "o"}
    labels  = {"baseline": "Baseline", "reweigh": "Reweighing",
               "threshold": "Threshold Cal."}

    n_panels = len(so_tradeoffs) + 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels, 6),
                             facecolor=PALETTE["bg"])
    fig.suptitle(
        "Accuracy vs Fairness Trade-off\n"
        f"Dashed line = |DPD| <= {DPD_THRESHOLD} (target threshold)",
        fontweight="bold"
    )
    if n_panels == 1:
        axes = [axes]

    for i, (key, res) in enumerate(so_tradeoffs.items()):
        ax = axes[i]
        ax.set_facecolor("#fff8f0")
        for strat, (auc, dpd) in res.items():
            ax.scatter(dpd, auc, s=220, color=PALETTE["so"],
                       marker=markers[strat], zorder=5,
                       edgecolors="white", lw=1.2, label=labels[strat])
            ax.annotate(labels[strat], (dpd, auc),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.axvline(DPD_THRESHOLD, color=PALETTE["bad"],
                   linestyle="--", lw=1.5)
        ax.set_xlabel("|DPD|  ->  lower is fairer")
        ax.set_ylabel("ROC-AUC  ->  higher is better")
        ax.set_title(f"SO 2024 -- {key}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

    ax_gh = axes[-1]
    ax_gh.set_facecolor("#f0f4ff")
    for strat, (auc, dpd) in gh_tradeoff.items():
        ax_gh.scatter(dpd, auc, s=220, color=PALETTE["gh"],
                      marker=markers[strat], zorder=5,
                      edgecolors="white", lw=1.2, label=labels[strat])
        ax_gh.annotate(labels[strat], (dpd, auc),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax_gh.axvline(DPD_THRESHOLD, color=PALETTE["bad"],
                  linestyle="--", lw=1.5)
    ax_gh.set_xlabel("|DPD|  ->  lower is fairer")
    ax_gh.set_ylabel("ROC-AUC")
    ax_gh.set_title("GH OSS 2017 -- Gender", fontsize=9, fontweight="bold")
    ax_gh.legend(fontsize=7)

    plt.tight_layout()
    out = f"{OUT}/nb5_mitigation_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")


def plot_eu_compliance_table(so_tradeoffs: dict, gh_tradeoff: dict):
    so_post_ok = any(
        v["reweigh"][1] <= DPD_THRESHOLD or v["threshold"][1] <= DPD_THRESHOLD
        for v in so_tradeoffs.values()
    )
    gh_post_ok = (
        gh_tradeoff["reweigh"][1] <= DPD_THRESHOLD or
        gh_tradeoff["threshold"][1] <= DPD_THRESHOLD
    )

    rows = [
        ("Art. 10 -- bias-free training data",
         "NO -- age gap present in data",
         "NO -- gender gap present in data"),
        ("Art. 10 -- sensitive attribute declared",
         "YES -- Age (3 composite defs)",
         "YES -- Gender (binary)"),
        ("Art. 22 -- right to explanation (SHAP)",
         "YES -- SHAP in NB4",
         "YES -- SHAP in NB4"),
        ("Art. 22 -- human override mechanism",
         "NO -- not implemented",
         "NO -- not implemented"),
        (f"|DPD| <= {DPD_THRESHOLD} pre-mitigation",
         "NO -- all defs exceed threshold",
         "NO -- exceeds threshold"),
        (f"|DPD| <= {DPD_THRESHOLD} post-mitigation",
         "PARTIAL -- some defs/strats" if so_post_ok else "NO",
         "PARTIAL" if gh_post_ok else "NO"),
    ]
    col_labels = ["EU AI Act / GDPR Requirement",
                  "SO 2024 (age bias)",
                  "GH OSS 2017 (gender bias)"]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=PALETTE["bg"])
    ax.axis("off")
    fig.suptitle("EU AI Act / GDPR Art. 10 & Art. 22 Compliance Assessment",
                 fontweight="bold", y=1.02)

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        if r == 0:
            cell.set_facecolor("#dde4f0")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#f5f5f5")
        elif c > 0:
            t = cell.get_text().get_text()
            if t.startswith("YES"):     cell.set_facecolor("#d5f5e3")
            elif t.startswith("NO"):    cell.set_facecolor("#fde8e8")
            elif t.startswith("PART"):  cell.set_facecolor("#fef9e7")

    plt.tight_layout()
    out = f"{OUT}/nb5_eu_compliance_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NOTEBOOK 5 -- MITIGATION & COMPLIANCE REPORTING")
    print("=" * 65)

    so_m = SOMitigation().load()
    so_tradeoffs = so_m.run()

    gh_m = GHMitigation().load()
    gh_tradeoff = gh_m.run()

    try:
        plot_mitigation_tradeoff(so_tradeoffs, gh_tradeoff)
    except Exception as e:
        print(f"  Warning: tradeoff plot failed -- {e}")

    try:
        plot_eu_compliance_table(so_tradeoffs, gh_tradeoff)
    except Exception as e:
        print(f"  Warning: compliance table failed -- {e}")

    print("\n" + "=" * 65)
    print("  NOTEBOOK 5 COMPLETE")
    print("  -> outputs/nb5_mitigation_tradeoff.png")
    print("  -> outputs/nb5_eu_compliance_table.png")
    print("  -> outputs/so_mitigation_results.csv")
    print("  -> outputs/gh_mitigation_results.csv")
    print("=" * 65)
