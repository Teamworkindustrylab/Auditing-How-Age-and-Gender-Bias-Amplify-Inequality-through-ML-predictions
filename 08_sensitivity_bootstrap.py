"""

  NOTEBOOK 8 -- SENSITIVITY & BOOTSTRAP ANALYSIS

  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/fcc_preprocessed.csv
            data/preprocessed/adult_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/fcc_xgboost.pkl
            outputs/models/adult_xgboost.pkl

  Outputs : outputs/nb8_bootstrap_ci.csv
            outputs/nb8_sensitivity_per_dataset.csv
            outputs/nb8_subgroup_reweigh.csv
            outputs/nb8_bootstrap_forest.png
            outputs/nb8_sensitivity_chart.png
            outputs/nb8_subgroup_reweigh_chart.png

  Implements three open items from FINDINGS.md section 9, ALL three across
  all three datasets (each section is symmetric and uses the same machinery
  regardless of which dataset has data on disk).

  (1) BOOTSTRAPPED 95% CONFIDENCE INTERVALS on DPD and amplification.
      For each (dataset, model) we resample the test set B=1000 times
      with replacement and recompute DPD / amplification on each resample.
      The CI tells us whether each dataset's amplification verdict is
      statistically distinguishable from "no amplification":
        - FCC: expected CI strictly below 1.0 (under-amplification real)
        - SO:  expected CI strictly above 1.0 (amplification real)
        - Adult: expected CI may straddle 1.0 (mirrors raw gap)

  (2) PROXY-FEATURE SENSITIVITY: drop one candidate proxy per dataset.
      Each dataset has one feature flagged in NB4 (SHAP) as the most
      likely vector for laundering the protected attribute. We retrain
      XGBoost without that feature and report the change in DPD,
      amplification, and AUC.
        - FCC: drop log_expected_earning (NB4 FINDINGS section 3 flag)
        - SO:  drop years_code_pro       (SHAP top-1 feature, age proxy)
        - Adult: drop is_married         (NB4 documents this as the
                                          closest gender proxy left in
                                          the feature set after the
                                          relationship column was excluded)
      Reading the result: if amplification drops AND the verdict
      flips (e.g. CI now includes 1.0), the original conclusion was
      partly driven by the proxy feature; if amplification is roughly
      unchanged, the conclusion is robust.

  (3) SUBGROUP-AWARE REWEIGHING (intersectional mitigation), all 3 datasets:
        - FCC:   gender x experience-bracket    (the project headline)
        - SO:    age_group x experience-bracket (SO's analog axis)
        - Adult: gender x age-bracket           (Adult's analog axis)
      NB5's reweighing uses the single sensitive attribute alone, so the
      intersectional gap survives mitigation there. NB8 uses the FULL
      joint subgroup as the reweighing key and recomputes the
      intersectional max DPD. The question for each dataset is whether
      intersectional-aware mitigation can bring it under 0.10.

  Compliance threshold : |DPD| <= 0.10  (consistent across NB3 / NB5 / NB6)
"""

from __future__ import annotations
import os
import pickle
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score
from xgboost                 import XGBClassifier
from fairlearn.metrics       import demographic_parity_difference

from config import (
    SO_BASE_FEATURES, FCC_FEATURE_COLS, ADULT_BASE_FEATURES,
    DPD_THRESHOLD, PALETTE,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO    = "data/preprocessed/so_preprocessed.csv"
IN_FCC   = "data/preprocessed/fcc_preprocessed.csv"
IN_ADULT = "data/preprocessed/adult_preprocessed.csv"
OUT      = "outputs"
MDL_DIR  = "outputs/models"
os.makedirs(OUT, exist_ok=True)

# SO_BASE_FEATURES, FCC_FEATURE_COLS, ADULT_BASE_FEATURES, DPD_THRESHOLD,
# and PALETTE are all imported from config.py.
N_BOOTSTRAP = 1000


# SHARED HELPERS

def _load_xgb(prefix: str):
    for name in [f"{prefix}_xgboost.pkl", f"{prefix}_xgb.pkl"]:
        p = os.path.join(MDL_DIR, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def _max_pairwise_gap(y_pred: np.ndarray, g_series: pd.Series) -> float:
    rates = []
    for grp in np.unique(g_series):
        mask = (g_series.values == grp) if hasattr(g_series, "values") \
               else (g_series == grp)
        if mask.sum() >= 10:
            rates.append(y_pred[mask].mean())
    if len(rates) < 2:
        return float("nan")
    return float(max(rates) - min(rates))


# SECTION 1 -- BOOTSTRAPPED 95% CIs

def bootstrap_dpd(y_true: np.ndarray, y_pred: np.ndarray,
                  g_series: pd.Series, data_gap: float,
                  n_boot: int = N_BOOTSTRAP, seed: int = 42) -> dict:
    """Resample (y_true, y_pred, g_series) WITH REPLACEMENT n_boot times
    and recompute DPD + amplification on each resample. Returns mean and
    2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    g_arr = g_series.values if hasattr(g_series, "values") else np.asarray(g_series)

    dpds, amps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        g_b  = g_arr[idx]

        # If a group has too few samples in this resample, skip it
        groups = np.unique(g_b)
        rates = []
        for grp in groups:
            mask = (g_b == grp)
            if mask.sum() >= 5:  # tiny floor; resamples can be uneven
                rates.append(yp_b[mask].mean())
        if len(rates) < 2:
            continue
        gap = float(max(rates) - min(rates))
        dpds.append(gap)
        amps.append(gap / (abs(data_gap) + 1e-9))

    return dict(
        dpd_mean=float(np.mean(dpds)) if dpds else float("nan"),
        dpd_lo=float(np.percentile(dpds, 2.5)) if dpds else float("nan"),
        dpd_hi=float(np.percentile(dpds, 97.5)) if dpds else float("nan"),
        amp_mean=float(np.mean(amps)) if amps else float("nan"),
        amp_lo=float(np.percentile(amps, 2.5)) if amps else float("nan"),
        amp_hi=float(np.percentile(amps, 97.5)) if amps else float("nan"),
        n_valid=len(dpds),
    )


def _bootstrap_one_dataset(name: str, csv_path: str, model_prefix: str,
                            features: list, target: str,
                            sensitive_col: str) -> list:
    """Returns a list of dict rows ready for the bootstrap CSV."""
    if not os.path.exists(csv_path):
        print(f"  [skip] {name}: {csv_path} not found")
        return []
    model = _load_xgb(model_prefix)
    if model is None:
        print(f"  [skip] {name}: no XGBoost model in {MDL_DIR}")
        return []

    df = pd.read_csv(csv_path)
    # Some datasets use occ_/devtype_ dummies discovered at load time
    if model_prefix == "so":
        features = features + [c for c in df.columns if c.startswith("devtype_")]
    elif model_prefix == "adult":
        features = features + [c for c in df.columns if c.startswith("occ_")]
    features = [c for c in features if c in df.columns]

    X = df[features]
    y = df[target]
    g = df[sensitive_col]

    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=0.25, stratify=y, random_state=42
    )

    # Data gap from training split (defines amplification denominator)
    train_rates = [y_tr[g_tr == grp].mean() for grp in np.unique(g_tr)]
    data_gap = float(max(train_rates) - min(train_rates))

    y_pred = model.predict(X_te)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    res = bootstrap_dpd(y_te.values, y_pred, g_te, data_gap)

    point_dpd = _max_pairwise_gap(y_pred, g_te)
    point_amp = point_dpd / (abs(data_gap) + 1e-9)

    print(f"\n  {name}  (XGBoost, sensitive={sensitive_col})")
    print(f"    Data gap            : {data_gap:.4f}")
    print(f"    Point DPD           : {point_dpd:.4f}")
    print(f"    Point amplification : {point_amp:.3f}x")
    print(f"    DPD 95% CI          : [{res['dpd_lo']:.4f}, {res['dpd_hi']:.4f}]")
    print(f"    Amp 95% CI          : [{res['amp_lo']:.3f}x, {res['amp_hi']:.3f}x]")
    print(f"    AUC                 : {auc:.4f}")
    print(f"    Test n / bootstrap n: {len(y_te):,} / {res['n_valid']:,}")

    # Quick interpretation: does CI exclude 1.0 ?
    if not np.isnan(res["amp_hi"]):
        if res["amp_hi"] < 1.0:
            verdict = "compresses (CI strictly below 1.0)"
        elif res["amp_lo"] > 1.0:
            verdict = "amplifies (CI strictly above 1.0)"
        else:
            verdict = "inconclusive (CI straddles 1.0)"
        print(f"    Verdict             : {verdict}")
    else:
        verdict = "n/a"

    return [{
        "dataset":   name,
        "model":     "XGBoost",
        "sensitive": sensitive_col,
        "data_gap":  data_gap,
        "point_dpd": point_dpd,
        "dpd_lo":    res["dpd_lo"],
        "dpd_hi":    res["dpd_hi"],
        "point_amp": point_amp,
        "amp_lo":    res["amp_lo"],
        "amp_hi":    res["amp_hi"],
        "auc":       auc,
        "verdict":   verdict,
        "n_test":    int(len(y_te)),
        "n_boot":    int(res["n_valid"]),
    }]


def run_bootstrap_all() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  SECTION 1 -- BOOTSTRAPPED 95% CIs (B=" + str(N_BOOTSTRAP) + ")")
    print("=" * 65)
    rows = []
    rows += _bootstrap_one_dataset(
        "Stack Overflow 2024", IN_SO, "so",
        SO_BASE_FEATURES, "above_median_salary", "age_group",
    )
    rows += _bootstrap_one_dataset(
        "freeCodeCamp 2018", IN_FCC, "fcc",
        FCC_FEATURE_COLS, "paid_contributor", "gender_clean",
    )
    rows += _bootstrap_one_dataset(
        "UCI Adult", IN_ADULT, "adult",
        ADULT_BASE_FEATURES, "above_50k", "gender_clean",
    )
    if not rows:
        print("  No datasets available -- skipping bootstrap output.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/nb8_bootstrap_ci.csv", index=False)
    print(f"\n  Saved -> {OUT}/nb8_bootstrap_ci.csv")
    return df


def plot_bootstrap_forest(boot_df: pd.DataFrame):
    if len(boot_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 1.0 * len(boot_df) + 2)),
                             facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Bootstrapped 95% CIs (B={N_BOOTSTRAP})\n"
        "Point estimate (dot) + CI (whiskers)",
        fontweight="bold"
    )

    color_map = {"Stack Overflow 2024": PALETTE["so"],
                 "freeCodeCamp 2018":   PALETTE["fcc"],
                 "UCI Adult":           PALETTE["adult"]}

    # DPD panel
    ax = axes[0]
    y = np.arange(len(boot_df))
    for i, (_, r) in enumerate(boot_df.iterrows()):
        c = color_map.get(r["dataset"], "#444")
        ax.errorbar(r["point_dpd"], i,
                    xerr=[[r["point_dpd"] - r["dpd_lo"]],
                          [r["dpd_hi"] - r["point_dpd"]]],
                    fmt="o", color=c, ecolor=c, capsize=5, markersize=10,
                    linewidth=2)
    ax.axvline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle="--", lw=1.5,
               label=f"|DPD| <= {DPD_THRESHOLD}")
    ax.set_yticks(y)
    ax.set_yticklabels(boot_df["dataset"], fontsize=9)
    ax.set_xlabel("|DPD| (lower is fairer)")
    ax.set_title("DPD with 95% CI", fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Amplification panel
    ax2 = axes[1]
    for i, (_, r) in enumerate(boot_df.iterrows()):
        c = color_map.get(r["dataset"], "#444")
        ax2.errorbar(r["point_amp"], i,
                     xerr=[[r["point_amp"] - r["amp_lo"]],
                           [r["amp_hi"] - r["point_amp"]]],
                     fmt="o", color=c, ecolor=c, capsize=5, markersize=10,
                     linewidth=2)
    ax2.axvline(1.0, color="black", linestyle="--", lw=1.5,
                label="No amplification (1.0x)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(boot_df["dataset"], fontsize=9)
    ax2.set_xlabel("Amplification ratio")
    ax2.set_title("Amplification with 95% CI", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    out = f"{OUT}/nb8_bootstrap_forest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# SECTION 2 -- PROXY-FEATURE SENSITIVITY (per-dataset)

# Per-dataset config: which proxy feature to drop, and why it was picked.
# Each entry's "drop_feature" is the strongest gender/age proxy left in
# that dataset's feature set after NB2's deliberate exclusions, as
# documented in NB4 SHAP comments and FINDINGS.md section 3.
SENSITIVITY_CONFIGS = [
    {
        "name":          "Stack Overflow 2024",
        "key":           "so",
        "csv":           IN_SO,
        "features":      SO_BASE_FEATURES,
        "drop_feature":  "years_code_pro",
        "target":        "above_median_salary",
        "sensitive":     "age_group",
        "unprivileged":  "older",   # for binarised DPD; works either way
        "rationale":     "SHAP top-1 feature; correlates strongly with age",
    },
    {
        "name":          "freeCodeCamp 2018",
        "key":           "fcc",
        "csv":           IN_FCC,
        "features":      FCC_FEATURE_COLS,
        "drop_feature":  "log_expected_earning",
        "target":        "paid_contributor",
        "sensitive":     "gender_clean",
        "unprivileged":  "woman",
        "rationale":     "FINDINGS section 3 flag; plausibly downstream of gender",
    },
    {
        "name":          "UCI Adult",
        "key":           "adult",
        "csv":           IN_ADULT,
        "features":      ADULT_BASE_FEATURES,
        "drop_feature":  "is_married",
        "target":        "above_50k",
        "sensitive":     "gender_clean",
        "unprivileged":  "woman",
        "rationale":     "NB4 documents this as the closest gender proxy left "
                         "in the feature set after `relationship` was excluded",
    },
]


def _run_one_sensitivity(cfg: dict) -> list[dict]:
    """Run the full vs drop-feature comparison for one dataset.
    Returns a list of two row dicts (one per variant) or [] if the
    dataset's preprocessed CSV isn't available in this environment."""
    if not os.path.exists(cfg["csv"]):
        print(f"  [skip] {cfg['name']}: {cfg['csv']} not found")
        return []

    df = pd.read_csv(cfg["csv"])

    # Add dataset-specific dummy columns to the feature list (devtype_/occ_)
    features = list(cfg["features"])
    if cfg["key"] == "so":
        features = features + [c for c in df.columns if c.startswith("devtype_")]
    elif cfg["key"] == "adult":
        features = features + [c for c in df.columns if c.startswith("occ_")]
    features = [c for c in features if c in df.columns]

    full_feats    = features
    drop_feat     = cfg["drop_feature"]
    if drop_feat not in full_feats:
        print(f"  [skip] {cfg['name']}: feature `{drop_feat}` not in column set; "
              f"can't run drop-variant.")
        return []
    dropped_feats = [c for c in full_feats if c != drop_feat]

    rows = []
    for variant_name, feats in [("full", full_feats),
                                 (f"drop_{drop_feat}", dropped_feats)]:
        X = df[feats]
        y = df[cfg["target"]]
        g = df[cfg["sensitive"]]

        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.25, stratify=y, random_state=42
        )
        train_rates = [y_tr[g_tr == grp].mean() for grp in np.unique(g_tr)]
        data_gap = float(max(train_rates) - min(train_rates))

        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb.fit(X_tr, y_tr)
        y_pred = xgb.predict(X_te)
        auc    = roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1])
        dpd    = _max_pairwise_gap(y_pred, g_te)
        amp    = dpd / (abs(data_gap) + 1e-9)
        boot   = bootstrap_dpd(y_te.values, y_pred, g_te, data_gap)

        rows.append({
            "dataset":      cfg["name"],
            "key":          cfg["key"],
            "variant":      variant_name,
            "dropped":      "" if variant_name == "full" else drop_feat,
            "n_features":   len(feats),
            "data_gap":     data_gap,
            "point_dpd":    dpd,
            "dpd_lo":       boot["dpd_lo"],
            "dpd_hi":       boot["dpd_hi"],
            "point_amp":    amp,
            "amp_lo":       boot["amp_lo"],
            "amp_hi":       boot["amp_hi"],
            "auc":          auc,
        })

    # Per-dataset interpretation block
    full_row = next(r for r in rows if r["variant"] == "full")
    drop_row = next(r for r in rows if r["variant"] != "full")
    delta_amp = drop_row["point_amp"] - full_row["point_amp"]
    delta_auc = drop_row["auc"]       - full_row["auc"]

    print(f"\n  {cfg['name']}  -- dropping `{drop_feat}`")
    print(f"    Rationale  : {cfg['rationale']}")
    print(f"    Full       : DPD={full_row['point_dpd']:.4f}, "
          f"amp={full_row['point_amp']:.3f}x  "
          f"[CI {full_row['amp_lo']:.3f}x, {full_row['amp_hi']:.3f}x], "
          f"AUC={full_row['auc']:.4f}")
    print(f"    Drop       : DPD={drop_row['point_dpd']:.4f}, "
          f"amp={drop_row['point_amp']:.3f}x  "
          f"[CI {drop_row['amp_lo']:.3f}x, {drop_row['amp_hi']:.3f}x], "
          f"AUC={drop_row['auc']:.4f}")
    print(f"    Delta amp  : {delta_amp:+.3f}x")
    print(f"    Delta AUC  : {delta_auc:+.4f}")

    # Verdict on whether the original amplification finding is robust
    # to dropping the candidate proxy
    if full_row["point_amp"] < 1.0 and drop_row["point_amp"] < 1.0 \
            and drop_row["amp_hi"] < 1.0:
        verdict = "Under-amplification verdict SURVIVES the drop."
    elif full_row["point_amp"] > 1.0 and drop_row["point_amp"] > 1.0 \
            and drop_row["amp_lo"] > 1.0:
        verdict = "Amplification verdict SURVIVES the drop."
    elif (full_row["point_amp"] - 1.0) * (drop_row["point_amp"] - 1.0) < 0:
        verdict = "Verdict FLIPS sign once proxy is removed."
    else:
        verdict = "Verdict becomes INCONCLUSIVE (CI crosses 1.0)."
    print(f"    Verdict    : {verdict}")
    for r in rows:
        r["verdict"] = verdict
    return rows


def run_sensitivity_all() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  SECTION 2 -- PROXY-FEATURE SENSITIVITY (per dataset)")
    print("=" * 65)
    all_rows = []
    for cfg in SENSITIVITY_CONFIGS:
        all_rows.extend(_run_one_sensitivity(cfg))
    if not all_rows:
        print("  No datasets available -- skipping sensitivity output.")
        return pd.DataFrame()

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(f"{OUT}/nb8_sensitivity_per_dataset.csv", index=False)
    print(f"\n  Saved -> {OUT}/nb8_sensitivity_per_dataset.csv")
    return out_df


def plot_sensitivity(sens_df: pd.DataFrame):
    """Per-dataset two-bar chart showing full vs drop-proxy. Each
    dataset gets one subplot; missing datasets just don't appear."""
    if len(sens_df) == 0:
        return
    datasets = list(sens_df["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5),
                              facecolor=PALETTE["bg"], squeeze=False)
    color_map = {"Stack Overflow 2024": PALETTE["so"],
                 "freeCodeCamp 2018":   PALETTE["fcc"],
                 "UCI Adult":           PALETTE["adult"]}

    for ax, ds in zip(axes[0], datasets):
        sub = sens_df[sens_df["dataset"] == ds]
        labels = [v if v == "full" else v.replace("drop_", "drop ")
                  for v in sub["variant"]]
        col = color_map.get(ds, "#444")
        x = np.arange(len(sub))
        w = 0.38
        bars_dpd = ax.bar(x - w/2, sub["point_dpd"], w, color=col, alpha=0.85,
                          label="|DPD|")
        bars_amp = ax.bar(x + w/2, sub["point_amp"], w, color="#888888",
                          alpha=0.85, label="Amplification")
        for bar, v in zip(bars_dpd, sub["point_dpd"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)
        for bar, v in zip(bars_amp, sub["point_amp"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}x", ha="center", fontsize=9)
        ax.axhline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle=":", lw=1,
                   alpha=0.6, label=f"|DPD|<={DPD_THRESHOLD}")
        ax.axhline(1.0, color="black", linestyle="--", lw=1, alpha=0.6,
                   label="amp=1.0")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(ds, fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(0.6, sub["point_amp"].max() * 1.25))
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Proxy-Feature Sensitivity -- full vs drop candidate proxy",
                 fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/nb8_sensitivity_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# SECTION 3 -- SUBGROUP-AWARE REWEIGHING

def reweigh_samples_intersectional(y: pd.Series, g: pd.Series) -> np.ndarray:
    """Same Kamiran-Calders formula as NB5, but `g` is the FULL
    intersectional subgroup string (e.g. 'man_senior'), not the
    sensitive attribute alone. This pushes weight toward the
    historically smallest x lowest-positive subgroups (e.g. junior
    women) which is exactly the cell driving the project's headline
    finding."""
    df_w = pd.DataFrame({"y": y.values, "g": g.values})
    n = len(df_w)
    lut = {}
    for (gi, yi), grp in df_w.groupby(["g", "y"]):
        n_g = (df_w["g"] == gi).sum()
        n_y = (df_w["y"] == yi).sum()
        lut[(gi, yi)] = (n_g * n_y) / (n * max(len(grp), 1))
    return df_w.apply(lambda r: lut.get((r["g"], r["y"]), 1.0), axis=1).values


def _fcc_subgroup_key(df: pd.DataFrame) -> pd.Series:
    def _b(m):
        if m < 24:  return "junior"
        if m < 72:  return "mid"
        return "senior"
    return df["gender_clean"] + "_" + df["months_programming"].apply(_b)


def _so_subgroup_key(df: pd.DataFrame) -> pd.Series:
    """SO has age (binary) + years_code_pro. We bucket experience the same
    way FCC does (junior/mid/senior in years rather than months) and cross
    it with the age_group column NB3 already uses as the sensitive axis.
    Falls back to age column if age_group isn't materialised."""
    def _b(yrs):
        if yrs < 3:   return "junior"
        if yrs < 10:  return "mid"
        return "senior"
    if "age_group" in df.columns:
        axis = df["age_group"].astype(str)
    elif "age" in df.columns:
        axis = df["age"].apply(lambda a: "older" if a >= 35 else "younger")
    else:
        raise KeyError("SO data has neither age_group nor age column.")
    bracket = df["years_code_pro"].fillna(0).apply(_b)
    return axis.astype(str) + "_" + bracket


def run_subgroup_reweigh_so() -> dict | None:
    print("\n" + "=" * 65)
    print("  SECTION 3a -- SUBGROUP-AWARE REWEIGHING (Stack Overflow)")
    print("=" * 65)
    if not os.path.exists(IN_SO):
        print(f"  [skip] {IN_SO} not found.")
        return None

    df = pd.read_csv(IN_SO)
    devtype_cols = [c for c in df.columns if c.startswith("devtype_")]
    feats = [c for c in SO_BASE_FEATURES + devtype_cols if c in df.columns]
    X = df[feats]
    y = df["above_median_salary"]
    g_int = _so_subgroup_key(df)

    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g_int, test_size=0.25, stratify=y, random_state=42
    )

    xgb_base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_base.fit(X_tr, y_tr)
    y_base = xgb_base.predict(X_te)

    sw = reweigh_samples_intersectional(y_tr, g_tr)
    xgb_rw = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_rw.fit(X_tr, y_tr, sample_weight=sw)
    y_rw = xgb_rw.predict(X_te)

    isec_base = _max_pairwise_gap(y_base, g_te)
    isec_rw   = _max_pairwise_gap(y_rw,   g_te)

    # Single-axis DPD: recover the age axis by stripping the experience
    # suffix (e.g. 'experienced_senior' -> 'experienced').
    # classify_age() always returns 'young' or 'experienced', so we
    # check for 'experienced' directly and never fall through to an
    # unreliable fallback.
    age_te = g_te.str.split("_").str[0]
    age_bin = (age_te == "experienced").astype(int)
    single_base = abs(float(demographic_parity_difference(
        y_te, y_base, sensitive_features=age_bin
    )))
    single_rw = abs(float(demographic_parity_difference(
        y_te, y_rw, sensitive_features=age_bin
    )))

    auc_base = roc_auc_score(y_te, xgb_base.predict_proba(X_te)[:, 1])
    auc_rw   = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])

    print(f"\n  SO -- intersectional reweighing (age x experience-bracket)")
    print(f"    Baseline: age DPD={single_base:.4f}, "
          f"intersectional={isec_base:.4f}, AUC={auc_base:.4f}")
    print(f"    Reweigh : age DPD={single_rw:.4f}, "
          f"intersectional={isec_rw:.4f}, AUC={auc_rw:.4f}")
    print(f"    Delta intersectional DPD: {isec_rw - isec_base:+.4f}")
    return dict(
        dataset="SO",
        baseline_gender_dpd=single_base,
        baseline_isec_dpd=isec_base,
        baseline_auc=auc_base,
        reweighed_gender_dpd=single_rw,
        reweighed_isec_dpd=isec_rw,
        reweighed_auc=auc_rw,
        delta_isec=isec_rw - isec_base,
        baseline_meets_threshold=bool(isec_base <= DPD_THRESHOLD),
        reweighed_meets_threshold=bool(isec_rw <= DPD_THRESHOLD),
    )


def run_subgroup_reweigh_fcc() -> dict | None:
    print("\n" + "=" * 65)
    print("  SECTION 3a -- SUBGROUP-AWARE REWEIGHING (FCC)")
    print("=" * 65)
    if not os.path.exists(IN_FCC):
        print(f"  [skip] {IN_FCC} not found.")
        return None

    df    = pd.read_csv(IN_FCC)
    feats = [c for c in FCC_FEATURE_COLS if c in df.columns]
    X     = df[feats]
    y     = df["paid_contributor"]
    g_int = _fcc_subgroup_key(df)

    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g_int, test_size=0.25, stratify=y, random_state=42
    )

    # Baseline: just retrain unweighted to keep comparison apples-to-apples
    xgb_base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_base.fit(X_tr, y_tr)
    y_base = xgb_base.predict(X_te)

    # Intersectional reweighing
    sw = reweigh_samples_intersectional(y_tr, g_tr)
    xgb_rw = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_rw.fit(X_tr, y_tr, sample_weight=sw)
    y_rw = xgb_rw.predict(X_te)

    # Intersectional max DPD before/after
    isec_base = _max_pairwise_gap(y_base, g_te)
    isec_rw   = _max_pairwise_gap(y_rw,   g_te)

    # Single-axis (gender) DPD before/after -- so we can show the
    # mitigation didn't break gender-only compliance
    gender_te = g_te.str.split("_").str[0]
    gender_base = abs(float(demographic_parity_difference(
        y_te, y_base, sensitive_features=(gender_te == "woman").astype(int)
    )))
    gender_rw = abs(float(demographic_parity_difference(
        y_te, y_rw, sensitive_features=(gender_te == "woman").astype(int)
    )))

    auc_base = roc_auc_score(y_te, xgb_base.predict_proba(X_te)[:, 1])
    auc_rw   = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])

    print(f"\n  FCC -- intersectional reweighing (gender x experience-bracket)")
    print(f"    Baseline (unweighted retrain):")
    print(f"      Gender-only DPD       : {gender_base:.4f}")
    print(f"      Intersectional max DPD: {isec_base:.4f}")
    print(f"      AUC                   : {auc_base:.4f}")
    print(f"    Intersectional reweighing:")
    print(f"      Gender-only DPD       : {gender_rw:.4f}")
    print(f"      Intersectional max DPD: {isec_rw:.4f}")
    print(f"      AUC                   : {auc_rw:.4f}")
    print(f"    Delta intersectional DPD: {isec_rw - isec_base:+.4f}")
    print(f"    Intersectional meets {DPD_THRESHOLD}: "
          f"baseline={isec_base <= DPD_THRESHOLD}, "
          f"reweighed={isec_rw <= DPD_THRESHOLD}")
    return dict(
        dataset="FCC",
        baseline_gender_dpd=gender_base,
        baseline_isec_dpd=isec_base,
        baseline_auc=auc_base,
        reweighed_gender_dpd=gender_rw,
        reweighed_isec_dpd=isec_rw,
        reweighed_auc=auc_rw,
        delta_isec=isec_rw - isec_base,
        baseline_meets_threshold=bool(isec_base <= DPD_THRESHOLD),
        reweighed_meets_threshold=bool(isec_rw <= DPD_THRESHOLD),
    )


def run_subgroup_reweigh_adult() -> dict | None:
    print("\n" + "=" * 65)
    print("  SECTION 3b -- SUBGROUP-AWARE REWEIGHING (Adult)")
    print("=" * 65)
    if not os.path.exists(IN_ADULT):
        print(f"  [skip] {IN_ADULT} not found.")
        return None

    df = pd.read_csv(IN_ADULT)
    occ_cols = [c for c in df.columns if c.startswith("occ_")]
    feats = [c for c in ADULT_BASE_FEATURES + occ_cols if c in df.columns]
    X = df[feats]
    y = df["above_50k"]
    if "gender_age_bracket" in df.columns:
        g_int = df["gender_age_bracket"]
    else:
        # Fallback: build it inline
        def _b(a):
            if a < 30: return "junior"
            if a < 50: return "mid"
            return "senior"
        g_int = df["gender_clean"] + "_" + df["age"].apply(_b)

    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g_int, test_size=0.25, stratify=y, random_state=42
    )

    xgb_base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_base.fit(X_tr, y_tr)
    y_base = xgb_base.predict(X_te)

    sw = reweigh_samples_intersectional(y_tr, g_tr)
    xgb_rw = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_rw.fit(X_tr, y_tr, sample_weight=sw)
    y_rw = xgb_rw.predict(X_te)

    isec_base = _max_pairwise_gap(y_base, g_te)
    isec_rw   = _max_pairwise_gap(y_rw,   g_te)

    gender_te = g_te.str.split("_").str[0]
    gender_base = abs(float(demographic_parity_difference(
        y_te, y_base, sensitive_features=(gender_te == "woman").astype(int)
    )))
    gender_rw = abs(float(demographic_parity_difference(
        y_te, y_rw, sensitive_features=(gender_te == "woman").astype(int)
    )))

    auc_base = roc_auc_score(y_te, xgb_base.predict_proba(X_te)[:, 1])
    auc_rw   = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])

    print(f"\n  ADULT -- intersectional reweighing (gender x age bracket)")
    print(f"    Baseline: gender DPD={gender_base:.4f}, "
          f"intersectional={isec_base:.4f}, AUC={auc_base:.4f}")
    print(f"    Reweigh : gender DPD={gender_rw:.4f}, "
          f"intersectional={isec_rw:.4f}, AUC={auc_rw:.4f}")
    print(f"    Delta intersectional DPD: {isec_rw - isec_base:+.4f}")
    return dict(
        dataset="Adult",
        baseline_gender_dpd=gender_base,
        baseline_isec_dpd=isec_base,
        baseline_auc=auc_base,
        reweighed_gender_dpd=gender_rw,
        reweighed_isec_dpd=isec_rw,
        reweighed_auc=auc_rw,
        delta_isec=isec_rw - isec_base,
        baseline_meets_threshold=bool(isec_base <= DPD_THRESHOLD),
        reweighed_meets_threshold=bool(isec_rw <= DPD_THRESHOLD),
    )


def save_and_plot_subgroup_reweigh(results: list[dict]):
    results = [r for r in results if r is not None]
    if not results:
        print("  [skip] no subgroup reweighing results to plot.")
        return
    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/nb8_subgroup_reweigh.csv", index=False)
    print(f"\n  Saved -> {OUT}/nb8_subgroup_reweigh.csv")

    color_map = {"SO": PALETTE["so"], "FCC": PALETTE["fcc"],
                 "Adult": PALETTE["adult"]}

    fig, ax = plt.subplots(figsize=(10, max(5, 1.8 * len(results) + 2)),
                           facecolor=PALETTE["bg"])
    x = np.arange(len(df))
    w = 0.35
    bars_base = ax.bar(x - w/2, df["baseline_isec_dpd"], w,
                       label="Baseline (no reweigh)",
                       color="#aaaaaa", alpha=0.85)
    bars_rw = ax.bar(x + w/2, df["reweighed_isec_dpd"], w,
                     label="Intersectional reweigh",
                     color=[color_map.get(d, "#444") for d in df["dataset"]],
                     alpha=0.85)
    for bar, v in zip(bars_base, df["baseline_isec_dpd"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)
    for bar, v in zip(bars_rw, df["reweighed_isec_dpd"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle="--", lw=1.5,
               label=f"|DPD| <= {DPD_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], fontsize=10)
    ax.set_ylabel("Intersectional max DPD")
    ax.set_title(
        "Intersectional max DPD: baseline vs subgroup-aware reweighing\n"
        "Reweighing key: age x experience (SO) / gender x experience (FCC) / "
        "gender x age (Adult)",
        fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    out = f"{OUT}/nb8_subgroup_reweigh_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# MAIN

def main() -> int:
    print("\n" + "=" * 65)
    print("  NOTEBOOK 8 -- SENSITIVITY & BOOTSTRAP ANALYSIS")
    print("=" * 65)

    boot_df = run_bootstrap_all()
    if len(boot_df) > 0:
        plot_bootstrap_forest(boot_df)

    sens_df = run_sensitivity_all()
    if len(sens_df) > 0:
        plot_sensitivity(sens_df)

    sub_results = [
        run_subgroup_reweigh_so(),
        run_subgroup_reweigh_fcc(),
        run_subgroup_reweigh_adult(),
    ]
    save_and_plot_subgroup_reweigh(sub_results)

    print("\n  NOTEBOOK 8 COMPLETE")
    print("  -> outputs/nb8_bootstrap_ci.csv")
    print("  -> outputs/nb8_sensitivity_per_dataset.csv")
    print("  -> outputs/nb8_subgroup_reweigh.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())