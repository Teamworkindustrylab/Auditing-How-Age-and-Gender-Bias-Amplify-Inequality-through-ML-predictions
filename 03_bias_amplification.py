"""

  NOTEBOOK 3 -- BIAS BASELINE & AMPLIFICATION MEASUREMENT

  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/fcc_preprocessed.csv

  Outputs : outputs/so_bias_results.csv
            outputs/fcc_bias_results.csv
            outputs/models/so_logistic_regression.pkl
            outputs/models/so_xgboost.pkl
            outputs/models/fcc_logistic_regression.pkl
            outputs/models/fcc_xgboost.pkl
            outputs/nb3_amplification_chart.png

  Sensitive definitions evaluated
  -------------------------------
  SO:
    S1  age_group       -- binary  young / experienced
    S2  age_exp_pro     -- 6-way   age x YearsCodePro bracket
    S3  age_exp_total   -- 6-way   age x YearsCode bracket
  FCC:
    gender_clean        -- binary  man / woman

  Amplification ratio = model_max_gap / data_max_gap
  >1.0 means the model widens the gap that already existed in the data.

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
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import classification_report, roc_auc_score
from xgboost                 import XGBClassifier
from fairlearn.metrics       import (
    demographic_parity_difference,
    equalized_odds_difference,
)

from config import (
    SO_BASE_FEATURES, FCC_FEATURE_COLS, ADULT_BASE_FEATURES,
    SENSITIVE_DEFS, DPD_THRESHOLD, PALETTE,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_FCC  = "data/preprocessed/fcc_preprocessed.csv"
IN_ADULT = "data/preprocessed/adult_preprocessed.csv"
OUT     = "outputs"
MDL_DIR = "outputs/models"
os.makedirs(OUT,     exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

# SO_BASE_FEATURES, FCC_FEATURE_COLS, ADULT_BASE_FEATURES, SENSITIVE_DEFS,
# DPD_THRESHOLD, and PALETTE are all imported from config.py.


# SHARED HELPERS

def _make_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
    }


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def train_and_eval(X_tr, X_te, y_tr, y_te) -> dict:
    fitted = {}
    for name, m in _make_models().items():
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_prob = m.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
        print(f"\n  {name}  AUC={auc:.4f}")
        print(classification_report(y_te, y_pred, digits=3))
        fitted[name] = dict(model=m, y_pred=y_pred, y_prob=y_prob, auc=auc)
    return fitted


def measure_bias_multigroup(y_true, y_pred, g_series) -> dict:
    groups      = np.unique(g_series)
    group_rates = {grp: y_pred[g_series == grp].mean() for grp in groups}
    rates       = np.array(list(group_rates.values()))
    max_gap     = float(rates.max() - rates.min())

    if len(groups) == 2:
        minority = pd.Series(group_rates).idxmin()
        g_bin    = (g_series == minority).astype(int)
        dpd      = float(demographic_parity_difference(
                       y_true, y_pred, sensitive_features=g_bin))
        eod      = float(equalized_odds_difference(
                       y_true, y_pred, sensitive_features=g_bin))
    else:
        dpd = max_gap
        eod = float("nan")

    return dict(group_rates=group_rates, max_gap=max_gap, dpd=dpd, eod=eod)


# SECTION 1 -- STACK OVERFLOW 2024 (UNCHANGED)

class SOBiasAmplification:

    def load(self, path: str = IN_SO) -> "SOBiasAmplification":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[SO] {path} not found -- run Notebook 2 first."
            )
        df = pd.read_csv(path)
        devtype_cols = [c for c in df.columns if c.startswith("devtype_")]
        self.feature_cols = [c for c in SO_BASE_FEATURES + devtype_cols
                             if c in df.columns]
        self.df = df
        print(f"[SO] Loaded {len(df):,} rows | "
              f"{len(self.feature_cols)} feature columns")
        return self

    def data_baseline(self) -> dict:
        df = self.df
        print("\n" + "=" * 60)
        print("SO -- STAGE 1: DATA BIAS BASELINE")
        print("=" * 60)
        baselines = {}
        for key, col in SENSITIVE_DEFS.items():
            if col not in df.columns:
                continue
            rates = (
                df.groupby(col)["above_median_salary"]
                  .agg(["mean", "count"])
                  .rename(columns={"mean": "rate", "count": "n"})
                  .sort_values("rate", ascending=False)
            )
            gap = float(rates["rate"].max() - rates["rate"].min())
            print(f"\n  {key}  (max gap = {gap:.4f})")
            print(rates.to_string())
            baselines[key] = {"rates": rates, "gap": gap, "col": col}
        self.baselines = baselines
        return baselines

    def train(self) -> "SOBiasAmplification":
        X = self.df[self.feature_cols]
        y = self.df["above_median_salary"]
        idx = self.df.index.to_series()
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y, idx, test_size=0.25, stratify=y, random_state=42
        )
        print("\n" + "=" * 60)
        print("SO -- STAGE 2: MODEL TRAINING")
        print("=" * 60)
        print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

        self.fitted  = train_and_eval(X_tr, X_te, y_tr, y_te)
        self.X_tr, self.X_te = X_tr, X_te
        self.y_tr, self.y_te = y_tr, y_te
        self.idx_te = idx_te

        for name, res in self.fitted.items():
            p = f"{MDL_DIR}/so_{_slug(name)}.pkl"
            with open(p, "wb") as f:
                pickle.dump(res["model"], f)
            print(f"  Model saved -> {p}")
        return self

    def measure_amplification(self) -> dict:
        amp_results = {}
        print("\n" + "=" * 60)
        print("SO -- STAGE 3: BIAS AMPLIFICATION")
        print("=" * 60)

        for key, base in self.baselines.items():
            col   = base["col"]
            gap_d = base["gap"]
            g_te  = self.df.loc[self.idx_te, col]
            amp_results[key] = {}

            for mname, res in self.fitted.items():
                bm  = measure_bias_multigroup(
                    self.y_te.values, res["y_pred"], g_te.values
                )
                amp = bm["max_gap"] / (gap_d + 1e-9)
                amp_results[key][mname] = {
                    "dpd": bm["dpd"], "eod": bm["eod"],
                    "max_gap": bm["max_gap"], "amplification": amp,
                    "auc": res["auc"], "group_rates": bm["group_rates"],
                }
                eod_str = ("N/A (multi-group)"
                           if np.isnan(bm["eod"])
                           else f"{bm['eod']:+.4f}")
                print(f"\n  {key} | {mname}")
                print(f"    Data gap        : {gap_d:.4f}")
                print(f"    Model max gap   : {bm['max_gap']:.4f}")
                print(f"    Amplification   : {amp:.2f}x")
                print(f"    DPD             : {bm['dpd']:+.4f}")
                print(f"    EOD             : {eod_str}")

        self.amp_results = amp_results
        rows = []
        for key, models in amp_results.items():
            for mname, v in models.items():
                rows.append({
                    "sensitive_def": key, "model": mname,
                    "data_gap":      self.baselines[key]["gap"],
                    "model_max_gap": v["max_gap"],
                    "dpd":           v["dpd"],
                    "eod":           v["eod"],
                    "amplification": v["amplification"],
                    "auc":           v["auc"],
                })
        pd.DataFrame(rows).to_csv(f"{OUT}/so_bias_results.csv", index=False)
        print(f"\n  Saved -> {OUT}/so_bias_results.csv")
        return amp_results


# SECTION 2 -- freeCodeCamp 2018 (REPLACES GH OSS 2017)

class FCCBiasAmplification:

    def load(self, path: str = IN_FCC) -> "FCCBiasAmplification":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[FCC] {path} not found -- run Notebook 2 first."
            )
        self.df = pd.read_csv(path)
        print(f"\n[FCC] Loaded {len(self.df):,} rows from {path}")
        return self

    def data_baseline(self) -> float:
        df = self.df
        print("\n" + "=" * 60)
        print("FCC -- STAGE 1: DATA BIAS BASELINE")
        print("=" * 60)
        stats = (
            df.groupby("gender_clean")["paid_contributor"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "paid_rate", "count": "n"})
        )
        print(stats.to_string())
        m = stats.loc["man",   "paid_rate"] if "man"   in stats.index else np.nan
        w = stats.loc["woman", "paid_rate"] if "woman" in stats.index else np.nan
        gap = float(m - w)
        print(f"\n  Gap (man - woman): {gap:.4f}")
        self.gap_data = gap
        return gap

    def train(self) -> "FCCBiasAmplification":
        feats = [c for c in FCC_FEATURE_COLS if c in self.df.columns]
        X = self.df[feats]
        y = self.df["paid_contributor"]
        g = self.df["gender_clean"]

        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.25, stratify=y, random_state=42
        )
        print("\n" + "=" * 60)
        print("FCC -- STAGE 2: MODEL TRAINING")
        print("=" * 60)
        print(f"  Features used: {feats}")
        print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

        self.fitted = train_and_eval(X_tr, X_te, y_tr, y_te)
        self.X_tr, self.X_te = X_tr, X_te
        self.y_tr, self.y_te = y_tr, y_te
        self.g_tr, self.g_te = g_tr, g_te
        self.fcc_feats = feats

        for name, res in self.fitted.items():
            p = f"{MDL_DIR}/fcc_{_slug(name)}.pkl"
            with open(p, "wb") as f:
                pickle.dump(res["model"], f)
            print(f"  Model saved -> {p}")
        return self

    def measure_amplification(self) -> dict:
        results = {}
        print("\n" + "=" * 60)
        print("FCC -- STAGE 3: BIAS AMPLIFICATION")
        print("=" * 60)
        g_bin = (self.g_te == "woman").astype(int).values
        for name, res in self.fitted.items():
            dpd = float(demographic_parity_difference(
                self.y_te.values, res["y_pred"],
                sensitive_features=g_bin))
            eod = float(equalized_odds_difference(
                self.y_te.values, res["y_pred"],
                sensitive_features=g_bin))
            amp = abs(dpd) / (abs(self.gap_data) + 1e-9)
            print(f"\n  {name}: DPD={dpd:+.4f}  EOD={eod:+.4f}  "
                  f"Amplification={amp:.2f}x")
            results[name] = dict(dpd=dpd, eod=eod, amplification=amp,
                                 auc=res["auc"])
        self.bias_results = results
        rows = [{"model": k, "data_gap": self.gap_data,
                 "dpd": v["dpd"], "eod": v["eod"],
                 "amplification": v["amplification"], "auc": v["auc"]}
                for k, v in results.items()]
        pd.DataFrame(rows).to_csv(f"{OUT}/fcc_bias_results.csv", index=False)
        print(f"\n  Saved -> {OUT}/fcc_bias_results.csv")
        return results


# SECTION 3 -- UCI ADULT / CENSUS INCOME (NEW)
# ----------------------------------------------------------------------
# Structurally mirrors FCCBiasAmplification: single binary sensitive
# attribute (gender_clean), fairlearn DPD/EOD directly (no multi-group
# max-gap fallback needed, unlike SO's S1/S2/S3 setup).

class AdultBiasAmplification:

    def load(self, path: str = IN_ADULT) -> "AdultBiasAmplification":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ADULT] {path} not found -- run Notebook 2 first."
            )
        df = pd.read_csv(path)
        occ_cols = [c for c in df.columns if c.startswith("occ_")]
        self.feature_cols = [c for c in ADULT_BASE_FEATURES + occ_cols
                             if c in df.columns]
        self.df = df
        print(f"\n[ADULT] Loaded {len(df):,} rows | "
              f"{len(self.feature_cols)} feature columns")
        return self

    def data_baseline(self) -> float:
        df = self.df
        print("\n" + "=" * 60)
        print("ADULT -- STAGE 1: DATA BIAS BASELINE")
        print("=" * 60)
        stats = (
            df.groupby("gender_clean")["above_50k"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "above_50k_rate", "count": "n"})
        )
        print(stats.to_string())
        m = stats.loc["man",   "above_50k_rate"] if "man"   in stats.index else np.nan
        w = stats.loc["woman", "above_50k_rate"] if "woman" in stats.index else np.nan
        gap = float(m - w)
        print(f"\n  Gap (man - woman): {gap:.4f}")
        self.gap_data = gap
        return gap

    def train(self) -> "AdultBiasAmplification":
        X = self.df[self.feature_cols]
        y = self.df["above_50k"]
        g = self.df["gender_clean"]

        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.25, stratify=y, random_state=42
        )
        print("\n" + "=" * 60)
        print("ADULT -- STAGE 2: MODEL TRAINING")
        print("=" * 60)
        print(f"  Features used: {self.feature_cols}")
        print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

        self.fitted = train_and_eval(X_tr, X_te, y_tr, y_te)
        self.X_tr, self.X_te = X_tr, X_te
        self.y_tr, self.y_te = y_tr, y_te
        self.g_tr, self.g_te = g_tr, g_te

        for name, res in self.fitted.items():
            p = f"{MDL_DIR}/adult_{_slug(name)}.pkl"
            with open(p, "wb") as f:
                pickle.dump(res["model"], f)
            print(f"  Model saved -> {p}")
        return self

    def measure_amplification(self) -> dict:
        results = {}
        print("\n" + "=" * 60)
        print("ADULT -- STAGE 3: BIAS AMPLIFICATION")
        print("=" * 60)
        g_bin = (self.g_te == "woman").astype(int).values
        for name, res in self.fitted.items():
            dpd = float(demographic_parity_difference(
                self.y_te.values, res["y_pred"],
                sensitive_features=g_bin))
            eod = float(equalized_odds_difference(
                self.y_te.values, res["y_pred"],
                sensitive_features=g_bin))
            amp = abs(dpd) / (abs(self.gap_data) + 1e-9)
            print(f"\n  {name}: DPD={dpd:+.4f}  EOD={eod:+.4f}  "
                  f"Amplification={amp:.2f}x")
            results[name] = dict(dpd=dpd, eod=eod, amplification=amp,
                                 auc=res["auc"])
        self.bias_results = results
        rows = [{"model": k, "data_gap": self.gap_data,
                 "dpd": v["dpd"], "eod": v["eod"],
                 "amplification": v["amplification"], "auc": v["auc"]}
                for k, v in results.items()]
        pd.DataFrame(rows).to_csv(f"{OUT}/adult_bias_results.csv", index=False)
        print(f"\n  Saved -> {OUT}/adult_bias_results.csv")
        return results


# COMPARISON CHART

def plot_amplification_comparison(so: SOBiasAmplification,
                                  fcc: FCCBiasAmplification):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Bias Amplification Ratio by Model & Sensitive-Attribute Definition\n"
        "(ratio > 1.0 means the model widens the gap found in the data)",
        fontweight="bold"
    )

    ax = axes[0]
    ax.set_facecolor("#fff8f0")
    skeys = list(so.amp_results.keys())
    mnames = list(so.fitted.keys())
    x, w = np.arange(len(skeys)), 0.35
    for i, mname in enumerate(mnames):
        amps = [so.amp_results[sk][mname]["amplification"] for sk in skeys]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, amps, w,
                      label=mname,
                      color=[PALETTE["so"], "#1a5276"][i], alpha=0.85)
        for bar, val in zip(bars, amps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}x", ha="center", fontsize=8)
    ax.axhline(1.0, color="black", linestyle="--", lw=1.5,
               label="No amplification (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in skeys], fontsize=8)
    ax.set_ylabel("Amplification ratio")
    ax.set_title("SO 2024 -- Age / Age x Experience", fontweight="bold")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.set_facecolor("#f0f4ff")
    gkeys = list(fcc.bias_results.keys())
    colors = [PALETTE["fcc"], "#2c7a2c"]
    for i, (mname, v) in enumerate(fcc.bias_results.items()):
        ax2.bar(i, v["amplification"], 0.5,
                label=mname, color=colors[i % len(colors)], alpha=0.85)
        ax2.text(i, v["amplification"] + 0.02,
                 f"{v['amplification']:.2f}x", ha="center", fontsize=9)
    ax2.axhline(1.0, color="black", linestyle="--", lw=1.5,
                label="No amplification (1.0x)")
    ax2.set_xticks(range(len(gkeys)))
    ax2.set_xticklabels([k.replace(" ", "\n") for k in gkeys], fontsize=9)
    ax2.set_ylabel("Amplification ratio")
    ax2.set_title("FCC 2018 -- Gender", fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = f"{OUT}/nb3_amplification_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")


def plot_amplification_adult(adult: "AdultBiasAmplification"):
    """
    Single-panel chart for Adult, parallel in style to the FCC panel
    above. Kept as its own function rather than folded into
    plot_amplification_comparison() so the existing SO/FCC chart logic
    -- already validated -- doesn't need to be touched.
    """
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor("#eef9f0")
    gkeys = list(adult.bias_results.keys())
    colors = [PALETTE["adult"], "#1b5e20"]
    for i, (mname, v) in enumerate(adult.bias_results.items()):
        ax.bar(i, v["amplification"], 0.5, label=mname,
               color=colors[i % len(colors)], alpha=0.85)
        ax.text(i, v["amplification"] + 0.02, f"{v['amplification']:.2f}x",
                ha="center", fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", lw=1.5,
               label="No amplification (1.0x)")
    ax.set_xticks(range(len(gkeys)))
    ax.set_xticklabels([k.replace(" ", "\n") for k in gkeys], fontsize=9)
    ax.set_ylabel("Amplification ratio")
    ax.set_title("UCI Adult -- Gender", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = f"{OUT}/nb3_amplification_adult_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")


# MAIN

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NOTEBOOK 3 -- BIAS BASELINE & AMPLIFICATION MEASUREMENT")
    print("=" * 65)

    if os.path.exists(IN_SO):
        so = SOBiasAmplification().load()
        so.data_baseline()
        so.train()
        so.measure_amplification()
    else:
        print(f"[SO] {IN_SO} missing -- skipping SO section.")
        so = None

    fcc = FCCBiasAmplification().load()
    fcc.data_baseline()
    fcc.train()
    fcc.measure_amplification()

    if so is not None:
        try:
            plot_amplification_comparison(so, fcc)
        except Exception as e:
            print(f"  Warning: comparison chart failed -- {e}")

    # UCI Adult / Census Income (new)
    if os.path.exists(IN_ADULT):
        adult = AdultBiasAmplification().load()
        adult.data_baseline()
        adult.train()
        adult.measure_amplification()
        try:
            plot_amplification_adult(adult)
        except Exception as e:
            print(f"  Warning: Adult amplification chart failed -- {e}")
    else:
        print(f"[ADULT] {IN_ADULT} missing -- skipping Adult section.")

    print("\n  NOTEBOOK 3 COMPLETE")
