"""

  NOTEBOOK 4 -- EXPLAINABILITY (SHAP)

  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/fcc_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/fcc_xgboost.pkl

  Outputs : outputs/so_shap_importance.csv
            outputs/fcc_shap_importance.csv
            outputs/nb4_shap_so.png
            outputs/nb4_shap_fcc.png
            outputs/nb4_shap_compare.png

  Why separate?

  SHAP TreeExplainer on XGBoost is slow. 
  
  Key interpretation
  For SO: if years_code / years_code_pro dominate importance it means
  the model uses seniority as a strong salary proxy, amplifying the
  age-based gap measured in NB3.

  For FCC: if months_programming / log_expected_earning dominate it
  means the model is largely using effort/aspiration signals rather
  than gender-correlated structural features,  expect a smaller
  amplification ratio than the SO case.

"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import (
    SO_BASE_FEATURES, FCC_FEATURE_COLS, ADULT_BASE_FEATURES,
    PALETTE,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_FCC  = "data/preprocessed/fcc_preprocessed.csv"
IN_ADULT = "data/preprocessed/adult_preprocessed.csv"
MDL_DIR = "outputs/models"
OUT     = "outputs"

os.makedirs(OUT, exist_ok=True)

# SO_BASE_FEATURES, FCC_FEATURE_COLS, ADULT_BASE_FEATURES, and PALETTE
# are all imported from config.py.


# HELPERS

def _load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_model(prefix: str) -> str:
    for name in [f"{prefix}_xgboost.pkl", f"{prefix}_xgb.pkl"]:
        p = os.path.join(MDL_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No XGBoost model found for prefix '{prefix}' in {MDL_DIR}. "
        f"Run Notebook 3 first."
    )


def _get_so_features(df: pd.DataFrame) -> list:
    devtype = [c for c in df.columns if c.startswith("devtype_")]
    return [c for c in SO_BASE_FEATURES + devtype if c in df.columns]


def _get_adult_features(df: pd.DataFrame) -> list:
    occ = [c for c in df.columns if c.startswith("occ_")]
    return [c for c in ADULT_BASE_FEATURES + occ if c in df.columns]


def run_shap(model, X: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """
    Compute SHAP values for an XGBoost model, handling both plain
    XGBClassifier and sklearn Pipeline wrappers.

    FIX (was: manually re-applied each pipeline step with incorrect
    pd.DataFrame reconstruction that broke on StandardScaler output):
    We now use model[:-1].transform(X) which correctly traverses all
    pre-processing steps via sklearn's built-in Pipeline slice syntax,
    preserving column alignment before passing to TreeExplainer.
    """
    if hasattr(model, "named_steps"):
        # Pipeline: extract the final classifier and transform X through
        # all preceding steps using sklearn's slice syntax (avoids the
        # manual step-by-step reconstruction).
        clf = model[-1]
        X_t = pd.DataFrame(
            model[:-1].transform(X),
            columns=X.columns,
            index=X.index,
        )
    else:
        clf = model
        X_t = X

    try:
        explainer  = shap.TreeExplainer(clf)
        shap_exp   = explainer(X_t)
        shap_array = shap_exp.values
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
    except Exception as e:
        print(f"  TreeExplainer failed ({e}) -- trying KernelExplainer ...")
        bg         = shap.sample(X_t, min(200, len(X_t)), random_state=42)
        explainer  = shap.KernelExplainer(clf.predict_proba, bg)
        raw        = explainer.shap_values(X_t, nsamples=100)
        shap_array = raw[1] if isinstance(raw, list) else raw

    importance = pd.Series(
        np.abs(shap_array).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)
    return importance, shap_array


def _save_bar_only(importance: pd.Series, color: str, highlight_keys: list,
                   title: str, out_path: str):
    top_n = importance.head(12)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    bar_colors = [color if any(k in c for k in highlight_keys) else "#4a4a4a"
                  for c in top_n.index]
    top_n.sort_values().plot.barh(
        ax=ax, color=list(reversed(bar_colors)), alpha=0.85
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance -- {title}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# SECTION 1 -- STACK OVERFLOW 2024

def so_shap_analysis() -> pd.Series:
    print("\n" + "=" * 60)
    print("SO 2024 -- SHAP EXPLAINABILITY")
    print("=" * 60)

    df    = pd.read_csv(IN_SO)
    fcols = _get_so_features(df)
    X     = df[fcols]

    model_path = _find_model("so")
    model      = _load_model(model_path)
    print(f"  Model: {model_path}")
    print(f"  Features: {len(fcols)}, Samples: {len(X):,}")

    importance, shap_array = run_shap(model, X)
    print(f"\n  Feature importances (mean |SHAP|):")
    print(importance.to_string())
    importance.to_csv(f"{OUT}/so_shap_importance.csv",
                      header=["mean_abs_shap"])

    sen_cols  = [c for c in ["years_code", "years_code_pro"]
                 if c in importance.index]
    sen_share = importance[sen_cols].sum() / importance.sum() * 100
    print(f"\n  Seniority feature share of total importance: {sen_share:.1f}%")
    print("  (high value means model proxies age through coding experience)")

    _save_bar_only(importance, PALETTE["so"],
                   ["years", "pro"],
                   "SO 2024 -- salary prediction",
                   f"{OUT}/nb4_shap_so.png")
    return importance


# SECTION 2 freeCodeCamp 2018

def fcc_shap_analysis() -> pd.Series:
    print("\n" + "=" * 60)
    print("FCC 2018 -- SHAP EXPLAINABILITY")
    print("=" * 60)

    df    = pd.read_csv(IN_FCC)
    fcols = [c for c in FCC_FEATURE_COLS if c in df.columns]
    X     = df[fcols]

    model_path = _find_model("fcc")
    model      = _load_model(model_path)
    print(f"  Model: {model_path}")
    print(f"  Features: {len(fcols)}, Samples: {len(X):,}")

    importance, shap_array = run_shap(model, X)
    print(f"\n  Feature importances (mean |SHAP|):")
    print(importance.to_string())
    importance.to_csv(f"{OUT}/fcc_shap_importance.csv",
                      header=["mean_abs_shap"])

    # Effort / experience features
    eff_cols  = [c for c in ["months_programming", "hours_learning_per_week",
                             "num_learning_resources", "log_expected_earning"]
                 if c in importance.index]
    eff_share = importance[eff_cols].sum() / importance.sum() * 100
    print(f"\n  Effort/experience feature share: {eff_share:.1f}%")

    _save_bar_only(importance, PALETTE["fcc"],
                   ["months", "hours", "expected"],
                   "FCC 2018 -- working-as-developer prediction",
                   f"{OUT}/nb4_shap_fcc.png")
    return importance


# SECTION 3 -- UCI ADULT / CENSUS INCOME 

def adult_shap_analysis() -> pd.Series:
    print("\n" + "=" * 60)
    print("ADULT -- SHAP EXPLAINABILITY")
    print("=" * 60)

    df    = pd.read_csv(IN_ADULT)
    fcols = _get_adult_features(df)
    X     = df[fcols]

    model_path = _find_model("adult")
    model      = _load_model(model_path)
    print(f"  Model: {model_path}")
    print(f"  Features: {len(fcols)}, Samples: {len(X):,}")

    importance, shap_array = run_shap(model, X)
    print(f"\n  Feature importances (mean |SHAP|):")
    print(importance.to_string())
    importance.to_csv(f"{OUT}/adult_shap_importance.csv",
                      header=["mean_abs_shap"])

    # `is_married` is the closest thing to a gender proxy left in this
    # feature set (relationship was deliberately excluded in NB2 for
    # being too direct a proxy). A high share here is the Adult-dataset
    # analog of SO's years_code_pro proxy-discrimination story.
    proxy_cols = [c for c in ["is_married", "hours_per_week", "education_num"]
                 if c in importance.index]
    proxy_share = importance[proxy_cols].sum() / importance.sum() * 100
    print(f"\n  is_married/hours/education share of total importance: "
          f"{proxy_share:.1f}%")

    _save_bar_only(importance, PALETTE["adult"],
                   ["married", "hours", "education"],
                   "UCI Adult -- income prediction",
                   f"{OUT}/nb4_shap_adult.png")
    return importance


# CROSS-DATASET COMPARISON

def plot_shap_comparison(so_imp: pd.Series = None, fcc_imp: pd.Series = None,
                         adult_imp: pd.Series = None):
    """
    N-way version: any of the three datasets may be None (skipped if
    its model/preprocessed CSV wasn't available), and the chart adapts
    to however many are actually present (minimum 2 to be worth plotting).
    """
    series = {}
    if so_imp is not None:
        series["SO 2024"] = (so_imp, PALETTE["so"])
    if fcc_imp is not None:
        series["FCC 2018"] = (fcc_imp, PALETTE["fcc"])
    if adult_imp is not None:
        series["UCI Adult"] = (adult_imp, PALETTE["adult"])

    if len(series) < 2:
        print("  Skipping cross-dataset SHAP comparison -- need at least "
              "2 datasets with importance available.")
        return

    normed = {name: (imp / imp.sum()).head(10) for name, (imp, _) in series.items()}
    all_feats = list(dict.fromkeys(
        [f for s in normed.values() for f in s.index]
    ))
    y = np.arange(len(all_feats))
    n = len(series)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["bg"])
    for i, (name, (imp, color)) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        ax.barh(y + offset, [normed[name].get(f, 0) for f in all_feats], width,
                label=name, color=color, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(all_feats, fontsize=9)
    ax.set_xlabel("Normalised mean |SHAP| (share of total importance)")
    ax.set_title(
        "Cross-Dataset SHAP Comparison\n"
        + " vs ".join(series.keys()),
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{OUT}/nb4_shap_compare.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")


# MAIN

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NOTEBOOK 4 -- EXPLAINABILITY (SHAP)")
    print("=" * 65)

    so_imp = None
    if os.path.exists(IN_SO) and os.path.exists(os.path.join(MDL_DIR, "so_xgboost.pkl")):
        so_imp = so_shap_analysis()
    else:
        print("[SO] Skipping -- preprocessed CSV or model missing.")

    fcc_imp = fcc_shap_analysis()

    adult_imp = None
    if os.path.exists(IN_ADULT) and os.path.exists(os.path.join(MDL_DIR, "adult_xgboost.pkl")):
        adult_imp = adult_shap_analysis()
    else:
        print("[ADULT] Skipping -- preprocessed CSV or model missing.")

    try:
        plot_shap_comparison(so_imp, fcc_imp, adult_imp)
    except Exception as e:
        print(f"  Warning: comparison chart failed -- {e}")

    print("\n  NOTEBOOK 4 COMPLETE")
