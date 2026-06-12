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

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_FCC  = "data/preprocessed/fcc_preprocessed.csv"
MDL_DIR = "outputs/models"
OUT     = "outputs"

os.makedirs(OUT, exist_ok=True)

PALETTE = {"so": "#f48024", "fcc": "#0a0a23", "bg": "#fafafa"}

SO_BASE_FEATURES = [
    "ed_level_enc", "is_employed", "is_remote", "is_student",
    "years_code", "years_code_pro",
]
FCC_FEATURE_COLS = [
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


def run_shap(model, X: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        steps = list(model.named_steps.items())[:-1]
        X_t = X.copy()
        for _, step in steps:
            X_t = pd.DataFrame(step.transform(X_t), columns=X.columns,
                                index=X.index)
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


# CROSS-DATASET COMPARISON

def plot_shap_comparison(so_imp: pd.Series, fcc_imp: pd.Series):
    so_n = (so_imp / so_imp.sum()).head(10)
    fcc_n = (fcc_imp / fcc_imp.sum()).head(10)

    all_feats = list(dict.fromkeys(list(so_n.index) + list(fcc_n.index)))
    y = np.arange(len(all_feats))

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["bg"])
    ax.barh(y - 0.2, [so_n.get(f, 0) for f in all_feats], 0.35,
            label="SO 2024", color=PALETTE["so"], alpha=0.85)
    ax.barh(y + 0.2, [fcc_n.get(f, 0) for f in all_feats], 0.35,
            label="FCC 2018", color=PALETTE["fcc"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(all_feats, fontsize=9)
    ax.set_xlabel("Normalised mean |SHAP| (share of total importance)")
    ax.set_title(
        "Cross-Dataset SHAP Comparison\n"
        "SO 2024 (salary) vs FCC 2018 (working-as-developer)",
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

    if so_imp is not None:
        try:
            plot_shap_comparison(so_imp, fcc_imp)
        except Exception as e:
            print(f"  Warning: comparison chart failed -- {e}")

    print("\n  NOTEBOOK 4 COMPLETE")
