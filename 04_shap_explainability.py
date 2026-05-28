"""
=============================================================================
  NOTEBOOK 4 -- EXPLAINABILITY (SHAP)
=============================================================================
  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/gh_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/gh_xgboost.pkl

  Outputs : outputs/so_shap_importance.csv
            outputs/gh_shap_importance.csv
            outputs/nb4_shap_so.png
            outputs/nb4_shap_gh.png
            outputs/nb4_shap_compare.png

  Why separate?
  -------------
  SHAP TreeExplainer on XGBoost is slow. Keeping it separate means you
  can re-run explainability without re-training and vice versa.

  Key interpretation
  ------------------
  For SO: if years_code / years_code_pro dominate importance it means
  the model uses seniority as a strong salary proxy -- amplifying the
  age-based gap measured in NB3.
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
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_GH   = "data/preprocessed/gh_preprocessed.csv"
MDL_DIR = "outputs/models"
OUT     = "outputs"
os.makedirs(OUT, exist_ok=True)

PALETTE = {"so": "#f48024", "gh": "#24292e", "bg": "#fafafa"}

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


# =============================================================================
# HELPERS
# =============================================================================

def _load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_model(prefix: str) -> str:
    """
    Look for models saved by NB3. NB3 saves as <prefix>_xgboost.pkl.
    Falls back to legacy name <prefix>_xgb.pkl.
    """
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
    """
    Compute SHAP values using TreeExplainer.
    If the model is a sklearn Pipeline, the classifier is unwrapped and
    X is transformed through the pre-processing steps first.

    Returns
    -------
    importance : pd.Series  mean absolute SHAP per feature, sorted desc
    shap_array : np.ndarray  raw SHAP values (n_samples x n_features)
    """
    # Unwrap Pipeline
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        steps = list(model.named_steps.items())[:-1]  # all except last
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
        # Handle multi-output (binary classification sometimes returns 2-d)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
    except Exception as e:
        print(f"  TreeExplainer failed ({e}) -- trying KernelExplainer on "
              f"a 200-row sample (this will be slow)...")
        bg         = shap.sample(X_t, min(200, len(X_t)), random_state=42)
        explainer  = shap.KernelExplainer(clf.predict_proba, bg)
        raw        = explainer.shap_values(X_t, nsamples=100)
        # KernelExplainer returns list[array] for multi-class; take class-1
        shap_array = raw[1] if isinstance(raw, list) else raw

    importance = pd.Series(
        np.abs(shap_array).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    return importance, shap_array


def _save_shap_plots(importance: pd.Series, shap_array: np.ndarray,
                     X: pd.DataFrame, title: str, color: str,
                     highlight_keys: list, out_path: str):
    """
    Two-panel figure: horizontal bar importance + beeswarm summary.
    Beeswarm is drawn into its own figure then composited to avoid the
    plt.sca() / SHAP axes-ownership conflict.
    """
    top_n   = importance.head(12)
    n_feats = len(top_n)

    # --- Panel A: bar chart ---------------------------------------------------
    fig_bar, ax_bar = plt.subplots(figsize=(8, max(4, n_feats * 0.4)),
                                   facecolor=PALETTE["bg"])
    bar_colors = [color if any(k in c for k in highlight_keys) else "#4a4a4a"
                  for c in top_n.index]
    top_n.sort_values().plot.barh(
        ax=ax_bar, color=list(reversed(bar_colors)), alpha=0.85
    )
    ax_bar.set_xlabel("Mean |SHAP value|")
    ax_bar.set_title(f"Global Feature Importance\n{title}", fontweight="bold",
                     fontsize=9)
    mean_imp = top_n.mean()
    ax_bar.axvline(mean_imp, color="gray", linestyle="--", lw=1,
                   alpha=0.6, label=f"Mean={mean_imp:.4f}")
    ax_bar.legend(fontsize=8)
    fig_bar.tight_layout()

    # --- Panel B: SHAP summary (beeswarm) ------------------------------------
    # Draw into its own figure via show=False, then save separately.
    # We composite them side by side using a wider figure.
    import io
    buf = io.BytesIO()
    fig_bee = plt.figure(figsize=(8, max(4, n_feats * 0.4)),
                         facecolor=PALETTE["bg"])
    shap.summary_plot(
        shap_array[:, :len(top_n)] if shap_array.shape[1] >= len(top_n)
        else shap_array,
        X[top_n.index[:shap_array.shape[1]]],
        feature_names=list(top_n.index[:shap_array.shape[1]]),
        max_display=12,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Beeswarm\n(colour = feature value)", fontsize=9,
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(buf, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig_bee)
    buf.seek(0)

    # --- Composite -----------------------------------------------------------
    from PIL import Image
    img_bar = _fig_to_pil(fig_bar)
    img_bee = Image.open(buf)

    # Resize to same height
    h = max(img_bar.height, img_bee.height)
    img_bar = img_bar.resize(
        (int(img_bar.width * h / img_bar.height), h), Image.LANCZOS
    )
    img_bee = img_bee.resize(
        (int(img_bee.width * h / img_bee.height), h), Image.LANCZOS
    )
    composite = Image.new("RGB", (img_bar.width + img_bee.width, h),
                          (250, 250, 250))
    composite.paste(img_bar, (0, 0))
    composite.paste(img_bee, (img_bar.width, 0))
    composite.save(out_path, dpi=(150, 150))
    plt.close("all")
    print(f"  Saved -> {out_path}")


def _fig_to_pil(fig):
    """Convert a matplotlib Figure to a PIL Image."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _pil_available() -> bool:
    try:
        from PIL import Image
        return True
    except ImportError:
        return False


# =============================================================================
# SECTION 1 -- STACK OVERFLOW 2024
# =============================================================================

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

    if _pil_available():
        _save_shap_plots(
            importance, shap_array, X,
            title="SO 2024 -- salary prediction",
            color=PALETTE["so"],
            highlight_keys=["years", "pro"],
            out_path=f"{OUT}/nb4_shap_so.png",
        )
    else:
        # Fallback: save bar chart only (no PIL dependency)
        _save_bar_only(importance, PALETTE["so"],
                       ["years", "pro"],
                       "SO 2024 -- salary prediction",
                       f"{OUT}/nb4_shap_so.png")

    return importance


# =============================================================================
# SECTION 2 -- GITHUB OSS SURVEY 2017
# =============================================================================

def gh_shap_analysis() -> pd.Series:
    print("\n" + "=" * 60)
    print("GH OSS 2017 -- SHAP EXPLAINABILITY")
    print("=" * 60)

    df    = pd.read_csv(IN_GH)
    fcols = [c for c in GH_FEATURE_COLS if c in df.columns]
    X     = df[fcols]

    model_path = _find_model("gh")
    model      = _load_model(model_path)
    print(f"  Model: {model_path}")

    importance, shap_array = run_shap(model, X)

    print(f"\n  Feature importances (mean |SHAP|):")
    print(importance.to_string())
    importance.to_csv(f"{OUT}/gh_shap_importance.csv",
                      header=["mean_abs_shap"])

    neg_cols  = [c for c in ["had_negative_exp", "had_harassment"]
                 if c in importance.index]
    neg_share = importance[neg_cols].sum() / importance.sum() * 100
    print(f"\n  Negative-experience feature share: {neg_share:.1f}%")

    if _pil_available():
        _save_shap_plots(
            importance, shap_array, X,
            title="GH OSS 2017 -- paid-contributor prediction",
            color=PALETTE["gh"],
            highlight_keys=["negative", "harassment"],
            out_path=f"{OUT}/nb4_shap_gh.png",
        )
    else:
        _save_bar_only(importance, PALETTE["gh"],
                       ["negative", "harassment"],
                       "GH OSS 2017 -- paid-contributor prediction",
                       f"{OUT}/nb4_shap_gh.png")

    return importance


def _save_bar_only(importance: pd.Series, color: str, highlight_keys: list,
                   title: str, out_path: str):
    """Simple fallback when PIL is not available -- bar chart only."""
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
    print(f"  Saved (bar only, install Pillow for beeswarm) -> {out_path}")


# =============================================================================
# CROSS-DATASET COMPARISON
# =============================================================================

def plot_shap_comparison(so_imp: pd.Series, gh_imp: pd.Series):
    so_n = (so_imp / so_imp.sum()).head(10)
    gh_n = (gh_imp / gh_imp.sum()).head(10)

    all_feats = list(dict.fromkeys(list(so_n.index) + list(gh_n.index)))
    y = np.arange(len(all_feats))

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["bg"])
    ax.barh(y - 0.2, [so_n.get(f, 0) for f in all_feats], 0.35,
            label="SO 2024", color=PALETTE["so"], alpha=0.85)
    ax.barh(y + 0.2, [gh_n.get(f, 0) for f in all_feats], 0.35,
            label="GH OSS 2017", color=PALETTE["gh"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(all_feats, fontsize=9)
    ax.set_xlabel("Normalised mean |SHAP| (share of total importance)")
    ax.set_title(
        "Cross-Dataset SHAP Comparison\n"
        "SO 2024 (salary prediction) vs GH OSS 2017 (paid-contributor)",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{OUT}/nb4_shap_compare.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NOTEBOOK 4 -- EXPLAINABILITY (SHAP)")
    print("=" * 65)

    so_imp = so_shap_analysis()
    gh_imp = gh_shap_analysis()

    try:
        plot_shap_comparison(so_imp, gh_imp)
    except Exception as e:
        print(f"  Warning: comparison chart failed -- {e}")

    print("\n  NOTEBOOK 4 COMPLETE")
