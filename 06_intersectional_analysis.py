"""
=============================================================================
  NOTEBOOK 6 -- INTERSECTIONAL BIAS ANALYSIS
=============================================================================
  Extends the project by examining bias at the intersection of BOTH age
  and gender simultaneously, rather than auditing each dimension separately.

  Research question:
    Do young women or older women face compounded disadvantage beyond what
    age-alone or gender-alone audits would reveal?

  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/gh_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/gh_xgboost.pkl

  Outputs : outputs/nb6_intersectional_dpd.png
              — grouped bar chart: DPD per (age x gender) subgroup
            outputs/nb6_intersectional_heatmap.png
              — heatmap of positive-prediction rates across subgroups
            outputs/so_intersectional_results.csv
            outputs/gh_intersectional_results.csv

  Method:
    Intersectional fairness (Foulds et al. 2020) evaluates whether a model
    disadvantages people who belong to *multiple* protected groups at once.
    We create four subgroups:
        young_man   / young_woman
        experienced_man / experienced_woman
    and compute DPD for each pair, plus an "intersectional gap" defined as
    the max subgroup gap minus the worst single-attribute gap already
    reported in NB3.  A positive intersectional gap means compounding harm.

  Reference:
    Foulds, J. R., Islam, R., Keya, K. N., & Pan, S. (2020).
    An intersectional definition of fairness. ICDE 2020.
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
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_GH   = "data/preprocessed/gh_preprocessed.csv"
OUT     = "outputs"
MDL_DIR = "outputs/models"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "so":       "#f48024",
    "gh":       "#24292e",
    "young_m":  "#3498db",
    "young_f":  "#e74c3c",
    "exp_m":    "#2980b9",
    "exp_f":    "#c0392b",
    "ok":       "#27ae60",
    "bad":      "#e74c3c",
    "bg":       "#fafafa",
}

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

DPD_THRESHOLD = 0.10   # consistent with NB3/NB5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def intersectional_subgroups(df, age_col, gender_col):
    """
    Return a Series of four-way subgroup labels:
        young_man / young_woman / experienced_man / experienced_woman
    Rows where either attribute is missing are labeled 'unknown'.
    """
    labels = pd.Series("unknown", index=df.index)
    for age_tag, age_val in [("young", 0), ("experienced", 1)]:
        for gen_tag, gen_val in [("man", 0), ("woman", 1)]:
            mask = (df[age_col] == age_val) & (df[gender_col] == gen_val)
            labels[mask] = f"{age_tag}_{gen_tag}"
    return labels


def subgroup_positive_rates(y_pred, subgroups):
    """Positive-prediction rate per subgroup."""
    result = {}
    for sg in subgroups.unique():
        if sg == "unknown":
            continue
        mask = subgroups == sg
        if mask.sum() == 0:
            continue
        result[sg] = float(y_pred[mask].mean())
    return result


def intersectional_dpd(pos_rates):
    """
    Maximum pairwise DPD across all subgroup combinations.
    Returns the (max_gap, worst_pair) tuple.
    """
    groups = list(pos_rates.keys())
    max_gap = 0.0
    worst_pair = (None, None)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gap = abs(pos_rates[groups[i]] - pos_rates[groups[j]])
            if gap > max_gap:
                max_gap = gap
                worst_pair = (groups[i], groups[j])
    return max_gap, worst_pair


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# STACK OVERFLOW  — intersectional analysis
# =============================================================================
print("\n" + "=" * 70)
print("STACK OVERFLOW — Intersectional Analysis")
print("=" * 70)

so = pd.read_csv(IN_SO)

# Gender column: expected to be 'is_female' (1 = woman, 0 = man)
# Age column: 'age_group' (0 = young, 1 = experienced) from NB3
if "is_female" not in so.columns or "age_group" not in so.columns:
    raise ValueError(
        "Expected columns 'is_female' and 'age_group' in so_preprocessed.csv. "
        "Check that NB1/NB2 ran successfully."
    )

SO_FEATURES = SO_BASE_FEATURES + [
    c for c in so.columns
    if c.startswith("devtype_") and c in so.columns
]
SO_FEATURES = [c for c in SO_FEATURES if c in so.columns]

X_so = so[SO_FEATURES].fillna(0)
y_so = so["high_salary"].values

X_so_tr, X_so_te, y_so_tr, y_so_te, idx_tr, idx_te = train_test_split(
    X_so, y_so, so.index, test_size=0.2, random_state=42
)
so_test = so.loc[idx_te].reset_index(drop=True)

# Load the pre-trained XGBoost (same split as NB3 via random_state=42)
so_xgb = load_model(os.path.join(MDL_DIR, "so_xgboost.pkl"))
y_so_pred = so_xgb.predict(X_so_te.reset_index(drop=True))

# Build intersectional subgroups
so_subgroups = intersectional_subgroups(
    so_test, age_col="age_group", gender_col="is_female"
)

so_pos_rates = subgroup_positive_rates(
    pd.Series(y_so_pred), so_subgroups
)
so_max_dpd, so_worst_pair = intersectional_dpd(so_pos_rates)

print("\nPositive-prediction rates by intersectional subgroup (SO):")
for sg, rate in sorted(so_pos_rates.items()):
    flag = " ← lowest" if rate == min(so_pos_rates.values()) else \
           " ← highest" if rate == max(so_pos_rates.values()) else ""
    print(f"  {sg:25s}: {rate:.4f}{flag}")

print(f"\nMax intersectional DPD : {so_max_dpd:.4f}")
print(f"Worst subgroup pair     : {so_worst_pair[0]}  vs  {so_worst_pair[1]}")
print(f"Meets threshold (≤{DPD_THRESHOLD}): {so_max_dpd <= DPD_THRESHOLD}")

# NB3 worst single-attribute DPD for SO (age, XGBoost, S1): ~0.567
SO_SINGLE_ATTR_DPD = 0.5669907974069541
so_compounding = so_max_dpd - SO_SINGLE_ATTR_DPD
print(f"\nIntersectional compounding gap vs NB3 age-only DPD:")
print(f"  {so_max_dpd:.4f} − {SO_SINGLE_ATTR_DPD:.4f} = {so_compounding:+.4f}")
if so_compounding > 0:
    print("  ⚠  Intersectionality AMPLIFIES disadvantage beyond age alone.")
else:
    print("  ✓  No additional compounding detected.")


# =============================================================================
# GITHUB OSS  — intersectional analysis
# =============================================================================
print("\n" + "=" * 70)
print("GITHUB OSS — Intersectional Analysis")
print("=" * 70)

gh = pd.read_csv(IN_GH)

if "is_female" not in gh.columns or "age_group" not in gh.columns:
    raise ValueError(
        "Expected columns 'is_female' and 'age_group' in gh_preprocessed.csv."
    )

X_gh = gh[GH_FEATURE_COLS].fillna(0)
y_gh = gh["oss_contributor"].values

X_gh_tr, X_gh_te, y_gh_tr, y_gh_te, idx_gh_tr, idx_gh_te = train_test_split(
    X_gh, y_gh, gh.index, test_size=0.2, random_state=42
)
gh_test = gh.loc[idx_gh_te].reset_index(drop=True)

gh_xgb = load_model(os.path.join(MDL_DIR, "gh_xgboost.pkl"))
y_gh_pred = gh_xgb.predict(X_gh_te.reset_index(drop=True))

gh_subgroups = intersectional_subgroups(
    gh_test, age_col="age_group", gender_col="is_female"
)
gh_pos_rates = subgroup_positive_rates(
    pd.Series(y_gh_pred), gh_subgroups
)
gh_max_dpd, gh_worst_pair = intersectional_dpd(gh_pos_rates)

print("\nPositive-prediction rates by intersectional subgroup (GH):")
for sg, rate in sorted(gh_pos_rates.items()):
    flag = " ← lowest" if rate == min(gh_pos_rates.values()) else \
           " ← highest" if rate == max(gh_pos_rates.values()) else ""
    print(f"  {sg:25s}: {rate:.4f}{flag}")

print(f"\nMax intersectional DPD : {gh_max_dpd:.4f}")
print(f"Worst subgroup pair     : {gh_worst_pair[0]}  vs  {gh_worst_pair[1]}")
print(f"Meets threshold (≤{DPD_THRESHOLD}): {gh_max_dpd <= DPD_THRESHOLD}")

GH_SINGLE_ATTR_DPD = 0.024035369774919615
gh_compounding = gh_max_dpd - GH_SINGLE_ATTR_DPD
print(f"\nIntersectional compounding gap vs NB3 gender-only DPD:")
print(f"  {gh_max_dpd:.4f} − {GH_SINGLE_ATTR_DPD:.4f} = {gh_compounding:+.4f}")
if gh_compounding > 0:
    print("  ⚠  Intersectionality AMPLIFIES disadvantage beyond gender alone.")
else:
    print("  ✓  No additional compounding detected.")


# =============================================================================
# SAVE CSV RESULTS
# =============================================================================

def save_results(pos_rates, max_dpd, worst_pair, compounding,
                 single_attr_dpd, path):
    rows = []
    for sg, rate in sorted(pos_rates.items()):
        rows.append({
            "subgroup":            sg,
            "positive_pred_rate":  round(rate, 6),
        })
    df_out = pd.DataFrame(rows)
    df_out["intersectional_max_dpd"]   = round(max_dpd, 6)
    df_out["worst_pair"]               = f"{worst_pair[0]} vs {worst_pair[1]}"
    df_out["single_attr_dpd_ref"]      = round(single_attr_dpd, 6)
    df_out["compounding_gap"]          = round(compounding, 6)
    df_out["meets_threshold"]          = max_dpd <= DPD_THRESHOLD
    df_out.to_csv(path, index=False)
    print(f"\nSaved: {path}")

save_results(so_pos_rates, so_max_dpd, so_worst_pair,
             so_compounding, SO_SINGLE_ATTR_DPD,
             os.path.join(OUT, "so_intersectional_results.csv"))

save_results(gh_pos_rates, gh_max_dpd, gh_worst_pair,
             gh_compounding, GH_SINGLE_ATTR_DPD,
             os.path.join(OUT, "gh_intersectional_results.csv"))


# =============================================================================
# PLOT 1 — Grouped bar chart: positive-prediction rates per subgroup
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])
fig.suptitle(
    "Intersectional Bias: Positive-Prediction Rates by Age × Gender Subgroup",
    fontsize=13, fontweight="bold", y=1.02
)

SUBGROUP_COLORS = {
    "young_man":        PALETTE["young_m"],
    "young_woman":      PALETTE["young_f"],
    "experienced_man":  PALETTE["exp_m"],
    "experienced_woman":PALETTE["exp_f"],
}
SUBGROUP_LABELS = {
    "young_man":        "Young\nMan",
    "young_woman":      "Young\nWoman",
    "experienced_man":  "Experienced\nMan",
    "experienced_woman":"Experienced\nWoman",
}

for ax, (pos_rates, max_dpd, title) in zip(
    axes,
    [
        (so_pos_rates, so_max_dpd, "Stack Overflow — High Salary"),
        (gh_pos_rates, gh_max_dpd, "GitHub OSS — OSS Contributor"),
    ],
):
    sgs   = [sg for sg in SUBGROUP_COLORS if sg in pos_rates]
    rates = [pos_rates[sg] for sg in sgs]
    colors = [SUBGROUP_COLORS[sg] for sg in sgs]
    labels = [SUBGROUP_LABELS[sg] for sg in sgs]

    bars = ax.bar(labels, rates, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.55)

    # Annotate bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Threshold line (for reference)
    ax.axhline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle="--",
               linewidth=1, label=f"DPD threshold ({DPD_THRESHOLD})")

    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Positive-Prediction Rate")
    ax.set_ylim(0, min(1.0, max(rates) * 1.25))
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    dpd_color = PALETTE["ok"] if max_dpd <= DPD_THRESHOLD else PALETTE["bad"]
    ax.set_xlabel(
        f"Intersectional DPD = {max_dpd:.4f}  "
        f"({'✓ meets' if max_dpd <= DPD_THRESHOLD else '✗ exceeds'} threshold)",
        fontsize=9, color=dpd_color, fontweight="bold"
    )

plt.tight_layout()
chart_path = os.path.join(OUT, "nb6_intersectional_dpd.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()
print(f"Saved: {chart_path}")


# =============================================================================
# PLOT 2 — Heatmap of rates: rows = dataset, cols = subgroup
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 3), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

all_sgs = [sg for sg in SUBGROUP_COLORS
           if sg in so_pos_rates or sg in gh_pos_rates]

heatmap_data = pd.DataFrame(
    {
        "Stack Overflow": {sg: so_pos_rates.get(sg, np.nan) for sg in all_sgs},
        "GitHub OSS":     {sg: gh_pos_rates.get(sg, np.nan) for sg in all_sgs},
    }
).T

sns.heatmap(
    heatmap_data,
    annot=True, fmt=".3f",
    cmap="RdYlGn",
    vmin=0, vmax=1,
    linewidths=0.5, linecolor="white",
    ax=ax,
    cbar_kws={"label": "Positive-Prediction Rate"},
)
ax.set_title(
    "Positive-Prediction Rate Heatmap — Intersectional Subgroups",
    fontsize=11, fontweight="bold"
)
ax.set_xticklabels(
    [SUBGROUP_LABELS[sg].replace("\n", " ") for sg in heatmap_data.columns],
    fontsize=9
)
ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=9)

plt.tight_layout()
heatmap_path = os.path.join(OUT, "nb6_intersectional_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()
print(f"Saved: {heatmap_path}")

print("\n✓  Notebook 6 complete.")
