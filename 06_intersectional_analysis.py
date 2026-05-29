"""
  NOTEBOOK 6 -- INTERSECTIONAL BIAS ANALYSIS

  Examines bias at the intersection of two protected attributes
  simultaneously, rather than auditing each dimension in isolation.

  Research question:
    Does belonging to multiple disadvantaged groups create compounded
    disadvantage beyond what single-attribute audits would reveal?

  Dataset-specific intersections
  SO  : age_group × experience_bracket  (age_exp_pro, already in NB2)
        Groups: young_junior / young_mid / young_senior /
                experienced_junior / experienced_mid / experienced_senior
        Target : above_median_salary

  GH  : gender × experience_bracket  (derived here from pro_experience_yrs)
        Groups: man_junior / man_mid / man_senior /
                woman_junior / woman_mid / woman_senior
        Target : paid_contributor (OSS participation)

  NOTE: SO has no gender column and GH has no age column. The datasets
  are NOT combined. Each is audited on the sensitive dimensions it
  actually contains. This design follows the principle that intersectional
  fairness must be grounded in the data as collected, not imputed.

  Method:
    Intersectional fairness (Foulds et al. 2020) computes the maximum
    pairwise Demographic Parity Difference (DPD) across all subgroup
    combinations. An "intersectional gap" is defined as the difference
    between this max and the worst single-attribute DPD from NB3 —
    a positive value indicates compounding harm.

  Compliance threshold : |DPD| <= 0.10  (consistent with NB3/NB5)

  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/gh_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/gh_xgboost.pkl

  Outputs : outputs/nb6_so_intersectional_dpd.png
            outputs/nb6_gh_intersectional_dpd.png
            outputs/nb6_intersectional_heatmap.png
            outputs/so_intersectional_results.csv
            outputs/gh_intersectional_results.csv

    
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

#  Paths 
IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_GH   = "data/preprocessed/gh_preprocessed.csv"
OUT     = "outputs"
MDL_DIR = "outputs/models"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "so":  "#f48024",
    "gh":  "#24292e",
    "ok":  "#27ae60",
    "bad": "#e74c3c",
    "bg":  "#fafafa",
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

# Reference single-attribute DPDs from NB3 (XGBoost baseline)
SO_SINGLE_ATTR_DPD = 0.5669907974069541   # S1 age_group
GH_SINGLE_ATTR_DPD = 0.024035369774919615  # gender


# HELPERS

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def subgroup_positive_rates(y_pred, subgroups):
    """Positive-prediction rate per subgroup (excludes 'unknown')."""
    result = {}
    for sg in sorted(subgroups.unique()):
        if sg == "unknown":
            continue
        mask = subgroups == sg
        if mask.sum() < 10:   # skip tiny cells
            continue
        result[sg] = float(pd.Series(y_pred)[mask.values].mean())
    return result


def intersectional_dpd(pos_rates):
    """Max pairwise DPD across all subgroup combinations."""
    groups = list(pos_rates.keys())
    max_gap, worst_pair = 0.0, (None, None)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gap = abs(pos_rates[groups[i]] - pos_rates[groups[j]])
            if gap > max_gap:
                max_gap = gap
                worst_pair = (groups[i], groups[j])
    return max_gap, worst_pair


def save_results(pos_rates, max_dpd, worst_pair, compounding,
                 single_attr_dpd, path):
    rows = [{"subgroup": sg,
             "positive_pred_rate": round(rate, 6)}
            for sg, rate in sorted(pos_rates.items())]
    df_out = pd.DataFrame(rows)
    df_out["intersectional_max_dpd"]  = round(max_dpd, 6)
    df_out["worst_pair"]              = f"{worst_pair[0]} vs {worst_pair[1]}"
    df_out["single_attr_dpd_ref"]     = round(single_attr_dpd, 6)
    df_out["compounding_gap"]         = round(compounding, 6)
    df_out["meets_threshold"]         = max_dpd <= DPD_THRESHOLD
    df_out.to_csv(path, index=False)
    print(f"  Saved: {path}")


def bar_chart(pos_rates, max_dpd, title, dataset_color, out_path):
    """Horizontal bar chart of positive-prediction rates per subgroup."""
    sgs   = sorted(pos_rates.keys())
    rates = [pos_rates[sg] for sg in sgs]

    fig, ax = plt.subplots(figsize=(9, max(4, len(sgs) * 0.7)),
                           facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    colors = [dataset_color if "young" in sg or "woman" in sg
              else "#888888" for sg in sgs]
    bars = ax.barh(sgs, rates, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8)

    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{rate:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.axvline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle="--",
               linewidth=1.2, label=f"DPD threshold ({DPD_THRESHOLD})")
    ax.set_xlabel("Positive-Prediction Rate")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, min(1.0, max(rates) * 1.35))

    dpd_color = PALETTE["ok"] if max_dpd <= DPD_THRESHOLD else PALETTE["bad"]
    verdict = "✓ meets" if max_dpd <= DPD_THRESHOLD else "✗ exceeds"
    ax.set_xlabel(
        f"Positive-Prediction Rate\n"
        f"Intersectional DPD = {max_dpd:.4f}  ({verdict} threshold)",
        fontsize=9, color=dpd_color
    )
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {out_path}")


# STACK OVERFLOW  — age_group × experience_bracket

print("\n" + "=" * 70)
print("STACK OVERFLOW — Intersectional Analysis: Age × Experience")
print("  Subgroup column: age_exp_pro  (already built in NB2)")
print("=" * 70)

so = pd.read_csv(IN_SO)

SO_FEATURES = SO_BASE_FEATURES + [
    c for c in so.columns if c.startswith("devtype_")
]
SO_FEATURES = [c for c in SO_FEATURES if c in so.columns]

X_so = so[SO_FEATURES].fillna(0)
y_so = so["above_median_salary"].values

# Same split as NB3 (random_state=42 → identical test partition)
_, X_so_te, _, y_so_te, _, idx_te = train_test_split(
    X_so, y_so, so.index, test_size=0.2, random_state=42
)
so_test = so.loc[idx_te].reset_index(drop=True)

so_xgb  = load_model(os.path.join(MDL_DIR, "so_xgboost.pkl"))
y_so_pred = so_xgb.predict(X_so_te.reset_index(drop=True))

# age_exp_pro is already the intersectional column (young/experienced × junior/mid/senior)
so_subgroups = so_test["age_exp_pro"].fillna("unknown")

so_pos_rates = subgroup_positive_rates(y_so_pred, so_subgroups)
so_max_dpd, so_worst_pair = intersectional_dpd(so_pos_rates)
so_compounding = so_max_dpd - SO_SINGLE_ATTR_DPD

print("\nPositive-prediction rates by intersectional subgroup (SO):")
for sg, rate in sorted(so_pos_rates.items()):
    flag = " ← lowest" if rate == min(so_pos_rates.values()) else \
           " ← highest" if rate == max(so_pos_rates.values()) else ""
    print(f"  {sg:30s}: {rate:.4f}{flag}")

print(f"\nMax intersectional DPD:    {so_max_dpd:.4f}")
print(f"Worst pair:                {so_worst_pair[0]}  vs  {so_worst_pair[1]}")
print(f"Meets threshold (≤{DPD_THRESHOLD}):   {so_max_dpd <= DPD_THRESHOLD}")
print(f"\nCompounding gap vs age-only DPD ({SO_SINGLE_ATTR_DPD:.4f}):")
print(f"  {so_max_dpd:.4f} − {SO_SINGLE_ATTR_DPD:.4f} = {so_compounding:+.4f}")
if so_compounding > 0:
    print("  ⚠  Intersectionality AMPLIFIES disadvantage beyond age alone.")
else:
    print("  ✓  No additional compounding: experience bracket absorbs age gap.")

save_results(so_pos_rates, so_max_dpd, so_worst_pair,
             so_compounding, SO_SINGLE_ATTR_DPD,
             os.path.join(OUT, "so_intersectional_results.csv"))

bar_chart(
    so_pos_rates, so_max_dpd,
    "SO 2024 — Salary Prediction Rate\nby Age × Experience Subgroup",
    PALETTE["so"],
    os.path.join(OUT, "nb6_so_intersectional_dpd.png"),
)



# GITHUB OSS  — gender × experience_bracket

print("\n" + "=" * 70)
print("GITHUB OSS — Intersectional Analysis: Gender × Experience")
print("  Deriving experience bracket from pro_experience_yrs")
print("=" * 70)

gh = pd.read_csv(IN_GH)


def exp_bracket(yrs):
    """Map numeric years to junior / mid / senior."""
    if yrs < 3:   return "junior"
    if yrs < 9:   return "mid"
    return "senior"


gh["exp_bracket"] = gh["pro_experience_yrs"].apply(exp_bracket)
gh["subgroup"]    = gh["gender_clean"] + "_" + gh["exp_bracket"]

# Check GH has a target column
TARGET_GH = "paid_contributor" if "paid_contributor" in gh.columns else "oss_contributor"
if TARGET_GH not in gh.columns:
    raise ValueError(f"Expected 'paid_contributor' in gh_preprocessed.csv. "
                     f"Available: {list(gh.columns)}")

X_gh = gh[GH_FEATURE_COLS].fillna(0)
y_gh = gh[TARGET_GH].values

_, X_gh_te, _, y_gh_te, _, idx_gh_te = train_test_split(
    X_gh, y_gh, gh.index, test_size=0.2, random_state=42
)
gh_test = gh.loc[idx_gh_te].reset_index(drop=True)

gh_xgb    = load_model(os.path.join(MDL_DIR, "gh_xgboost.pkl"))
y_gh_pred = gh_xgb.predict(X_gh_te.reset_index(drop=True))

gh_subgroups = gh_test["subgroup"].fillna("unknown")

gh_pos_rates = subgroup_positive_rates(y_gh_pred, gh_subgroups)
gh_max_dpd, gh_worst_pair = intersectional_dpd(gh_pos_rates)
gh_compounding = gh_max_dpd - GH_SINGLE_ATTR_DPD

print("\nPositive-prediction rates by intersectional subgroup (GH):")
for sg, rate in sorted(gh_pos_rates.items()):
    flag = " ← lowest" if rate == min(gh_pos_rates.values()) else \
           " ← highest" if rate == max(gh_pos_rates.values()) else ""
    print(f"  {sg:25s}: {rate:.4f}{flag}")

print(f"\nMax intersectional DPD:    {gh_max_dpd:.4f}")
print(f"Worst pair:                {gh_worst_pair[0]}  vs  {gh_worst_pair[1]}")
print(f"Meets threshold (≤{DPD_THRESHOLD}):   {gh_max_dpd <= DPD_THRESHOLD}")
print(f"\nCompounding gap vs gender-only DPD ({GH_SINGLE_ATTR_DPD:.4f}):")
print(f"  {gh_max_dpd:.4f} − {GH_SINGLE_ATTR_DPD:.4f} = {gh_compounding:+.4f}")
if gh_compounding > 0:
    print("  ⚠  Intersectionality AMPLIFIES disadvantage beyond gender alone.")
else:
    print("  ✓  No additional compounding detected.")

save_results(gh_pos_rates, gh_max_dpd, gh_worst_pair,
             gh_compounding, GH_SINGLE_ATTR_DPD,
             os.path.join(OUT, "gh_intersectional_results.csv"))

bar_chart(
    gh_pos_rates, gh_max_dpd,
    "GH OSS 2017 — OSS Participation Rate\nby Gender × Experience Subgroup",
    PALETTE["gh"],
    os.path.join(OUT, "nb6_gh_intersectional_dpd.png"),
)


# COMBINED HEATMAP  — side-by-side summary


fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=PALETTE["bg"])
fig.suptitle(
    "Intersectional Positive-Prediction Rates\n"
    "Left: SO (Age × Experience) | Right: GH OSS (Gender × Experience)",
    fontsize=11, fontweight="bold"
)

for ax, (pos_rates, title, cmap) in zip(
    axes,
    [
        (so_pos_rates, "Stack Overflow\n(above-median salary)", "YlOrRd"),
        (gh_pos_rates, "GitHub OSS\n(OSS participation)",      "YlGnBu"),
    ]
):
    if not pos_rates:
        ax.axis("off")
        continue

    sgs   = sorted(pos_rates.keys())
    rates = [[pos_rates[sg]] for sg in sgs]
    df_h  = pd.DataFrame(rates, index=sgs, columns=["rate"])

    sns.heatmap(
        df_h, annot=True, fmt=".3f", cmap=cmap,
        vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Positive-Prediction Rate"}
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout()
heatmap_path = os.path.join(OUT, "nb6_intersectional_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()
print(f"  Saved: {heatmap_path}")

print("\n✓  Notebook 6 complete.")
