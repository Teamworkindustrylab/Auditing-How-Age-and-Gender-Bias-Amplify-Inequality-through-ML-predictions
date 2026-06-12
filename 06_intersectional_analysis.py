"""
  NOTEBOOK 6 -- INTERSECTIONAL BIAS ANALYSIS

  Examines bias at the intersection of two protected attributes
  simultaneously, rather than auditing each dimension in isolation.

  Research question:
    Does belonging to multiple disadvantaged groups create compounded
    disadvantage beyond what single-attribute audits would reveal?

  Dataset-specific intersections
  ------------------------------
  SO  : age_group x experience_bracket  (age_exp_pro, already in NB2)
        Groups: young_junior / young_mid / young_senior /
                experienced_junior / experienced_mid / experienced_senior
        Target : above_median_salary

  FCC : gender_clean x experience_bracket  (derived here from months_programming)
        Groups: man_junior / man_mid / man_senior /
                woman_junior / woman_mid / woman_senior
        Target : paid_contributor

  Method:
    Intersectional fairness (Foulds et al. 2020) computes the maximum
    pairwise Demographic Parity Difference (DPD) across all subgroup
    combinations. An "intersectional gap" is defined as the difference
    between this max and the worst single-attribute DPD from NB3 --
    a positive value indicates compounding harm.

  Compliance threshold : |DPD| <= 0.10  (consistent with NB3/NB5)

  Inputs  : data/preprocessed/so_preprocessed.csv
            data/preprocessed/fcc_preprocessed.csv
            outputs/models/so_xgboost.pkl
            outputs/models/fcc_xgboost.pkl
            outputs/so_bias_results.csv     (for reference DPDs)
            outputs/fcc_bias_results.csv

  Outputs : outputs/nb6_so_intersectional_dpd.png
            outputs/nb6_fcc_intersectional_dpd.png
            outputs/nb6_intersectional_heatmap.png
            outputs/so_intersectional_results.csv
            outputs/fcc_intersectional_results.csv
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

IN_SO   = "data/preprocessed/so_preprocessed.csv"
IN_FCC  = "data/preprocessed/fcc_preprocessed.csv"
OUT     = "outputs"
MDL_DIR = "outputs/models"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "so":  "#f48024",
    "fcc": "#0a0a23",
    "ok":  "#27ae60",
    "bad": "#e74c3c",
    "bg":  "#fafafa",
}

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

DPD_THRESHOLD = 0.10


# Reference single-attribute DPDs are READ from NB3's results files
# instead of hard-coded -- previously this was a brittle constant
# that drifted out of sync whenever NB3 was re-run.

def _ref_so_dpd() -> float:
    p = os.path.join(OUT, "so_bias_results.csv")
    if not os.path.exists(p):
        print("  [warn] so_bias_results.csv not found, using fallback 0.0")
        return 0.0
    df = pd.read_csv(p)
    row = df[(df["sensitive_def"] == "S1_age_group") & (df["model"] == "XGBoost")]
    if len(row) == 0:
        return 0.0
    return float(abs(row["dpd"].iloc[0]))


def _ref_fcc_dpd() -> float:
    p = os.path.join(OUT, "fcc_bias_results.csv")
    if not os.path.exists(p):
        print("  [warn] fcc_bias_results.csv not found, using fallback 0.0")
        return 0.0
    df = pd.read_csv(p)
    row = df[df["model"] == "XGBoost"]
    if len(row) == 0:
        return 0.0
    return float(abs(row["dpd"].iloc[0]))


# HELPERS

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def subgroup_positive_rates(y_pred, subgroups):
    result = {}
    for sg in sorted(subgroups.unique()):
        if sg == "unknown":
            continue
        mask = subgroups == sg
        if mask.sum() < 10:
            continue
        result[sg] = float(pd.Series(y_pred)[mask.values].mean())
    return result


def intersectional_dpd(pos_rates):
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
    ax.set_xlim(0, min(1.0, max(rates) * 1.35) if rates else 1.0)

    dpd_color = PALETTE["ok"] if max_dpd <= DPD_THRESHOLD else PALETTE["bad"]
    verdict = "meets" if max_dpd <= DPD_THRESHOLD else "exceeds"
    ax.set_xlabel(
        f"Positive-Prediction Rate\n"
        f"Intersectional DPD = {max_dpd:.4f}  ({verdict} threshold)",
        fontsize=9, color=dpd_color
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {out_path}")


# STACK OVERFLOW  --  age_group x experience_bracket

def so_intersectional():
    if not os.path.exists(IN_SO):
        print(f"[SO] {IN_SO} missing -- skipping.")
        return None

    print("\n" + "=" * 70)
    print("STACK OVERFLOW -- Intersectional Analysis: Age x Experience")
    print("=" * 70)

    so = pd.read_csv(IN_SO)
    so_features = SO_BASE_FEATURES + [
        c for c in so.columns if c.startswith("devtype_")
    ]
    so_features = [c for c in so_features if c in so.columns]

    X_so = so[so_features].fillna(0)
    y_so = so["above_median_salary"].values

    _, X_so_te, _, y_so_te, _, idx_te = train_test_split(
        X_so, y_so, so.index, test_size=0.2, random_state=42
    )
    so_test = so.loc[idx_te].reset_index(drop=True)

    so_xgb  = load_model(os.path.join(MDL_DIR, "so_xgboost.pkl"))
    y_so_pred = so_xgb.predict(X_so_te.reset_index(drop=True))

    so_subgroups = so_test["age_exp_pro"].fillna("unknown")
    so_pos_rates = subgroup_positive_rates(y_so_pred, so_subgroups)
    so_max_dpd, so_worst_pair = intersectional_dpd(so_pos_rates)
    ref_dpd = _ref_so_dpd()
    so_compounding = so_max_dpd - ref_dpd

    print("\nPositive-prediction rates by intersectional subgroup (SO):")
    for sg, rate in sorted(so_pos_rates.items()):
        flag = " <- lowest" if rate == min(so_pos_rates.values()) else \
               " <- highest" if rate == max(so_pos_rates.values()) else ""
        print(f"  {sg:30s}: {rate:.4f}{flag}")

    print(f"\nMax intersectional DPD:    {so_max_dpd:.4f}")
    print(f"Worst pair:                {so_worst_pair[0]}  vs  {so_worst_pair[1]}")
    print(f"Meets threshold (<= {DPD_THRESHOLD}):   {so_max_dpd <= DPD_THRESHOLD}")
    print(f"Compounding gap vs age-only DPD ({ref_dpd:.4f}): {so_compounding:+.4f}")

    save_results(so_pos_rates, so_max_dpd, so_worst_pair,
                 so_compounding, ref_dpd,
                 os.path.join(OUT, "so_intersectional_results.csv"))

    bar_chart(
        so_pos_rates, so_max_dpd,
        "SO 2024 -- Salary Prediction Rate\nby Age x Experience Subgroup",
        PALETTE["so"],
        os.path.join(OUT, "nb6_so_intersectional_dpd.png"),
    )
    return so_pos_rates


# FCC --  gender x experience_bracket

def fcc_intersectional():
    print("\n" + "=" * 70)
    print("FCC 2018 -- Intersectional Analysis: Gender x Experience")
    print("  Deriving experience bracket from months_programming")
    print("=" * 70)

    fcc = pd.read_csv(IN_FCC)

    def exp_bracket(months):
        # months -> junior (<24), mid (24-72), senior (>=72)
        if months < 24:  return "junior"
        if months < 72:  return "mid"
        return "senior"

    fcc["exp_bracket"] = fcc["months_programming"].apply(exp_bracket)
    fcc["subgroup"]    = fcc["gender_clean"] + "_" + fcc["exp_bracket"]

    X_fcc = fcc[[c for c in FCC_FEATURE_COLS if c in fcc.columns]].fillna(0)
    y_fcc = fcc["paid_contributor"].values

    _, X_fcc_te, _, y_fcc_te, _, idx_fcc_te = train_test_split(
        X_fcc, y_fcc, fcc.index, test_size=0.2, random_state=42
    )
    fcc_test = fcc.loc[idx_fcc_te].reset_index(drop=True)

    fcc_xgb    = load_model(os.path.join(MDL_DIR, "fcc_xgboost.pkl"))
    y_fcc_pred = fcc_xgb.predict(X_fcc_te.reset_index(drop=True))

    fcc_subgroups = fcc_test["subgroup"].fillna("unknown")
    fcc_pos_rates = subgroup_positive_rates(y_fcc_pred, fcc_subgroups)
    fcc_max_dpd, fcc_worst_pair = intersectional_dpd(fcc_pos_rates)
    ref_dpd = _ref_fcc_dpd()
    fcc_compounding = fcc_max_dpd - ref_dpd

    print("\nPositive-prediction rates by intersectional subgroup (FCC):")
    for sg, rate in sorted(fcc_pos_rates.items()):
        flag = " <- lowest" if rate == min(fcc_pos_rates.values()) else \
               " <- highest" if rate == max(fcc_pos_rates.values()) else ""
        print(f"  {sg:25s}: {rate:.4f}{flag}")

    print(f"\nMax intersectional DPD:    {fcc_max_dpd:.4f}")
    print(f"Worst pair:                {fcc_worst_pair[0]}  vs  {fcc_worst_pair[1]}")
    print(f"Meets threshold (<= {DPD_THRESHOLD}):   {fcc_max_dpd <= DPD_THRESHOLD}")
    print(f"Compounding gap vs gender-only DPD ({ref_dpd:.4f}): {fcc_compounding:+.4f}")
    if fcc_compounding > 0:
        print("  Intersectionality AMPLIFIES disadvantage beyond gender alone.")
    else:
        print("  No additional compounding detected.")

    save_results(fcc_pos_rates, fcc_max_dpd, fcc_worst_pair,
                 fcc_compounding, ref_dpd,
                 os.path.join(OUT, "fcc_intersectional_results.csv"))

    bar_chart(
        fcc_pos_rates, fcc_max_dpd,
        "FCC 2018 -- Working-as-Developer Rate\nby Gender x Experience Subgroup",
        PALETTE["fcc"],
        os.path.join(OUT, "nb6_fcc_intersectional_dpd.png"),
    )
    return fcc_pos_rates


# COMBINED HEATMAP

def combined_heatmap(so_pos_rates, fcc_pos_rates):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Intersectional Positive-Prediction Rates\n"
        "Left: SO (Age x Experience) | Right: FCC (Gender x Experience)",
        fontsize=11, fontweight="bold"
    )

    for ax, (pos_rates, title, cmap) in zip(
        axes,
        [
            (so_pos_rates,  "Stack Overflow\n(above-median salary)", "YlOrRd"),
            (fcc_pos_rates, "FCC 2018\n(working as developer)",      "YlGnBu"),
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
            vmin=0, vmax=max(0.5, df_h.values.max() * 1.2),
            linewidths=0.5, linecolor="white",
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


# MAIN

if __name__ == "__main__":
    so_rates = so_intersectional()
    fcc_rates = fcc_intersectional()

    if so_rates and fcc_rates:
        combined_heatmap(so_rates, fcc_rates)

    print("\n  Notebook 6 complete.")
