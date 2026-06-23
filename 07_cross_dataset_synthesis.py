"""

  NOTEBOOK 7 -- CROSS-DATASET SYNTHESIS

  Inputs  : outputs/so_bias_results.csv          (NB3)
            outputs/fcc_bias_results.csv         (NB3)
            outputs/adult_bias_results.csv       (NB3)
            outputs/so_mitigation_results.csv    (NB5)
            outputs/fcc_mitigation_results.csv   (NB5)
            outputs/adult_mitigation_results.csv (NB5)
            outputs/so_intersectional_results.csv     (NB6)
            outputs/fcc_intersectional_results.csv    (NB6)
            outputs/adult_intersectional_results.csv  (NB6)

  Outputs : outputs/final_cross_dataset_comparison.csv
            outputs/nb7_amplification_comparison.png
            outputs/nb7_gap_comparison.png
            outputs/nb7_compliance_dashboard.png
            outputs/nb7_summary.md  (text summary used by Presentation.ipynb)

  Purpose
  -------
  Per-dataset notebooks (NB3-NB6) each produce their own CSVs and PNGs.
  This notebook does NOT re-train anything; it READS those CSVs and
  assembles a single comparative view across all available datasets.

  README.md and Presentation.ipynb both reference this notebook's
  outputs, but the script itself was missing in the original tree -- so
  the cross-dataset comparison in the writeup had no generating code.
  This file fills that gap.

  Design choice: gracefully handles any subset of the three datasets.
  If you only have FCC results (e.g. you can't run SO without the Kaggle
  download, or Adult is unreachable from your network), NB7 will still
  produce a 1-dataset comparison rather than crashing. The pipeline is
  resilient because the prof's "did we cross 10% DPD" question can be
  answered as soon as ANY high-bias dataset (Adult) has been run, even
  if the other two haven't yet.
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "so":    "#f48024",
    "fcc":   "#0a0a23",
    "adult": "#2e7d32",
    "ok":    "#27ae60",
    "bad":   "#e74c3c",
    "warn":  "#f39c12",
    "bg":    "#fafafa",
}

DPD_THRESHOLD = 0.10
PRIMARY_MODEL = "XGBoost"  # which model row to use as "the" headline result

DATASETS = [
    {
        "key":           "so",
        "label":         "Stack Overflow 2024",
        "n":             65_000,
        "sensitive":     "Age (binary: <35 vs >=35)",
        "target":        "above_median_salary",
        "color":         PALETTE["so"],
        "bias_csv":      "so_bias_results.csv",
        "mit_csv":       "so_mitigation_results.csv",
        "isec_csv":      "so_intersectional_results.csv",
        # SO has 3 sensitive defs; we report the binary one (S1_age_group)
        # as the headline so the table compares like-with-like (each
        # dataset reports a single binary sensitive attribute).
        "bias_filter":   {"sensitive_def": "S1_age_group"},
        "mit_filter":    {"sensitive_def": "S1_age_group"},
    },
    {
        "key":           "fcc",
        "label":         "freeCodeCamp 2018",
        "n":             30_047,
        "sensitive":     "Gender (binary: man vs woman)",
        "target":        "paid_contributor",
        "color":         PALETTE["fcc"],
        "bias_csv":      "fcc_bias_results.csv",
        "mit_csv":       "fcc_mitigation_results.csv",
        "isec_csv":      "fcc_intersectional_results.csv",
        "bias_filter":   {},
        "mit_filter":    {},
    },
    {
        "key":           "adult",
        "label":         "UCI Adult / Census Income",
        "n":             48_842,
        "sensitive":     "Gender (binary: man vs woman)",
        "target":        "above_50k",
        "color":         PALETTE["adult"],
        "bias_csv":      "adult_bias_results.csv",
        "mit_csv":       "adult_mitigation_results.csv",
        "isec_csv":      "adult_intersectional_results.csv",
        "bias_filter":   {},
        "mit_filter":    {},
    },
]


# READERS

def _read_csv_safe(path: str) -> pd.DataFrame | None:
    """Return None instead of raising if a per-dataset CSV is missing -- so
    NB7 still runs when only a subset of the three datasets has been
    executed."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}")
        return None


def _apply_filter(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    out = df
    for col, val in flt.items():
        if col in out.columns:
            out = out[out[col] == val]
    return out


def collect_dataset_row(ds: dict) -> dict | None:
    """Pull bias / mitigation / intersectional results for one dataset
    and flatten into a single comparison row. Returns None if no results
    files for this dataset exist at all (so it just gets skipped)."""
    bias_df = _read_csv_safe(os.path.join(OUT, ds["bias_csv"]))
    mit_df  = _read_csv_safe(os.path.join(OUT, ds["mit_csv"]))
    isec_df = _read_csv_safe(os.path.join(OUT, ds["isec_csv"]))

    if bias_df is None and mit_df is None and isec_df is None:
        return None

    row = {
        "dataset":             ds["label"],
        "key":                 ds["key"],
        "n_samples":           ds["n"],
        "sensitive_attribute": ds["sensitive"],
        "target":              ds["target"],
    }

    # Bias amplification
    if bias_df is not None:
        bf = _apply_filter(bias_df, ds["bias_filter"])
        bf_xgb = bf[bf["model"] == PRIMARY_MODEL]
        bf_lr  = bf[bf["model"] == "Logistic Regression"]

        if "data_gap" in bf.columns and len(bf) > 0:
            row["data_gap"] = float(abs(bf["data_gap"].iloc[0]))
        if len(bf_xgb) > 0:
            row["xgb_dpd"] = float(abs(bf_xgb["dpd"].iloc[0]))
            row["xgb_amplification"] = float(bf_xgb["amplification"].iloc[0])
            row["xgb_auc"] = float(bf_xgb["auc"].iloc[0])
        if len(bf_lr) > 0:
            row["lr_dpd"] = float(abs(bf_lr["dpd"].iloc[0]))
            row["lr_amplification"] = float(bf_lr["amplification"].iloc[0])
            row["lr_auc"] = float(bf_lr["auc"].iloc[0])

    # Mitigation -- best DPD across baseline/reweigh/threshold
    if mit_df is not None:
        mf = _apply_filter(mit_df, ds["mit_filter"])
        if len(mf) > 0:
            row["baseline_dpd"]  = float(abs(mf[mf["strategy"] == "baseline"]["dpd"].iloc[0])) \
                if "baseline" in mf["strategy"].values else np.nan
            row["reweigh_dpd"]   = float(abs(mf[mf["strategy"] == "reweigh"]["dpd"].iloc[0])) \
                if "reweigh" in mf["strategy"].values else np.nan
            row["threshold_dpd"] = float(abs(mf[mf["strategy"] == "threshold"]["dpd"].iloc[0])) \
                if "threshold" in mf["strategy"].values else np.nan
            valid = [v for v in [row.get("reweigh_dpd"), row.get("threshold_dpd")]
                     if v is not None and not np.isnan(v)]
            row["best_mitigated_dpd"] = float(min(valid)) if valid else np.nan

    # Intersectional
    if isec_df is not None and "intersectional_max_dpd" in isec_df.columns:
        row["intersectional_max_dpd"] = float(isec_df["intersectional_max_dpd"].iloc[0])
        if "single_attr_dpd_ref" in isec_df.columns:
            row["single_attr_dpd_ref"] = float(isec_df["single_attr_dpd_ref"].iloc[0])
        if "compounding_gap" in isec_df.columns:
            row["compounding_gap"] = float(isec_df["compounding_gap"].iloc[0])
        if "worst_pair" in isec_df.columns:
            row["intersectional_worst_pair"] = str(isec_df["worst_pair"].iloc[0])

    # Compliance flags
    row["meets_threshold_baseline"] = (
        row.get("xgb_dpd", np.inf) <= DPD_THRESHOLD
    )
    row["meets_threshold_mitigated"] = (
        row.get("best_mitigated_dpd", np.inf) <= DPD_THRESHOLD
    )
    row["meets_threshold_intersectional"] = (
        row.get("intersectional_max_dpd", np.inf) <= DPD_THRESHOLD
    )
    return row


# PLOTS

def plot_amplification_comparison(rows: list[dict]):
    """Bar chart: XGBoost amplification ratio per dataset, plus the
    1.0 'no amplification' line. Datasets where the model amplifies
    (>1) are above the line; datasets where it compresses (<1) are below."""
    plot_rows = [r for r in rows if "xgb_amplification" in r]
    if not plot_rows:
        print("  [skip] no rows with amplification ratios.")
        return

    labels = [r["dataset"] for r in plot_rows]
    xgb    = [r["xgb_amplification"] for r in plot_rows]
    lr     = [r.get("lr_amplification", np.nan) for r in plot_rows]
    colors = [next(d["color"] for d in DATASETS if d["key"] == r["key"])
              for r in plot_rows]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    x = np.arange(len(labels))
    w = 0.38

    bars_xgb = ax.bar(x - w/2, xgb, w, label="XGBoost",
                      color=colors, alpha=0.9, edgecolor="white")
    bars_lr  = ax.bar(x + w/2, lr, w, label="Logistic Regression",
                      color=colors, alpha=0.5, edgecolor="white",
                      hatch="///")

    for bar, v in zip(bars_xgb, xgb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{v:.2f}x", ha="center", fontsize=9, fontweight="bold")
    for bar, v in zip(bars_lr, lr):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{v:.2f}x", ha="center", fontsize=8)

    ax.axhline(1.0, color="black", linestyle="--", lw=1.5,
               label="No amplification (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Amplification ratio = |DPD_model| / |DPD_data|", fontsize=10)
    ax.set_title(
        "Cross-Dataset Bias Amplification\n"
        "Above 1.0 = model widens the gap; below 1.0 = model compresses it",
        fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(2.5, max(xgb + [v for v in lr if not np.isnan(v)]) * 1.25))
    plt.tight_layout()
    out = f"{OUT}/nb7_amplification_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_gap_comparison(rows: list[dict]):
    """Side-by-side bars: data gap vs model DPD (XGBoost) for each
    dataset. Visualises whether the model is preserving, amplifying, or
    compressing the raw disparity."""
    plot_rows = [r for r in rows if "data_gap" in r and "xgb_dpd" in r]
    if not plot_rows:
        print("  [skip] no rows with both data_gap and xgb_dpd.")
        return

    labels   = [r["dataset"] for r in plot_rows]
    data_gap = [r["data_gap"] for r in plot_rows]
    model_gap= [r["xgb_dpd"] for r in plot_rows]
    colors   = [next(d["color"] for d in DATASETS if d["key"] == r["key"])
                for r in plot_rows]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    x = np.arange(len(labels))
    w = 0.38

    bars_d = ax.bar(x - w/2, data_gap, w, label="Data DPD (raw)",
                    color=colors, alpha=0.45, edgecolor="white")
    bars_m = ax.bar(x + w/2, model_gap, w, label="Model DPD (XGBoost)",
                    color=colors, alpha=0.95, edgecolor="white")

    for bar, v in zip(bars_d, data_gap):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=8)
    for bar, v in zip(bars_m, model_gap):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle="--", lw=1.5,
               label=f"Compliance threshold |DPD| <= {DPD_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("|DPD| (lower is fairer)", fontsize=10)
    ax.set_title(
        "Data Bias vs Model Bias Across Datasets\n"
        "How much of the raw disparity does the model preserve?",
        fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    max_y = max(max(data_gap), max(model_gap)) * 1.25
    ax.set_ylim(0, max(0.2, max_y))
    plt.tight_layout()
    out = f"{OUT}/nb7_gap_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_compliance_dashboard(rows: list[dict]):
    """Three-row dashboard per dataset:
       (1) baseline DPD vs threshold
       (2) best mitigated DPD vs threshold
       (3) intersectional max DPD vs threshold
    Green if under, red if over -- gives the prof a one-glance view of
    whether the project's central claim (single-axis compliance can mask
    intersectional non-compliance) holds across all three datasets."""
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: r["key"])
    n_ds = len(rows_sorted)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 1.2 * n_ds + 2)),
                           facecolor=PALETTE["bg"])

    bands = [
        ("Baseline\n(XGBoost, no mitigation)",       "xgb_dpd"),
        ("Best mitigation\n(reweigh or threshold)",  "best_mitigated_dpd"),
        ("Intersectional\n(gender/age x experience)","intersectional_max_dpd"),
    ]
    y_labels = []
    y_positions = []
    bar_colors  = []
    bar_widths  = []
    text_labels = []

    for i, row in enumerate(rows_sorted):
        for j, (band_name, key) in enumerate(bands):
            y = i * len(bands) + j
            v = row.get(key, np.nan)
            y_positions.append(y)
            y_labels.append(f"{row['dataset']}\n{band_name}" if j == 0
                            else f"\n{band_name}")
            if np.isnan(v):
                bar_colors.append("#cccccc")
                bar_widths.append(0)
                text_labels.append("(no result)")
            else:
                bar_widths.append(v)
                if v <= DPD_THRESHOLD:
                    bar_colors.append(PALETTE["ok"])
                elif v <= DPD_THRESHOLD * 2:
                    bar_colors.append(PALETTE["warn"])
                else:
                    bar_colors.append(PALETTE["bad"])
                text_labels.append(f"{v:.3f}")

    ax.barh(y_positions, bar_widths, color=bar_colors, alpha=0.9,
            edgecolor="white")
    for y, v, txt in zip(y_positions, bar_widths, text_labels):
        ax.text(v + 0.01, y, txt, va="center", fontsize=9,
                fontweight="bold")

    ax.axvline(DPD_THRESHOLD, color=PALETTE["bad"], linestyle="--", lw=1.5,
               label=f"|DPD| <= {DPD_THRESHOLD}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|DPD|", fontsize=10)
    ax.set_title(
        "Compliance Dashboard -- |DPD| <= 0.10\n"
        "Green = compliant, orange = within 2x, red = far over",
        fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, max(0.3, max([w for w in bar_widths if w], default=0.2) * 1.25))

    # Visual dividers between datasets
    for i in range(1, n_ds):
        ax.axhline(i * len(bands) - 0.5, color="#888888", lw=0.8, alpha=0.5)

    plt.tight_layout()
    out = f"{OUT}/nb7_compliance_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# WRITERS

def write_summary_md(rows: list[dict], path: str):
    """Generate a markdown summary that Presentation.ipynb can include
    directly without re-implementing the table-building logic."""
    if not rows:
        with open(path, "w") as f:
            f.write("# Cross-Dataset Synthesis\n\nNo dataset results "
                    "available yet. Run NB1-NB6 first.\n")
        return

    lines = []
    lines.append("# Cross-Dataset Synthesis -- generated by NB7\n")
    lines.append(f"_Compliance threshold: |DPD| <= {DPD_THRESHOLD}_\n")
    lines.append("\n## Headline comparison\n")
    lines.append("| Dataset | N | Sensitive | Data DPD | XGB DPD | "
                 "Amp. | Best mit. | Inter. max | Single-axis OK? | "
                 "Intersectional OK? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def _fmt(v, p=3):
            return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
                   else f"{v:.{p}f}"
        lines.append(
            f"| {r['dataset']} "
            f"| {r['n_samples']:,} "
            f"| {r['sensitive_attribute']} "
            f"| {_fmt(r.get('data_gap'))} "
            f"| {_fmt(r.get('xgb_dpd'))} "
            f"| {_fmt(r.get('xgb_amplification'), 2)}x "
            f"| {_fmt(r.get('best_mitigated_dpd'))} "
            f"| {_fmt(r.get('intersectional_max_dpd'))} "
            f"| {'YES' if r.get('meets_threshold_baseline') else 'NO'} "
            f"| {'YES' if r.get('meets_threshold_intersectional') else 'NO'} |"
        )

    lines.append("\n## Interpretation\n")
    # Sort datasets along the spectrum described in Presentation.ipynb §8
    by_gap = sorted([r for r in rows if "data_gap" in r],
                    key=lambda r: r["data_gap"])
    if by_gap:
        lines.append("**Baseline-bias spectrum (low -> high):**")
        for r in by_gap:
            lines.append(f"- {r['dataset']}: data DPD = {r['data_gap']:.3f}")
        lines.append("")

    amp_ratios = [(r["dataset"], r.get("xgb_amplification"))
                  for r in rows if "xgb_amplification" in r]
    amp_above_1 = [d for d, v in amp_ratios if v is not None and v > 1]
    amp_below_1 = [d for d, v in amp_ratios if v is not None and v <= 1]
    if amp_above_1:
        lines.append(f"**Amplifies the data gap:** "
                     f"{', '.join(amp_above_1)}")
    if amp_below_1:
        lines.append(f"**Compresses the data gap:** "
                     f"{', '.join(amp_below_1)}")
    lines.append("")

    isec_fail = [r["dataset"] for r in rows
                 if not r.get("meets_threshold_intersectional", True)]
    isec_ok   = [r["dataset"] for r in rows
                 if r.get("meets_threshold_intersectional", False)]
    if isec_fail:
        lines.append(
            f"**Intersectional non-compliance:** "
            f"{', '.join(isec_fail)} cross the |DPD| <= {DPD_THRESHOLD} line "
            f"once a second axis (experience) is added. "
            f"This is the project's headline result -- single-axis "
            f"compliance can hide order-of-magnitude subgroup disparity."
        )
    if isec_ok:
        lines.append(f"**Intersectional compliance:** {', '.join(isec_ok)}.")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved -> {path}")


# MAIN

def main() -> int:
    print("\n" + "=" * 65)
    print("  NOTEBOOK 7 -- CROSS-DATASET SYNTHESIS")
    print("=" * 65)

    rows = []
    for ds in DATASETS:
        row = collect_dataset_row(ds)
        if row is None:
            print(f"  [skip] no results found for {ds['label']} "
                  f"(none of {ds['bias_csv']}, {ds['mit_csv']}, "
                  f"{ds['isec_csv']} exists)")
            continue
        rows.append(row)
        print(f"  [ok]   collected results for {ds['label']}")

    if not rows:
        print("\n  No per-dataset results available. Run NB1-NB6 first.")
        return 1

    # Write the comparison CSV
    df = pd.DataFrame(rows)
    csv_path = f"{OUT}/final_cross_dataset_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved -> {csv_path}")
    print("\n" + df.to_string(index=False))

    print("\n  Generating charts ...")
    plot_amplification_comparison(rows)
    plot_gap_comparison(rows)
    plot_compliance_dashboard(rows)

    print("\n  Generating markdown summary ...")
    write_summary_md(rows, f"{OUT}/nb7_summary.md")

    print("\n  NOTEBOOK 7 COMPLETE")
    print(f"  Datasets included: {len(rows)} of {len(DATASETS)}")
    if len(rows) < len(DATASETS):
        missing = [d["label"] for d in DATASETS
                   if d["label"] not in [r["dataset"] for r in rows]]
        print(f"  Missing: {', '.join(missing)}")
        print("  Re-run NB1-NB6 for those datasets to fill in the gaps, "
              "then re-run NB7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
