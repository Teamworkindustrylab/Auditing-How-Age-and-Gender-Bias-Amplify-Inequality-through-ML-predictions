"""
pipeline.py -- thin orchestrator that runs the numbered notebook-scripts in order.

The previous monolithic pipeline.py duplicated logic from 01..06 and referenced
the deprecated GitHub Open Source Survey 2017 (now replaced by the freeCodeCamp
2018 New Coder Survey -- see Presentation.ipynb, section 1, for the rationale).

To avoid implementation drift between two copies of the same code, this file
no longer reimplements anything: it just executes the numbered scripts in
sequence. They are the single source of truth.

Usage
-----
    python pipeline.py            # run 01..06 in order
    python pipeline.py 1 2 3      # run only NB1, NB2, NB3
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("01_preprocessing.py",            "Notebook 1 -- ingestion & EDA"),
    ("02_feature_engineering.py",      "Notebook 2 -- feature engineering"),
    ("03_bias_amplification.py",       "Notebook 3 -- bias amplification"),
    ("04_shap_explainability.py",      "Notebook 4 -- SHAP explainability"),
    ("05_mitigation_compliance.py",    "Notebook 5 -- mitigation & compliance"),
    ("06_intersectional_analysis.py",  "Notebook 6 -- intersectional analysis"),
    ("07_cross_dataset_synthesis.py",  "Notebook 7 -- cross-dataset synthesis"),
    ("08_sensitivity_bootstrap.py",    "Notebook 8 -- sensitivity & bootstrap CIs"),
]


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        try:
            selected = {int(a) for a in argv[1:]}
        except ValueError:
            print(f"Usage: python {Path(__file__).name} [step_numbers ...]")
            return 2
        steps = [s for i, s in enumerate(STEPS, start=1) if i in selected]
    else:
        steps = STEPS

    for fname, label in steps:
        path = HERE / fname
        print(f"\n{'='*70}\n  {label}\n  $ python {fname}\n{'='*70}")
        rc = subprocess.call([sys.executable, str(path)], cwd=str(HERE))
        if rc != 0:
            print(f"  [pipeline] {fname} exited with code {rc} -- stopping.")
            return rc

    print("\n[pipeline] all steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
