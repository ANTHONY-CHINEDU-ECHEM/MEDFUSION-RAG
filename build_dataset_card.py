"""Generate docs/dataset_card.md with schema and descriptive statistics.

Tables are emitted as HTML so that no Markdown table separators are needed.
Usage: python tools/build_dataset_card.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medfusion.data.generator import COLUMN_SPEC  # noqa: E402


def html_table(headers, rows):
    out = ["<table>", "  <tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr>"]
    for row in rows:
        out.append("  <tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def distribution(frame, column, top=12):
    counts = frame[column].astype(str).value_counts().head(top)
    rows = [(k, f"{v:,}", f"{100 * v / len(frame):.1f}%") for k, v in counts.items()]
    return html_table([column, "Studies", "Share"], rows)


def dictionary_table(frame):
    rows = []
    for column, (dtype, description) in COLUMN_SPEC.items():
        missing = frame[column].isna().mean() * 100
        rows.append((column, dtype, description, f"{missing:.1f}%"))
    return html_table(["Column", "Type", "Description", "Missing"], rows)


def numeric_summary(frame):
    cols = ["age_years", "pack_years", "bmi", "nodule_diameter_mm", "brock_risk_pct", "ctdivol_mgy",
            "volume_doubling_time_days", "ki67_index_pct", "pdl1_tps_pct"]
    rows = []
    for c in cols:
        s = pd.to_numeric(frame[c], errors="coerce").dropna()
        rows.append((c, f"{len(s):,}", f"{s.mean():.1f}", f"{s.median():.1f}", f"{s.quantile(0.05):.1f}",
                     f"{s.quantile(0.95):.1f}"))
    return html_table(["Variable", "Non missing", "Mean", "Median", "5th percentile", "95th percentile"], rows)


def main():
    frame = pd.read_csv(ROOT / "data" / "medfusion_lung_cohort.csv", keep_default_na=False, na_values=[""])
    nodules = frame[frame["nodule_present"] == 1]
    lines = [
        "# Dataset card: MedFusion synthetic lung nodule cohort", "",
        "All records are synthetic. No real patient data was used at any stage.", "",
        "## Overview", "",
        html_table(["Property", "Value"], [
            ("Studies (rows)", f"{len(frame):,}"), ("Columns", frame.shape[1]),
            ("Patients", f"{frame['patient_id'].nunique():,}"),
            ("Patients with repeat studies", f"{(frame.groupby('patient_id').size() > 1).sum():,}"),
            ("Studies with a nodule", f"{len(nodules):,}"),
            ("Malignancy prevalence among nodules", f"{nodules['malignancy_label'].mean() * 100:.1f}%"),
            ("Studies with histology and WSI", f"{int(frame['histology_available'].sum()):,}"),
            ("Mean report length (words)", f"{frame['integrated_report'].str.split().str.len().mean():.1f}"),
            ("Mean note length (words)", f"{frame['clinical_note'].str.split().str.len().mean():.1f}"),
        ]), "",
        "## Splits (assigned at patient level)", "", distribution(frame, "split"), "",
        "## Key distributions", "",
        distribution(frame, "referral_pathway"), "", distribution(frame, "lung_rads_category"), "",
        distribution(nodules, "nodule_type"), "", distribution(frame[frame["histology_available"] == 1], "histology_subtype"), "",
        distribution(frame, "recommendation_id"), "", distribution(frame, "growth_status"), "",
        "## Numeric summary", "", numeric_summary(frame), "",
        "## Data dictionary", "", dictionary_table(frame), "",
    ]
    out = ROOT / "docs" / "dataset_card.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
