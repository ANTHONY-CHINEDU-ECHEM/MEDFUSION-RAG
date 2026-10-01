"""Render a gallery of CT key images and histology tiles to docs/figures/rendered_examples.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medfusion.data.renderer import render_ct, render_wsi  # noqa: E402


def main():
    frame = pd.read_csv(ROOT / "data" / "medfusion_lung_cohort.csv", keep_default_na=False, na_values=[""])
    ct_rules = [("Solid, spiculated", (frame.nodule_type == "solid") & (frame.margin == "spiculated") & (frame.nodule_diameter_mm > 14)),
                ("Part solid", (frame.nodule_type == "part solid") & (frame.nodule_diameter_mm > 14)),
                ("Ground glass", (frame.nodule_type == "ground glass") & (frame.nodule_diameter_mm > 14)),
                ("Calcified", (frame.nodule_type == "calcified") & (frame.nodule_diameter_mm > 7)),
                ("Lobulated with nodes", (frame.margin == "lobulated") & (frame.mediastinal_lymphadenopathy == 1)),
                ("No nodule, severe emphysema", (frame.nodule_present == 0) & (frame.emphysema_severity == "severe"))]
    wsi_rules = ["adenocarcinoma", "squamous cell carcinoma", "small cell carcinoma", "carcinoid tumour", "granuloma", "hamartoma"]
    fig, axes = plt.subplots(2, 6, figsize=(17, 7.6), gridspec_kw={"hspace": 0.35})
    for ax, (title, rule) in zip(axes[0], ct_rules):
        row = frame[rule].iloc[0].to_dict()
        ax.imshow(render_ct(row), cmap="gray", vmin=0, vmax=1)
        if int(row["nodule_present"]):
            ax.add_patch(plt.Circle((row["nodule_x_norm"] * 128, row["nodule_y_norm"] * 128),
                                    row["nodule_diameter_mm"] / 5.2 + 4, fill=False, color="lime", lw=1))
        ax.set_title(f"{title}\n{row['case_id']}", fontsize=9)
        ax.axis("off")
    for ax, subtype in zip(axes[1], wsi_rules):
        row = frame[(frame.histology_subtype == subtype) & (frame.histology_available == 1)].iloc[0].to_dict()
        ax.imshow(render_wsi(row).transpose(1, 2, 0))
        ax.set_title(f"{subtype}\n{row['case_id']}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Deterministic synthetic renders: CT key images (top, green ring marks the dominant nodule) and H and E tiles (bottom)", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.05)
    out = ROOT / "docs" / "figures" / "rendered_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
