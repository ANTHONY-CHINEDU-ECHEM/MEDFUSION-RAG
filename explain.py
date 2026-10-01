"""Explainability (XAI) module.

For each explained case the module produces four complementary views:

1. Localisation heatmap from the dedicated detection head.
2. Grad CAM on the middle CT pyramid level with respect to the malignancy logit.
3. Decoder cross attention: where the language model looked on the CT while
   writing the findings section (attention to CT tokens, both pyramid levels).
4. Knowledge graph evidence chains linking observed findings to the predicted
   diagnosis, each edge citing the guidance passage that supports it, together
   with the retrieved passages that grounded the recommendation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from medfusion.data.dataset import paths
from medfusion.evaluation.metrics import extract_entities
from medfusion.inference.engine import InferenceEngine
from medfusion.rag.graph import KnowledgeGraph
from medfusion.training.trainer import load_trained, prepare_context
from medfusion.utils import get_logger, save_json

LOGGER = get_logger("xai")
TYPE_NODES = {"solid": "solid_nodule", "part solid": "part_solid_nodule", "ground glass": "ground_glass_nodule",
              "calcified": "calcified_nodule"}
DIAGNOSIS_NODES = {"adenocarcinoma": "adenocarcinoma", "squamous cell carcinoma": "squamous_cell_carcinoma",
                   "small cell carcinoma": "small_cell_carcinoma", "large cell carcinoma": "large_cell_carcinoma",
                   "carcinoid tumour": "carcinoid_tumour", "granuloma": "granuloma", "hamartoma": "hamartoma",
                   "organising pneumonia": "organising_pneumonia", "nondiagnostic sample": "nondiagnostic"}


def grad_cam(model, batch: dict):
    """Grad CAM over the middle CT pyramid level (16 by 16) for the malignancy logit."""
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        enc = model.encode(batch)
        fmap = enc["ct"]["mid_map"]
        fmap.retain_grad()
        enc["heads"]["malignancy"].sum().backward()
        weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * fmap).sum(1))[0].detach()
    cam = cam / cam.max().clamp(min=1.0 / 1e6)
    return cam.numpy()


def cross_attention_map(attention: torch.Tensor, layout: dict, n_tokens: int, grid: int = 8):
    """Aggregate decoder cross attention onto the CT token grid."""
    a, b = layout["ct"]
    att = attention[0, :n_tokens, a:b].mean(0)
    half = att.shape[0] // 2
    coarse, mid = att[:half], att[half:]
    grid_map = (coarse + mid).reshape(grid, grid)
    return (grid_map / grid_map.max().clamp(min=1.0 / 1e9)).numpy()


def observed_concepts(row: dict, finding: dict, report_entities: dict):
    concepts = []
    if finding["nodule_present"]:
        concepts.append(TYPE_NODES.get(finding["nodule_type"], "solid_nodule"))
        if finding["margin"] == "spiculated":
            concepts.append("spiculation")
        if finding["margin"] == "lobulated":
            concepts.append("lobulated_margin")
        if (finding["nodule_diameter_mm"] or 0) >= 15:
            concepts.append("large_nodule")
        if report_entities["lobe"] in ("RUL", "LUL"):
            concepts.append("upper_lobe")
    if finding["mediastinal_lymphadenopathy"]:
        concepts.append("lymphadenopathy")
    if report_entities["emphysema"] != "none":
        concepts.append("emphysema")
    if "growth" in row["findings_text"].lower() and "doubling time" in row["findings_text"].lower():
        concepts.append("rapid_growth")
    if row["smoking_status"] == "never":
        concepts.append("never_smoker")
    elif row["pack_years"] == row["pack_years"] and float(row["pack_years"]) >= 30:
        concepts.append("smoking")
    if int(row["family_history_lung_cancer"]):
        concepts.append("family_history")
    concepts.append(f"lung_rads_{finding['lung_rads_category'].lower()}")
    return concepts


def _select_cases(frame):
    test = frame[frame["split"] == "test"]
    rules = [
        test["histology_subtype"].eq("adenocarcinoma") & test["margin"].eq("spiculated"),
        test["histology_subtype"].eq("small cell carcinoma"),
        test["growth_status"].eq("growing"),
        test["nodule_type"].eq("calcified"),
        test["histology_subtype"].eq("granuloma"),
        test["nodule_present"].eq(0),
        test["nodule_type"].eq("part solid") & test["histology_available"].eq(0),
        test["histology_subtype"].eq("squamous cell carcinoma"),
    ]
    picked = []
    for rule in rules:
        candidates = [i for i in test.index[rule] if i not in picked]
        if candidates:
            picked.append(int(candidates[0]))
    return picked


def explain_cases(cfg: dict, case_indices: list[int] | None = None):
    torch.set_num_threads(cfg["training"]["num_threads"])
    p = paths(cfg)
    out_dir = p["output"] / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare_context(cfg)
    model = load_trained(cfg, ctx)
    engine = InferenceEngine(cfg, ctx, model)
    graph = KnowledgeGraph.load(p["kg"])
    frame = ctx["frame"]
    indices = case_indices or _select_cases(frame)[: cfg["evaluation"]["explanation_cases"]]
    explanations = []
    for idx in indices:
        result = engine.run([idx], task="report", retrieval="predicted")
        row, finding, report = result["rows"][0], result["findings"][0], result["texts"][0]
        gen = result["generation"]
        decoded = engine.tok.decode_tokens(gen["ids"][0])
        n_findings = len(decoded)
        if "pathology" in decoded:
            n_findings = decoded.index("pathology")
        att_map = cross_attention_map(gen["cross_attention"], result["encoding"]["layout"], max(n_findings, 1))
        cam = grad_cam(model, result["batch"])
        entities = extract_entities(report)
        concepts = observed_concepts(row, finding, entities)
        if finding["histology_subtype"] in DIAGNOSIS_NODES:
            target = DIAGNOSIS_NODES[finding["histology_subtype"]]
        else:
            target = "malignancy" if finding["malignancy_prob"] >= 0.5 else "benign"
        chains = graph.evidence_paths(concepts, target)
        parent = None
        if target in DIAGNOSIS_NODES.values() and target != "nondiagnostic":
            parent = "benign" if target in ("granuloma", "hamartoma", "organising_pneumonia") else "malignancy"
            chains += graph.evidence_paths(concepts, parent)
        chains = sorted(chains, key=lambda c: c["strength"], reverse=True)[:6]
        passages = [{"doc_id": h.doc_id, "title": engine.retriever.records[engine.retriever.index[h.doc_id]]["title"],
                     "rank": h.rank, "fused_score": h.score, "components": h.components} for h in result["hits"][0]]
        record = {
            "case_id": row["case_id"], "generated_report": report, "reference_report": row["integrated_report"],
            "predicted_findings": {k: v for k, v in finding.items() if k not in ("lung_rads_probs",)},
            "lung_rads_probabilities": finding["lung_rads_probs"],
            "reference": {"malignancy_label": int(row["malignancy_label"]), "lung_rads": str(row["lung_rads_category"]),
                          "histology": row["histology_subtype"]},
            "observed_concepts": concepts, "explanation_target": graph.label(target),
            "explanation_parent": graph.label(parent) if parent else None,
            "knowledge_graph_evidence": chains, "retrieved_guidance": passages,
        }
        explanations.append(record)
        save_json(record, out_dir / f"{row['case_id']}.json")
        _panel(ctx, idx, row, finding, result, cam, att_map, out_dir / f"{row['case_id']}.png", report)
        LOGGER.info("Explained %s", row["case_id"])
    _write_markdown(explanations, out_dir / "explanations_summary.md")
    return explanations


def _overlay(ax, base, overlay, title):
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    up = np.kron(overlay, np.ones((base.shape[0] // overlay.shape[0], base.shape[1] // overlay.shape[1])))
    ax.imshow(up, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _panel(ctx, idx, row, finding, result, cam, att_map, path, report):
    ct = np.asarray(ctx["ct"][idx], dtype=np.float32) / 255.0
    heat = torch.sigmoid(result["encoding"]["heads"]["heatmap"][0]).numpy()
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    if int(row["nodule_present"]):
        r = max(float(row["nodule_diameter_mm"]) / 2.6 / 2 + 3, 4)
        axes[0].add_patch(plt.Circle((float(row["nodule_x_norm"]) * 128, float(row["nodule_y_norm"]) * 128), r,
                                     fill=False, color="lime", lw=1.2))
    axes[0].set_title(f"{row['case_id']} CT key image (reference ring)", fontsize=9)
    axes[0].axis("off")
    _overlay(axes[1], ct, heat / max(heat.max(), 1.0 / 1e6), f"Localisation head (peak {finding['peak_confidence']:.2f})")
    _overlay(axes[2], ct, cam, f"Grad CAM malignancy (p={finding['malignancy_prob']:.2f})")
    _overlay(axes[3], ct, att_map, "Decoder cross attention (findings)")
    if int(row["histology_available"]):
        axes[4].imshow(np.asarray(ctx["wsi"][idx]).transpose(1, 2, 0))
        axes[4].set_title(f"WSI tile: predicted {finding['histology_subtype']}", fontsize=9)
    else:
        axes[4].text(0.5, 0.5, "No histology available", ha="center", va="center")
    axes[4].axis("off")
    fig.suptitle(f"Predicted Lung RADS {finding['lung_rads_category']} | reference {row['lung_rads_category']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_markdown(explanations: list[dict], path):
    lines = ["# Explainability summary", "", "Synthetic cases only. Not for clinical use.", ""]
    for e in explanations:
        lines += [f"## {e['case_id']}", "", f"![{e['case_id']}]({e['case_id']}.png)", "",
                  f"**Generated report.** {e['generated_report']}", "",
                  f"**Reference report.** {e['reference_report']}", "",
                  f"**Explanation target:** {e['explanation_target']}", "", "**Knowledge graph evidence chains**", ""]
        if not e["knowledge_graph_evidence"]:
            lines.append("* No supporting chain within three hops.")
        for chain in e["knowledge_graph_evidence"]:
            steps = " then ".join(f"{s['from']} {s['relation']} {s['to']} [{s['evidence']}]" for s in chain["steps"])
            lines.append(f"* Strength {chain['strength']}: {steps}")
        lines += ["", "**Retrieved guidance**", ""]
        lines += [f"* Rank {g['rank']}: {g['doc_id']} {g['title']}" for g in e["retrieved_guidance"]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf8")
