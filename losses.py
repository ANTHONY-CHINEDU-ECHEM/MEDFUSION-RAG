"""Multitask objective for MedFusion VLM.

Total loss = language modelling (report and QA generation)
           + localisation (CenterNet style penalty reduced focal loss)
           + clinical heads (presence, diameter, type, margin, Lung RADS,
             malignancy, lymphadenopathy, histology)
           + image findings contrastive alignment (CLIP style InfoNCE)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
    """Penalty reduced pixel wise focal loss on Gaussian heatmaps."""
    prob = torch.sigmoid(logits).clamp(0.0001, 0.9999)
    positive = target.ge(0.98).float()
    negative = torch.rsub(positive, 1.0)
    pos_loss = torch.log(prob) * torch.rsub(prob, 1.0).pow(alpha) * positive
    neg_loss = torch.log(torch.rsub(prob, 1.0)) * prob.pow(alpha) * torch.rsub(target, 1.0).pow(beta) * negative
    n_pos = positive.sum().clamp(min=1.0)
    return torch.neg(pos_loss.sum() + neg_loss.sum()) / n_pos


def multitask_loss(out: dict, batch: dict, weights: dict, pad_id: int, label_smoothing: float):
    logits = out["lm_logits"]
    lm = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=pad_id,
                         label_smoothing=label_smoothing)
    present = batch["presence"]
    terms = {
        "lm": lm,
        "heatmap": centernet_focal_loss(out["heatmap"], batch["heatmap"]),
        "presence": F.binary_cross_entropy_with_logits(out["presence"], present),
        "diameter": (F.smooth_l1_loss(out["log_diameter"], batch["log_diameter"], reduction="none") * present).sum()
        / present.sum().clamp(min=1.0),
        "nodule_type": F.cross_entropy(out["nodule_type"], batch["nodule_type"]),
        "margin": F.cross_entropy(out["margin"], batch["margin"]),
        "lung_rads": F.cross_entropy(out["lung_rads"], batch["lung_rads"]),
        "malignancy": F.binary_cross_entropy_with_logits(out["malignancy"], batch["malignancy"]),
        "lymph": F.binary_cross_entropy_with_logits(out["lymph"], batch["lymph"]),
    }
    mask = batch["histology_mask"]
    if mask.sum() > 0:
        per = F.cross_entropy(out["histology"], batch["histology"], reduction="none")
        terms["histology"] = (per * mask).sum() / mask.sum()
    else:
        terms["histology"] = logits.new_zeros(())
    if "image_embed" in out:
        sim = out["logit_scale"] * out["image_embed"] @ out["text_embed"].t()
        target = torch.arange(sim.shape[0], device=sim.device)
        terms["alignment"] = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))
    total = sum(weights.get(name, 0.0) * value for name, value in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}
