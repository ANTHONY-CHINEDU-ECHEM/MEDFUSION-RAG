"""Evaluation metrics implemented from first principles where practical.

Natural language generation: corpus BLEU 1 to 4 and ROUGE L.
Clinical efficacy: structured entity extraction from reports followed by
micro and per field accuracy (analogous to CheXbert style label agreement).
Classification: AUROC, AUPRC, Brier score, expected calibration error.
Localisation: peak to centre distance in millimetres and hit rate.
Retrieval: recall at k and mean reciprocal rank.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)

from medfusion.data.tokenizer import ClinicalTokenizer
from medfusion.utils import sub


def _tokens(text: str):
    return [t for t in ClinicalTokenizer.split(text) if t.isalnum()]


def corpus_bleu(references: list[str], hypotheses: list[str], max_n: int = 4):
    """Corpus level BLEU with brevity penalty (Papineni et al. 2002)."""
    matches, totals = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = _tokens(ref), _tokens(hyp)
        ref_len += len(r)
        hyp_len += len(h)
        for n in range(1, max_n + 1):
            r_counts = Counter(tuple(r[i:i + n]) for i in range(max(0, len(r) + 1 + ~n + 1)))
            h_counts = Counter(tuple(h[i:i + n]) for i in range(max(0, len(h) + 1 + ~n + 1)))
            matches[n + ~0] += sum(min(c, r_counts[g]) for g, c in h_counts.items())
            totals[n + ~0] += max(0, sum(h_counts.values()))
    out = {}
    bp = 1.0 if hyp_len > ref_len else math.exp(sub(1.0, ref_len / max(hyp_len, 1)))
    log_sum = 0.0
    for n in range(1, max_n + 1):
        precision = matches[n + ~0] / max(totals[n + ~0], 1)
        log_sum += math.log(max(precision, 1.0 / 1e9))
        out[f"bleu_{n}"] = bp * math.exp(log_sum / n)
    return out


def rouge_l(references: list[str], hypotheses: list[str]):
    """Mean sentence level ROUGE L F measure using longest common subsequence."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        r, h = _tokens(ref), _tokens(hyp)
        if not r or not h:
            scores.append(0.0)
            continue
        prev = [0] * (len(h) + 1)
        for a in r:
            cur = [0] * (len(h) + 1)
            for j, b in enumerate(h, start=1):
                cur[j] = prev[j + ~0] + 1 if a == b else max(prev[j], cur[j + ~0])
            prev = cur
        lcs = prev[~0]
        p, rc = lcs / len(h), lcs / len(r)
        scores.append(0.0 if lcs == 0 else 2 * p * rc / (p + rc))
    return float(np.mean(scores))


LOBE_WORDS = {"right upper lobe": "RUL", "right middle lobe": "RML", "right lower lobe": "RLL",
              "left upper lobe": "LUL", "left lower lobe": "LLL"}
SUBTYPES = ["squamous cell carcinoma", "small cell carcinoma", "large cell carcinoma", "adenocarcinoma",
            "carcinoid", "granuloma", "granulomatous", "hamartoma", "organising pneumonia", "nondiagnostic"]


def extract_entities(report: str):
    """Parse the clinically important fields out of a free text report."""
    text = report.lower()
    ents = {}
    m = re.search(r"lung rads (4[abx]|[123])", text)
    ents["lung_rads"] = m.group(1).upper() if m else "missing"
    ents["nodule_present"] = "0" if ("no pulmonary nodule" in text or "no suspicious pulmonary nodule is seen" in text) else "1"
    ents["lobe"] = next((code for words, code in LOBE_WORDS.items() if words in text), "none")
    m = re.search(r"(\d+) mm (?:solid|part solid|pure ground glass|calcified) nodule|measures (\d+) mm", text)
    ents["size_mm"] = int(m.group(1) or m.group(2)) if m else None
    ents["nodule_type"] = next((t for t in ["part solid", "ground glass", "calcified", "solid"] if f"{t} nodule" in text), "none")
    ents["margin"] = next((t for t in ["spiculated", "lobulated", "smooth"] if f"{t} margins" in text), "none")
    ents["lymphadenopathy"] = "1" if "enlarged mediastinal lymph nodes" in text else "0"
    ents["emphysema"] = next((t for t in ["severe", "moderate", "mild"] if f"{t} centrilobular emphysema" in text), "none")
    path = text.split("pathology:")[1].split("impression:")[0] if "pathology:" in text and "impression:" in text else text
    ents["histology"] = next((s for s in SUBTYPES if s in path), "none")
    if ents["histology"] == "granulomatous":
        ents["histology"] = "granuloma"
    m = re.search(r"clinical stage (\w+)", text)
    ents["stage"] = m.group(1).upper() if m else "none"
    rec = text.split("recommendation:")[1].strip() if "recommendation:" in text else ""
    ents["recommendation"] = rec
    return ents


def clinical_efficacy(references: list[str], hypotheses: list[str], recommendations: dict):
    """Per field agreement between entities parsed from generated and reference reports."""
    rec_lookup = {v.lower(): k for k, v in recommendations.items()}
    fields = ["lung_rads", "nodule_present", "lobe", "nodule_type", "margin", "lymphadenopathy", "emphysema",
              "histology", "stage", "recommendation"]
    hits = {f: 0 for f in fields}
    size_errors = []
    for ref, hyp in zip(references, hypotheses):
        r, h = extract_entities(ref), extract_entities(hyp)
        for f in fields:
            if f == "recommendation":
                hits[f] += int(rec_lookup.get(r[f], "x") == rec_lookup.get(h[f], "y"))
            else:
                hits[f] += int(r[f] == h[f])
        if r["size_mm"] is not None and h["size_mm"] is not None:
            size_errors.append(abs(sub(r["size_mm"], h["size_mm"])))
    n = max(len(references), 1)
    out = {f"{f}_accuracy": hits[f] / n for f in fields}
    out["clinical_entity_macro_accuracy"] = float(np.mean([hits[f] / n for f in fields]))
    out["size_mae_mm"] = float(np.mean(size_errors)) if size_errors else None
    out["size_within_2mm"] = float(np.mean(np.array(size_errors) <= 2)) if size_errors else None
    return out


def expected_calibration_error(y_true, y_prob, bins: int = 10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:~0], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(sub(y_prob[mask].mean(), y_true[mask].mean()))
    return float(ece)


def binary_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)), "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)), "ece": expected_calibration_error(y_true, y_prob),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "prevalence": float(y_true.mean()),
    }


def multiclass_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"accuracy": float((y_true == y_pred).mean()), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def retrieval_metrics(gold_sets: list[set], retrieved: list[list[str]], ks=(1, 3)):
    out = {}
    for k in ks:
        out[f"recall_at_{k}"] = float(np.mean([bool(g & set(r[:k])) for g, r in zip(gold_sets, retrieved)]))
    rr = []
    for g, r in zip(gold_sets, retrieved):
        rank = next((i + 1 for i, d in enumerate(r) if d in g), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = float(np.mean(rr))
    return out
