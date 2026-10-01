"""Full evaluation suite on the held out, patient level test split."""
from __future__ import annotations

import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from medfusion.data.dataset import QA_TEMPLATES, paths, qa_answer
from medfusion.data.knowledge import RECOMMENDATIONS
from medfusion.evaluation.metrics import (binary_metrics, clinical_efficacy, corpus_bleu, multiclass_metrics,
                                          retrieval_metrics, rouge_l)
from medfusion.inference.engine import InferenceEngine
from medfusion.models.components import LUNG_RADS_CLASSES
from medfusion.training.trainer import load_trained, prepare_context
from medfusion.utils import get_logger, load_json, save_json, sub, timestamp

LOGGER = get_logger("evaluate")
QA_KINDS = ["category", "location", "size", "morphology", "lymph", "next_step"]


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _normalise(text: str):
    return " ".join(text.lower().replace(".", " ").replace(",", " ").split())


def generation_scores(refs, hyps):
    out = corpus_bleu(refs, hyps)
    out["rouge_l"] = rouge_l(refs, hyps)
    out.update(clinical_efficacy(refs, hyps, RECOMMENDATIONS))
    return out


def nearest_neighbour_baseline(frame, test_rows):
    """Classic report generation baseline: copy the report of the most similar training note."""
    train = frame[frame["split"] == "train"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    train_m = vectorizer.fit_transform(train["clinical_note"])
    test_m = vectorizer.transform([r["clinical_note"] for r in test_rows])
    best = np.asarray((test_m @ train_m.T).argmax(axis=1)).ravel()
    return list(train["integrated_report"].iloc[best])


def evaluate(cfg: dict):
    torch.set_num_threads(cfg["training"]["num_threads"])
    p = paths(cfg)
    fig_dir, metric_dir = p["output"] / "figures", p["output"] / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare_context(cfg)
    model = load_trained(cfg, ctx)
    engine = InferenceEngine(cfg, ctx, model)
    frame = ctx["frame"]
    test_idx = list(np.flatnonzero((frame["split"] == "test").to_numpy()))
    bs = cfg["evaluation"]["batch_size"]
    start = time.time()

    # 1. Perception, localisation and retrieval on the full test split
    findings, rows = [], []
    for chunk in _chunks(test_idx, bs):
        batch, chunk_rows = engine.make_batch(chunk)
        _, f = engine.perceive(batch, chunk_rows)
        findings.extend(f)
        rows.extend(chunk_rows)
    LOGGER.info("Perception finished on %d test studies", len(rows))
    y_mal = np.array([r["malignancy_label"] for r in rows])
    p_mal = np.array([f["malignancy_prob"] for f in findings])
    nodule_mask = np.array([int(r["nodule_present"]) == 1 for r in rows])
    brock = np.array([r["brock_risk_pct"] if r["brock_risk_pct"] == r["brock_risk_pct"] else 0.0 for r in rows])
    results = {"evaluated_at": timestamp(), "test_studies": len(rows),
               "test_patients": int(frame.loc[test_idx, "patient_id"].nunique())}
    results["malignancy"] = binary_metrics(y_mal, p_mal)
    results["malignancy_nodules_only"] = binary_metrics(y_mal[nodule_mask], p_mal[nodule_mask])
    results["brock_reference_auroc_nodules_only"] = float(roc_auc_score(y_mal[nodule_mask], brock[nodule_mask]))
    results["nodule_presence"] = binary_metrics([int(r["nodule_present"]) for r in rows], [f["nodule_presence_prob"] for f in findings])
    results["lymphadenopathy"] = binary_metrics([int(r["mediastinal_lymphadenopathy"]) for r in rows], [f["lymph_prob"] for f in findings])
    rads_true = [str(r["lung_rads_category"]) for r in rows]
    rads_pred = [f["lung_rads_category"] for f in findings]
    results["lung_rads"] = multiclass_metrics(rads_true, rads_pred)
    order = {c: i for i, c in enumerate(LUNG_RADS_CLASSES)}
    within_one = np.mean([abs(sub(order[a], order[b])) <= 1 for a, b in zip(rads_true, rads_pred)])
    results["lung_rads"]["within_one_category"] = float(within_one)
    results["nodule_type"] = multiclass_metrics([r["nodule_type"] for r in rows], [f["nodule_type"] for f in findings])
    results["margin"] = multiclass_metrics([r["margin"] for r in rows], [f["margin"] for f in findings])
    hist_rows = [(r, f) for r, f in zip(rows, findings) if int(r["histology_available"])]
    results["histology"] = multiclass_metrics([r["histology_subtype"] for r, _ in hist_rows], [f["histology_subtype"] for _, f in hist_rows])
    results["histology"]["n"] = len(hist_rows)
    # Localisation: distance between heatmap peak and true centre for nodule cases
    spacing = 2.6 * 128
    dists, sizes = [], []
    for r, f in zip(rows, findings):
        if int(r["nodule_present"]):
            dx = sub(f["peak_x_norm"], float(r["nodule_x_norm"])) * spacing
            dy = sub(f["peak_y_norm"], float(r["nodule_y_norm"])) * spacing
            dists.append(math.hypot(dx, dy))
            sizes.append(float(r["nodule_diameter_mm"]))
    dists, sizes = np.array(dists), np.array(sizes)
    results["localisation"] = {
        "median_error_mm": float(np.median(dists)), "mean_error_mm": float(dists.mean()),
        "hit_rate_within_10mm": float((dists <= 10).mean()), "hit_rate_within_20mm": float((dists <= 20).mean()),
        "hit_rate_within_10mm_nodules_8mm_plus": float((dists[sizes >= 8] <= 10).mean()),
        "hit_rate_within_10mm_nodules_below_6mm": float((dists[sizes < 6] <= 10).mean()),
    }
    # Retrieval from predicted findings versus oracle findings
    gold = [set(r["gold_guideline_doc_ids"].split("|")) for r in rows]
    _, hits = engine.retrieve(findings)
    predicted_ids = [[h.doc_id for h in hs] for hs in hits]
    oracle_ids = [[engine.retriever.doc_ids[i] for i in ctx["passages_oracle"][j]] for j in test_idx]
    results["retrieval_predicted_findings"] = retrieval_metrics(gold, predicted_ids)
    results["retrieval_oracle_findings"] = retrieval_metrics(gold, oracle_ids)
    subgroups = {}
    for key in ["scanner_vendor", "sex", "referral_pathway"]:
        subgroups[key] = {}
        for value in sorted({r[key] for r in rows}):
            m = np.array([r[key] == value for r in rows]) & nodule_mask
            if m.sum() > 30 and len(set(y_mal[m])) == 2:
                subgroups[key][value] = {"n": int(m.sum()), "auroc": float(roc_auc_score(y_mal[m], p_mal[m]))}
    results["malignancy_subgroups_nodules_only"] = subgroups

    # 2. Report generation with predicted retrieval, plus baseline and ablations
    rng = np.random.default_rng(cfg["seed"])
    n_gen = min(cfg["evaluation"]["generation_cases"], len(test_idx))
    gen_idx = sorted(rng.choice(test_idx, n_gen, replace=False).tolist())
    refs, hyps, samples = [], [], []
    for chunk in _chunks(gen_idx, bs):
        out = engine.run(chunk, task="report", retrieval="predicted")
        for row, text, f, hs in zip(out["rows"], out["texts"], out["findings"], out["hits"]):
            refs.append(row["integrated_report"])
            hyps.append(text)
            if len(samples) < 40:
                samples.append({"case_id": row["case_id"], "reference": row["integrated_report"], "generated": text,
                                "retrieved": [h.doc_id for h in hs], "predicted_lung_rads": f["lung_rads_category"],
                                "malignancy_prob": f["malignancy_prob"]})
    LOGGER.info("Generated %d reports (%.0fs)", len(hyps), sub(time.time(), start))
    gen_rows = frame.iloc[gen_idx].to_dict("records")
    results["report_generation"] = {"n": len(hyps), "medfusion_rag": generation_scores(refs, hyps),
                                    "nearest_neighbour_baseline": generation_scores(refs, nearest_neighbour_baseline(frame, gen_rows))}
    n_ab = min(cfg["evaluation"]["ablation_cases"], len(gen_idx))
    ab_idx = gen_idx[:n_ab]
    ab_refs = refs[:n_ab]
    ablations = {"predicted_retrieval": generation_scores(ab_refs, hyps[:n_ab])}
    for mode in ["none", "oracle"]:
        texts = []
        for chunk in _chunks(ab_idx, bs):
            texts.extend(engine.run(chunk, task="report", retrieval=mode)["texts"])
        ablations[f"{mode}_retrieval"] = generation_scores(ab_refs, texts)
    results["retrieval_ablation"] = {"n": n_ab, **ablations}
    LOGGER.info("Ablations finished (%.0fs)", sub(time.time(), start))

    # 3. Visual question answering
    qa_idx = gen_idx[: cfg["evaluation"]["qa_cases"]]
    qa = {}
    for kind in QA_KINDS:
        correct = total = 0
        for chunk in _chunks(qa_idx, bs):
            out = engine.run(chunk, task="qa", qa_kind=kind, retrieval="predicted")
            for row, text in zip(out["rows"], out["texts"]):
                correct += int(_normalise(text) == _normalise(qa_answer(row, kind)))
                total += 1
        qa[kind] = correct / max(total, 1)
    qa["mean_exact_match"] = float(np.mean([qa[k] for k in QA_KINDS]))
    results["question_answering_exact_match"] = qa
    results["runtime_seconds"] = sub(time.time(), start)
    save_json(results, metric_dir / "test_metrics.json")
    with open(metric_dir / "sample_generations.jsonl", "w", encoding="utf8") as handle:
        for s in samples:
            handle.write(json.dumps(s, ensure_ascii=False) + "\n")
    _figures(cfg, fig_dir, y_mal, p_mal, nodule_mask, brock, rads_true, rads_pred, dists)
    LOGGER.info("Evaluation complete: %s", json.dumps({k: results[k] for k in ["malignancy", "lung_rads"]}))
    return results


def _figures(cfg, fig_dir, y_mal, p_mal, nodule_mask, brock, rads_true, rads_pred, dists):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for label, scores, mask in [("MedFusion (all studies)", p_mal, np.ones_like(nodule_mask)),
                                ("MedFusion (nodules)", p_mal, nodule_mask),
                                ("Brock reference (nodules)", brock / 100.0, nodule_mask)]:
        fpr, tpr, _ = roc_curve(y_mal[mask], scores[mask])
        ax[0].plot(fpr, tpr, label=f"{label}  AUROC {roc_auc_score(y_mal[mask], scores[mask]):.3f}")
    ax[0].plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax[0].set(title="Malignancy ROC", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(fontsize=8, loc="lower right")
    bins = np.linspace(0, 1, 11)
    centres, observed = [], []
    for lo, hi in zip(bins[:~0], bins[1:]):
        m = (p_mal >= lo) & (p_mal < hi)
        if m.sum() >= 10:
            centres.append(p_mal[m].mean())
            observed.append(y_mal[m].mean())
    ax[1].plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax[1].plot(centres, observed, marker="o")
    ax[1].set(title="Malignancy calibration", xlabel="Predicted probability", ylabel="Observed frequency")
    cm = confusion_matrix(rads_true, rads_pred, labels=LUNG_RADS_CLASSES)
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax[2].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax[2].set(xticks=range(6), yticks=range(6), xticklabels=LUNG_RADS_CLASSES, yticklabels=LUNG_RADS_CLASSES,
              title="Lung RADS confusion (row normalised)", xlabel="Predicted", ylabel="Reference")
    for i in range(6):
        for j in range(6):
            ax[2].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_performance.png", dpi=130)
    plt.close(fig)
    history_path = paths(cfg)["output"] / "metrics" / "training_history.json"
    if history_path.exists():
        hist = load_json(history_path)["history"]
        epochs = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(epochs, [h["train"]["lm"] for h in hist], marker="o", label="train")
        ax[0].plot(epochs, [h["validation"]["lm"] for h in hist], marker="o", label="validation")
        ax[0].set(title="Language modelling loss", xlabel="Epoch")
        ax[0].legend()
        for key in ["lung_rads", "malignancy", "histology", "heatmap"]:
            ax[1].plot(epochs, [h["validation"][key] for h in hist], marker="o", label=key)
        ax[1].set(title="Validation head losses", xlabel="Epoch")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=130)
        plt.close(fig)
