"""MedFusion clinical assistant: the conversational face of the model.

The assistant exposes two capabilities:
    analyse(case)          integrated report, structured findings, citations
    ask(case, question)    grounded answer to a clinical question about the case

Free text questions are routed to the closest trained instruction by lexical
similarity. The model was instruction tuned on eight question families, so
questions far outside them fall back to the full report rather than risking an
ungrounded answer.
"""
from __future__ import annotations

import numpy as np
import torch

from medfusion.data.dataset import QA_TEMPLATES, paths
from medfusion.data.generator import CohortGenerator
from medfusion.data.tokenizer import ClinicalTokenizer
from medfusion.inference.engine import InferenceEngine
from medfusion.training.trainer import load_trained, prepare_context
from medfusion.utils import get_logger, save_json

LOGGER = get_logger("assistant")
DISCLAIMER = "Research prototype trained on synthetic data. Not for clinical use."
INTENT_HINTS = {
    "category": "lung rads category risk class score",
    "location": "where location lobe which side located",
    "size": "size diameter how big large measure mm",
    "morphology": "morphology describe shape margin type appearance spiculated solid",
    "lymph": "lymph nodes lymphadenopathy mediastinal hilar",
    "histology": "histology pathology biopsy tissue diagnosis subtype",
    "impression": "impression summary summarise conclusion overall",
    "next_step": "next step recommend management follow up plan what should do",
}


def route_question(question: str):
    words = set(ClinicalTokenizer.split(question))
    best, best_score = None, 0.0
    for intent, template in QA_TEMPLATES.items():
        vocab = set(ClinicalTokenizer.split(template + " " + INTENT_HINTS[intent]))
        score = len(words & vocab) / max(len(words), 1)
        if score > best_score:
            best, best_score = intent, score
    return (best, best_score) if best_score >= 0.2 else (None, best_score)


class MedFusionAssistant:
    def __init__(self, cfg: dict):
        torch.set_num_threads(cfg["training"]["num_threads"])
        self.cfg = cfg
        self.ctx = prepare_context(cfg)
        self.engine = InferenceEngine(cfg, self.ctx, load_trained(cfg, self.ctx))
        self.case_lookup = {c: i for i, c in enumerate(self.ctx["frame"]["case_id"])}

    def _run(self, case, task="report", qa_kind=None):
        if isinstance(case, str):
            return self.engine.run([self.case_lookup[case]], task=task, qa_kind=qa_kind)
        batch, rows = self.engine.make_batch_from_records([case])
        enc, findings = self.engine.perceive(batch, rows)
        passage_idx, hits = self.engine.retrieve(findings)
        question = QA_TEMPLATES[qa_kind] if task == "qa" else None
        gen = self.engine.generate(enc, passage_idx, task, question)
        return {"rows": rows, "findings": findings, "hits": hits, "texts": gen["texts"]}

    def _citations(self, hits):
        records = self.engine.retriever.records
        index = self.engine.retriever.index
        return [{"doc_id": h.doc_id, "title": records[index[h.doc_id]]["title"],
                 "source": records[index[h.doc_id]]["source_family"]} for h in hits]

    def analyse(self, case):
        out = self._run(case)
        f = out["findings"][0]
        return {
            "case_id": out["rows"][0]["case_id"], "report": out["texts"][0],
            "malignancy_probability": round(f["malignancy_prob"], 3), "lung_rads": f["lung_rads_category"],
            "nodule_detected": bool(f["nodule_present"]), "histology_prediction": f["histology_subtype"],
            "citations": self._citations(out["hits"][0]), "disclaimer": DISCLAIMER,
        }

    def ask(self, case, question: str):
        intent, score = route_question(question)
        if intent is None:
            analysis = self.analyse(case)
            return {"question": question, "intent": "fallback_full_report", "answer": analysis["report"],
                    "citations": analysis["citations"], "routing_score": score}
        out = self._run(case, task="qa", qa_kind=intent)
        return {"question": question, "intent": intent, "answer": out["texts"][0],
                "citations": self._citations(out["hits"][0]), "routing_score": round(score, 3)}


def demo(cfg: dict):
    assistant = MedFusionAssistant(cfg)
    frame = assistant.ctx["frame"]
    test = frame[frame["split"] == "test"]
    cases = [test[test["histology_subtype"].eq("adenocarcinoma")].iloc[0]["case_id"],
             test[test["lung_rads_category"].astype(str).eq("4A") & test["histology_available"].eq(0)].iloc[0]["case_id"]]
    unseen = CohortGenerator(n_rows=40, seed=777).generate()
    unseen_case = unseen[unseen["nodule_type"].eq("solid") & unseen["nodule_diameter_mm"].gt(10)].iloc[0].to_dict()
    unseen_case["case_id"] = "NEW" + unseen_case["case_id"]
    questions = ["What is the Lung RADS category?", "Where is the nodule?", "How big is it?",
                 "What should happen next?", "Are the mediastinal lymph nodes enlarged?"]
    transcript = ["# MedFusion assistant demo transcript", "", DISCLAIMER, ""]
    sessions = []
    for case in cases + [unseen_case]:
        analysis = assistant.analyse(case)
        row = frame[frame["case_id"] == case].iloc[0].to_dict() if isinstance(case, str) else case
        qa = [assistant.ask(case, q) for q in questions]
        sessions.append({"analysis": analysis, "questions": qa})
        label = analysis["case_id"] + (" (unseen study rendered on demand)" if not isinstance(case, str) else "")
        transcript += [f"## Case {label}", "", "**Clinical note (input).** " + row["clinical_note"], "",
                       "**Generated report.** " + analysis["report"], "",
                       "**Reference report.** " + row["integrated_report"], "",
                       f"**Malignancy probability:** {analysis['malignancy_probability']}  ",
                       f"**Predicted Lung RADS:** {analysis['lung_rads']}  ",
                       "**Citations:** " + "; ".join(f"{c['doc_id']} {c['title']}" for c in analysis["citations"]), ""]
        for item in qa:
            transcript += [f"**Q:** {item['question']}  ", f"**A:** {item['answer']} *(intent: {item['intent']})*", ""]
        LOGGER.info("Demo case %s: %s", analysis["case_id"], analysis["report"][:120])
    out_dir = paths(cfg)["output"]
    (out_dir / "demo_transcript.md").write_text("\n".join(transcript), encoding="utf8")
    save_json(sessions, out_dir / "metrics" / "demo_sessions.json")
    return {"sessions": sessions}
