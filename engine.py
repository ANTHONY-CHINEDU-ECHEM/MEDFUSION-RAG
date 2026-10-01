"""Shared inference engine used by evaluation, explainability and the assistant.

Inference is a two pass retrieval augmented procedure:
    1. Encode images and note, run the clinical heads.
    2. Convert head predictions into structured findings and build a query.
    3. Retrieve guidance passages with the hybrid retriever.
    4. Decode the report or answer while attending to fused memory and passages.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from medfusion.data.dataset import QA_TEMPLATES, REPORT_PROMPT, MedFusionDataset, collate
from medfusion.models.components import HISTOLOGY_CLASSES, LUNG_RADS_CLASSES, MARGIN_CLASSES, NODULE_TYPE_CLASSES
from medfusion.rag.retriever import build_query


class InferenceEngine:
    def __init__(self, cfg: dict, ctx: dict, model):
        self.cfg, self.ctx, self.model = cfg, ctx, model.eval()
        self.tok = ctx["tokenizer"]
        self.retriever = ctx["retriever"]
        self.frame = ctx["frame"]
        self.k = cfg["retrieval"]["top_k"]

    # Batching
    def make_batch(self, row_indices: list[int]):
        sub_frame = self.frame.iloc[row_indices].reset_index(drop=True)
        ds = MedFusionDataset(sub_frame, np.asarray(row_indices), self.ctx["ct"], self.ctx["wsi"], self.tok,
                              self.ctx["passages_oracle"], self.cfg, train=False, task="report")
        items = [ds[i] for i in range(len(ds))]
        return collate(items, self.tok.pad_id), sub_frame.to_dict("records")

    def make_batch_from_records(self, records: list[dict]):
        """Build a batch for studies that are not in the cached cohort (images rendered on demand)."""
        import pandas as pd
        from medfusion.data.renderer import render_ct, render_wsi
        sub_frame = pd.DataFrame(records)
        ct = np.stack([np.round(render_ct(r) * 255).astype(np.uint8) for r in records])
        wsi = np.stack([np.round(render_wsi(r) * 255).astype(np.uint8) for r in records])
        dummy = np.zeros((len(records), self.k), dtype=np.int64)
        ds = MedFusionDataset(sub_frame, np.arange(len(records)), ct, wsi, self.tok, dummy, self.cfg,
                              train=False, task="report")
        return collate([ds[i] for i in range(len(ds))], self.tok.pad_id), records

    # Pass 1: perception
    @torch.no_grad()
    def perceive(self, batch: dict, rows: list[dict]):
        enc = self.model.encode(batch)
        h = enc["heads"]
        presence = torch.sigmoid(h["presence"])
        malignancy = torch.sigmoid(h["malignancy"])
        lymph = torch.sigmoid(h["lymph"])
        type_prob = torch.softmax(h["nodule_type"], ~0)
        margin_prob = torch.softmax(h["margin"], ~0)
        rads_prob = torch.softmax(h["lung_rads"], ~0)
        hist_prob = torch.softmax(h["histology"], ~0)
        heat = torch.sigmoid(h["heatmap"])
        findings = []
        for i, row in enumerate(rows):
            present = bool(presence[i] >= 0.5)
            type_idx = int(type_prob[i, 1:].argmax()) + 1 if present else 0
            margin_idx = int(margin_prob[i, 1:].argmax()) + 1 if present else 0
            grid = heat[i]
            flat = int(grid.argmax())
            gy, gx = divmod(flat, grid.shape[1])
            wsi = bool(batch["wsi_present"][i])
            findings.append({
                "case_id": row["case_id"], "referral_pathway": row["referral_pathway"],
                "nodule_present": int(present), "nodule_presence_prob": float(presence[i]),
                "nodule_type": NODULE_TYPE_CLASSES[type_idx], "margin": MARGIN_CLASSES[margin_idx],
                "nodule_diameter_mm": float(math.exp(float(h["log_diameter"][i]))) if present else None,
                "lung_rads_category": LUNG_RADS_CLASSES[int(rads_prob[i].argmax())],
                "lung_rads_probs": {c: float(p) for c, p in zip(LUNG_RADS_CLASSES, rads_prob[i])},
                "malignancy_prob": float(malignancy[i]), "mediastinal_lymphadenopathy": int(lymph[i] >= 0.5),
                "lymph_prob": float(lymph[i]),
                "histology_subtype": HISTOLOGY_CLASSES[int(hist_prob[i].argmax())] if wsi else "not sampled",
                "histology_prob": float(hist_prob[i].max()) if wsi else None,
                "peak_x_norm": (gx + 0.5) / grid.shape[1], "peak_y_norm": (gy + 0.5) / grid.shape[0],
                "peak_confidence": float(grid.max()),
            })
        return enc, findings

    # Pass 2: retrieval
    def retrieve(self, findings: list[dict]):
        idx, hits_all = [], []
        for f in findings:
            query, tags = build_query(f)
            hits = self.retriever.search(query, tags, k=self.k, management_slots=self.cfg["retrieval"]["management_slots"])
            idx.append([self.retriever.index[h.doc_id] for h in hits])
            hits_all.append(hits)
        return torch.tensor(idx, dtype=torch.long), hits_all

    def oracle_passages(self, row_indices: list[int]):
        return torch.from_numpy(self.ctx["passages_oracle"][row_indices])

    # Pass 3: generation
    def prompt(self, task: str, question: str | None, batch_size: int):
        if task == "report":
            ids = [self.tok.bos_id, self.tok.token_to_id["[TASK_REPORT]"]] + self.tok.encode(REPORT_PROMPT)
        else:
            ids = [self.tok.bos_id, self.tok.token_to_id["[TASK_QA]"]] + self.tok.encode(question)
        ids = ids + [self.tok.sep_id]
        return torch.tensor([ids] * batch_size, dtype=torch.long)

    @torch.no_grad()
    def generate(self, enc: dict, passage_idx: torch.Tensor | None, task: str = "report", question: str | None = None,
                 use_retrieval: bool = True):
        b = enc["fused"].shape[0]
        if passage_idx is None:
            passage_idx = torch.zeros(b, self.k, dtype=torch.long)
            use_retrieval = False
        keep = torch.full(passage_idx.shape, use_retrieval, dtype=torch.bool)
        out = self.model.generate(enc, self.prompt(task, question, b), passage_idx, keep,
                                  self.cfg["evaluation"]["max_new_tokens"], self.tok.eos_id)
        out["texts"] = [self.tok.decode(ids) for ids in out["ids"]]
        return out

    def run(self, row_indices: list[int], task: str = "report", qa_kind: str | None = None,
            retrieval: str = "predicted"):
        """Full pipeline for a list of dataset row indices."""
        batch, rows = self.make_batch(row_indices)
        enc, findings = self.perceive(batch, rows)
        hits = None
        if retrieval == "predicted":
            passage_idx, hits = self.retrieve(findings)
        elif retrieval == "oracle":
            passage_idx = self.oracle_passages(row_indices)
        else:
            passage_idx = None
        question = QA_TEMPLATES[qa_kind] if task == "qa" else None
        gen = self.generate(enc, passage_idx, task, question, use_retrieval=retrieval != "none")
        return {"rows": rows, "findings": findings, "hits": hits, "texts": gen["texts"], "generation": gen,
                "encoding": enc, "batch": batch, "passage_idx": passage_idx}
