"""Asset preparation and the PyTorch dataset for instruction style training.

The model is trained as an instruction following multimodal language model.
Each example pairs the images and clinical note with one task:

* [TASK_REPORT]  generate the full integrated radiology pathology report
* [TASK_QA]      answer a clinical question about the case (category, location,
                 size, morphology, lymph nodes, histology, impression, next step)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from medfusion.data.generator import LOBE_NAMES, CohortGenerator, data_dictionary
from medfusion.data.knowledge import write_knowledge_assets
from medfusion.data.renderer import nodule_heatmap, render_ct, render_wsi
from medfusion.data.tokenizer import ClinicalTokenizer
from medfusion.models.components import HISTOLOGY_CLASSES, LUNG_RADS_CLASSES, MARGIN_CLASSES, NODULE_TYPE_CLASSES
from medfusion.rag.graph import KnowledgeGraph
from medfusion.rag.retriever import HybridRetriever, build_query
from medfusion.utils import PROJECT_ROOT, get_logger, sub

LOGGER = get_logger("dataset")
REPORT_PROMPT = "generate the integrated radiology pathology report"
QA_TEMPLATES = {
    "category": "what is the lung rads category?",
    "location": "where is the dominant nodule located?",
    "size": "what is the size of the dominant nodule?",
    "morphology": "describe the dominant nodule morphology.",
    "lymph": "is there mediastinal lymphadenopathy?",
    "histology": "what does the histology show?",
    "impression": "summarise the impression.",
    "next_step": "what is the recommended next step?",
}


def qa_answer(row: dict, kind: str):
    present = int(row["nodule_present"])
    if kind == "category":
        return f"Lung RADS {row['lung_rads_category']}."
    if kind == "location":
        return f"The {LOBE_NAMES[row['dominant_nodule_lobe']]}." if present else "No pulmonary nodule is identified."
    if kind == "size":
        return f"{int(round(row['nodule_diameter_mm']))} mm." if present else "No pulmonary nodule is identified."
    if kind == "morphology":
        if not present:
            return "No pulmonary nodule is identified."
        if row["nodule_type"] == "calcified":
            return f"Calcified nodule with {row['calcification_pattern']} calcification."
        return f"{row['nodule_type'].capitalize()} nodule with {row['margin']} margins."
    if kind == "lymph":
        return "Yes, enlarged mediastinal lymph nodes are present." if int(row["mediastinal_lymphadenopathy"]) \
            else "No mediastinal or hilar lymphadenopathy."
    if kind == "histology":
        return row["pathology_text"]
    if kind == "impression":
        return row["impression_text"]
    return row["recommendation_text"]


def paths(cfg: dict):
    root = PROJECT_ROOT
    p = cfg["paths"]
    return {
        "data_dir": root / p["data_dir"], "dataset": root / p["dataset_csv"], "dictionary": root / p["dictionary_csv"],
        "kb": root / p["data_dir"] / "knowledge_base" / "clinical_guidance_passages.jsonl",
        "kg": root / p["data_dir"] / "knowledge_graph", "output": root / p["output_dir"],
        "cache": root / p["cache_dir"], "tokenizer": root / p["output_dir"] / "artifacts" / "tokenizer.json",
    }


def build_dataset(cfg: dict, force: bool = False):
    """Generate the synthetic cohort and knowledge assets if absent."""
    p = paths(cfg)
    stats = write_knowledge_assets(p["data_dir"])
    LOGGER.info("Knowledge assets ready: %s", stats)
    if p["dataset"].exists() and not force:
        return pd.read_csv(p["dataset"], keep_default_na=False, na_values=[""])
    frame = CohortGenerator(cfg["data"]["n_rows"], cfg["data"]["generator_seed"]).generate()
    frame.to_csv(p["dataset"], index=False)
    data_dictionary().to_csv(p["dictionary"], index=False)
    LOGGER.info("Dataset written to %s with shape %s", p["dataset"], frame.shape)
    return pd.read_csv(p["dataset"], keep_default_na=False, na_values=[""])


def build_tokenizer(cfg: dict, frame: pd.DataFrame, retriever: HybridRetriever):
    p = paths(cfg)
    if p["tokenizer"].exists():
        return ClinicalTokenizer.load(p["tokenizer"])
    train = frame[frame["split"] == "train"]
    texts = list(train["clinical_note"]) + list(train["integrated_report"]) + list(REPORT_PROMPT.split("\n"))
    texts += list(QA_TEMPLATES.values()) + [r["text"] for r in retriever.records]
    texts += [qa_answer(r, k) for r in train.head(3000).to_dict("records") for k in QA_TEMPLATES]
    tokenizer = ClinicalTokenizer().fit(texts, min_freq=1)
    tokenizer.save(p["tokenizer"])
    LOGGER.info("Tokenizer fitted with %d tokens", len(tokenizer))
    return tokenizer


def build_image_cache(cfg: dict, frame: pd.DataFrame):
    """Render every key image and histology tile once into uint8 arrays."""
    p = paths(cfg)
    p["cache"].mkdir(parents=True, exist_ok=True)
    ct_path, wsi_path = p["cache"] / "ct_uint8.npy", p["cache"] / "wsi_uint8.npy"
    if ct_path.exists() and wsi_path.exists():
        ct, wsi = np.load(ct_path, mmap_mode="r"), np.load(wsi_path, mmap_mode="r")
        if len(ct) == len(frame):
            return ct, wsi
    n, size, wsize = len(frame), cfg["data"]["ct_size"], cfg["data"]["wsi_size"]
    ct = np.zeros((n, size, size), dtype=np.uint8)
    wsi = np.zeros((n, 3, wsize, wsize), dtype=np.uint8)
    for i, row in enumerate(frame.to_dict("records")):
        ct[i] = np.round(render_ct(row) * 255).astype(np.uint8)
        if int(row["histology_available"]):
            wsi[i] = np.round(render_wsi(row) * 255).astype(np.uint8)
        if (i + 1) % 2000 == 0:
            LOGGER.info("Rendered %d of %d studies", i + 1, n)
    np.save(ct_path, ct)
    np.save(wsi_path, wsi)
    return ct, wsi


def build_retriever(cfg: dict):
    p = paths(cfg)
    graph = KnowledgeGraph.load(p["kg"])
    r = cfg["retrieval"]
    return HybridRetriever.from_jsonl(p["kb"], graph=graph, weights=r["weights"], rrf_k=r["rrf_k"])


def perturb_findings(row: dict, rng: np.random.Generator):
    """Simulate imperfect upstream predictions so the decoder learns to use,
    but not blindly trust, retrieved guidance (retrieval noise injection)."""
    noisy = dict(row)
    cats = LUNG_RADS_CLASSES
    idx = cats.index(str(row["lung_rads_category"]))
    step = 1 if rng.random() < 0.5 else ~0
    noisy["lung_rads_category"] = cats[int(np.clip(idx + step, 0, len(cats) + ~0))]
    if rng.random() < 0.3:
        noisy["histology_subtype"] = "not sampled"
    return noisy


def assign_passages(cfg: dict, frame: pd.DataFrame, retriever: HybridRetriever, noise: float, seed: int):
    """Retrieve top k passages for every case from (optionally noisy) findings."""
    rng = np.random.default_rng(seed)
    k = cfg["retrieval"]["top_k"]
    out = np.zeros((len(frame), k), dtype=np.int64)
    for i, row in enumerate(frame.to_dict("records")):
        findings = perturb_findings(row, rng) if rng.random() < noise else row
        query, tags = build_query(findings)
        hits = retriever.search(query, tags, k=k, management_slots=cfg["retrieval"]["management_slots"])
        out[i] = [retriever.index[h.doc_id] for h in hits]
    return out


def passage_bank(cfg: dict, tokenizer: ClinicalTokenizer, retriever: HybridRetriever):
    length = cfg["data"]["max_passage_len"]
    rows = [tokenizer.pad(tokenizer.encode(r["text"], max_len=length, add_cls=True), length) for r in retriever.records]
    return torch.tensor(rows, dtype=torch.long)


def encode_labels(row: dict):
    present = int(row["nodule_present"])
    diameter = float(row["nodule_diameter_mm"]) if present else 1.0
    subtype = row["histology_subtype"]
    return {
        "presence": float(present), "log_diameter": math.log(max(diameter, 1.0)),
        "nodule_type": NODULE_TYPE_CLASSES.index(row["nodule_type"]), "margin": MARGIN_CLASSES.index(row["margin"]),
        "lung_rads": LUNG_RADS_CLASSES.index(str(row["lung_rads_category"])),
        "malignancy": float(row["malignancy_label"]), "lymph": float(row["mediastinal_lymphadenopathy"]),
        "histology": HISTOLOGY_CLASSES.index(subtype) if subtype in HISTOLOGY_CLASSES else 0,
        "histology_mask": float(subtype in HISTOLOGY_CLASSES and int(row["histology_available"])),
    }


class MedFusionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, row_index: np.ndarray, ct: np.ndarray, wsi: np.ndarray,
                 tokenizer: ClinicalTokenizer, passages: np.ndarray, cfg: dict, train: bool,
                 task: str = "mixed", qa_kind: str | None = None, seed: int = 0):
        self.records = frame.to_dict("records")
        self.row_index = row_index
        self.ct, self.wsi = ct, wsi
        self.tok = tokenizer
        self.passages = passages
        self.cfg = cfg
        self.train = train
        self.task = task
        self.qa_kind = qa_kind
        self.rng = np.random.default_rng(seed)
        self.report_prob = cfg["training"]["report_task_prob"]

    def __len__(self):
        return len(self.records)

    def _task(self, row: dict):
        if self.task == "report" or (self.task == "mixed" and self.rng.random() < self.report_prob):
            return "[TASK_REPORT]", REPORT_PROMPT, row["integrated_report"]
        kind = self.qa_kind or str(self.rng.choice(list(QA_TEMPLATES)))
        return "[TASK_QA]", QA_TEMPLATES[kind], qa_answer(row, kind)

    def prompt_ids(self, task_token: str, prompt: str):
        return [self.tok.bos_id, self.tok.token_to_id[task_token]] + self.tok.encode(prompt) + [self.tok.sep_id]

    def __getitem__(self, i: int):
        row = self.records[i]
        ridx = int(self.row_index[i])
        ct = torch.from_numpy(np.asarray(self.ct[ridx], dtype=np.float32) / 255.0)[None]
        if self.train:
            ct = ct * float(self.rng.uniform(0.93, 1.07)) + float(self.rng.normal(0, 0.015))
        present = bool(int(row["histology_available"]))
        wsi = torch.from_numpy(np.asarray(self.wsi[ridx], dtype=np.float32) / 255.0)
        task_token, prompt, answer = self._task(row)
        prompt_ids = self.prompt_ids(task_token, prompt)
        target_ids = self.tok.encode(answer) + [self.tok.eos_id]
        full = (prompt_ids + target_ids)[: self.cfg["data"]["max_target_len"] + 1]
        decoder_input = full[:~0]
        labels = [self.tok.pad_id] * sub(len(prompt_ids), 1) + full[len(prompt_ids):]
        note = self.tok.encode(row["clinical_note"], max_len=self.cfg["data"]["max_note_len"], add_cls=True)
        findings = self.tok.encode(row["findings_text"], max_len=self.cfg["data"]["max_findings_len"], add_cls=True)
        return {
            "ct": ct, "wsi": wsi, "wsi_present": present, "note_ids": note, "findings_ids": findings,
            "decoder_input": decoder_input, "labels": labels[: len(decoder_input)],
            "prompt_len": len(prompt_ids), "heatmap": torch.from_numpy(nodule_heatmap(row, self.cfg["data"]["ct_size"] // 4)),
            "passage_idx": torch.from_numpy(self.passages[ridx]), "targets": encode_labels(row), "case_id": row["case_id"],
        }


def collate(batch: list[dict], pad_id: int):
    def pad_seq(key):
        length = max(len(b[key]) for b in batch)
        return torch.tensor([b[key] + [pad_id] * sub(length, len(b[key])) for b in batch], dtype=torch.long)

    out = {
        "ct": torch.stack([b["ct"] for b in batch]), "wsi": torch.stack([b["wsi"] for b in batch]),
        "wsi_present": torch.tensor([b["wsi_present"] for b in batch], dtype=torch.bool),
        "note_ids": pad_seq("note_ids"), "findings_ids": pad_seq("findings_ids"),
        "decoder_input": pad_seq("decoder_input"), "labels": pad_seq("labels"),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "passage_idx": torch.stack([b["passage_idx"] for b in batch]),
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch]), "case_id": [b["case_id"] for b in batch],
    }
    for key in batch[0]["targets"]:
        dtype = torch.long if key in ("nodule_type", "margin", "lung_rads", "histology") else torch.float32
        out[key] = torch.tensor([b["targets"][key] for b in batch], dtype=dtype)
    return out
