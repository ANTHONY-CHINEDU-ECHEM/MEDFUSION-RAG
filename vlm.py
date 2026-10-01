"""MedFusion VLM: a retrieval augmented multimodal clinical language model.

Data flow
    CT key image  → Hierarchical MedViT (CT)  → pyramid tokens + fine map
    WSI tile      → Hierarchical MedViT (WSI) → pyramid tokens
    Clinical note → Text encoder (BERT style) → note tokens
    Gated bidirectional cross attention + joint multimodal blocks → fused memory
    Fused memory  → clinical heads (localisation, risk, category, histology)
    Guidance bank → Text encoder (passage segment) → retrieved passage memory
    [fused memory | retrieved passages] → autoregressive decoder → report / answer
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from medfusion.models.components import ClinicalHeads, MultimodalFusion, RetrievalAugmentedDecoder, TextEncoder
from medfusion.models.image_encoder import HierarchicalMedViT


class MedFusionVLM(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int, pad_id: int, passage_bank: torch.Tensor):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        self.cfg = cfg
        self.pad_id = pad_id
        self.ct_encoder = HierarchicalMedViT(1, cfg["data"]["ct_size"], tuple(m["ct_channels"]), d, m["n_heads"], m["dropout"])
        self.wsi_encoder = HierarchicalMedViT(3, cfg["data"]["wsi_size"], tuple(m["wsi_channels"]), d, m["n_heads"], m["dropout"])
        max_text = max(cfg["data"]["max_note_len"], cfg["data"]["max_passage_len"], cfg["data"]["max_findings_len"])
        self.text_encoder = TextEncoder(vocab_size, d, m["n_heads"], m["text_layers"], max_text, m["dropout"], pad_id)
        self.fusion = MultimodalFusion(d, m["n_heads"], m["fusion_layers"], m["dropout"])
        self.heads = ClinicalHeads(d, m["dropout"])
        self.decoder = RetrievalAugmentedDecoder(vocab_size, d, m["n_heads"], m["decoder_layers"],
                                                 cfg["data"]["max_target_len"], m["dropout"], pad_id)
        self.rank_embed = nn.Embedding(8, d)
        self.memory_type = nn.Embedding(2, d)
        for table in (self.rank_embed, self.memory_type, self.fusion.type_embed):
            nn.init.normal_(table.weight, std=0.02)
        self.image_proj = nn.Linear(d, d)
        self.text_proj = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(2.65))
        self.register_buffer("passage_bank", passage_bank.long(), persistent=False)

    # Encoding
    def encode(self, batch: dict, modality_dropout: float = 0.0):
        ct = self.ct_encoder(batch["ct"])
        wsi = self.wsi_encoder(batch["wsi"])
        wsi_present = batch["wsi_present"].clone()
        note_ids = batch["note_ids"]
        if self.training and modality_dropout > 0:
            drop_wsi = torch.rand_like(wsi_present.float()) < modality_dropout
            wsi_present = wsi_present & ~drop_wsi
            drop_note = (torch.rand(note_ids.shape[0], device=note_ids.device) < modality_dropout)[:, None]
            keep_first = torch.zeros_like(note_ids, dtype=torch.bool)
            keep_first[:, 0] = True
            note_ids = torch.where(drop_note & ~keep_first, torch.full_like(note_ids, self.pad_id), note_ids)
        note_tokens, note_pad = self.text_encoder(note_ids, segment=0)
        fused, fused_pad, layout, t2i = self.fusion(ct["tokens"], wsi["tokens"], wsi_present, note_tokens, note_pad)
        heads = self.heads(fused, fused_pad, layout, ct["fine_map"])
        return {"fused": fused, "fused_pad": fused_pad, "layout": layout, "heads": heads, "ct": ct,
                "wsi_present": wsi_present, "text_to_image_attention": t2i}

    def passage_memory(self, passage_idx: torch.Tensor, passage_keep: torch.Tensor | None = None):
        """Encode the guidance bank and gather the retrieved passages per case.

        passage_idx: (B, K) indices into the bank. passage_keep: (B, K) bool,
        False removes a passage (passage dropout or ablation without retrieval).
        """
        bank_tokens, bank_pad = self.text_encoder(self.passage_bank, segment=1)
        b, k = passage_idx.shape
        tokens = bank_tokens[passage_idx]
        pad = bank_pad[passage_idx].clone()
        if passage_keep is not None:
            pad = pad | ~passage_keep[:, :, None]
        ranks = self.rank_embed(torch.arange(k, device=passage_idx.device))
        tokens = tokens + ranks[None, :, None, :] + self.memory_type.weight[1]
        return tokens.reshape(b, k * tokens.shape[2], ~0), pad.reshape(b, ~0)

    def build_memory(self, enc: dict, passage_idx, passage_keep):
        fused = enc["fused"] + self.memory_type.weight[0]
        if passage_idx is None:
            return fused, enc["fused_pad"], fused.shape[1]
        p_tokens, p_pad = self.passage_memory(passage_idx, passage_keep)
        return torch.cat([fused, p_tokens], 1), torch.cat([enc["fused_pad"], p_pad], 1), fused.shape[1]

    # Training forward
    def forward(self, batch: dict, modality_dropout: float = 0.0, passage_dropout: float = 0.0):
        enc = self.encode(batch, modality_dropout)
        keep = batch.get("passage_keep")
        if keep is None:
            keep = torch.ones_like(batch["passage_idx"], dtype=torch.bool)
        if self.training and passage_dropout > 0:
            keep = keep & (torch.rand(keep.shape, device=keep.device) >= passage_dropout)
        memory, memory_pad, _ = self.build_memory(enc, batch["passage_idx"], keep)
        logits, _ = self.decoder(batch["decoder_input"], memory, memory_pad)
        out = {"lm_logits": logits, **enc["heads"]}
        if "findings_ids" in batch:
            findings_tokens, findings_pad = self.text_encoder(batch["findings_ids"], segment=2)
            out["image_embed"] = F.normalize(self.image_proj(enc["ct"]["pooled"]), dim=~0)
            out["text_embed"] = F.normalize(self.text_proj(findings_tokens[:, 0]), dim=~0)
            out["logit_scale"] = self.logit_scale.exp().clamp(max=50.0)
        return out

    # Generation
    @torch.no_grad()
    def generate(self, enc: dict, prompt_ids: torch.Tensor, passage_idx, passage_keep, max_new_tokens: int,
                 eos_id: int):
        """Greedy, key value cached decoding for a batch of equal length prompts."""
        memory, memory_pad, n_fused = self.build_memory(enc, passage_idx, passage_keep)
        ids, attention = self.decoder.generate(prompt_ids, memory, memory_pad, max_new_tokens, eos_id)
        return {"ids": ids, "cross_attention": attention, "n_fused": n_fused}
