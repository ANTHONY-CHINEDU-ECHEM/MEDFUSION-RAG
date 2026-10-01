"""Transformer components: clinical text encoder, gated bidirectional cross
attention fusion, multitask clinical heads and the retrieval augmented
generative decoder (the language model of the system)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, pad_mask: torch.Tensor):
    """Mean over the sequence axis ignoring padded positions (pad_mask True)."""
    keep = (~pad_mask).float().unsqueeze(~0)
    return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)


class TextEncoder(nn.Module):
    """BERT style bidirectional encoder for clinical notes and guidance passages."""

    def __init__(self, vocab_size: int, d_model: int, heads: int, layers: int, max_len: int,
                 dropout: float, pad_id: int, segments: int = 3):
        super().__init__()
        self.pad_id = pad_id
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_len, d_model)
        for table in (self.token, self.position):
            nn.init.normal_(table.weight, std=0.02)
        self.segment = nn.Embedding(segments, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, segment: int = 0):
        pad_mask = ids.eq(self.pad_id)
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)[None] + self.segment.weight[segment]
        x = self.encoder(self.dropout(x), src_key_padding_mask=pad_mask)
        return self.norm(x), pad_mask


class GatedCrossAttention(nn.Module):
    """Cross attention with learnable tanh gates (Flamingo style residual gating).

    The gates let the network learn how much cross modal context to inject,
    which stabilises training when one modality is missing or uninformative.
    """

    def __init__(self, d_model: int, heads: int, dropout: float, gate_init: float = 0.25):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * 4, d_model))
        self.gate_attn = nn.Parameter(torch.tensor(gate_init))
        self.gate_ffn = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x, context, context_pad_mask):
        attended, weights = self.attn(self.norm_q(x), self.norm_kv(context), self.norm_kv(context),
                                      key_padding_mask=context_pad_mask, need_weights=True)
        x = x + torch.tanh(self.gate_attn) * attended
        x = x + torch.tanh(self.gate_ffn) * self.ffn(self.norm_ff(x))
        return x, weights


class MultimodalFusion(nn.Module):
    """Bidirectional gated cross attention followed by joint multimodal blocks.

    Step 1: clinical text queries attend to visual tokens (text grounded in image).
    Step 2: visual tokens attend to clinical text (image conditioned on context).
    Step 3: a joint transformer over [FUSION CLS | CT | WSI | NOTE] tokens with
            modality type embeddings produces the fused multimodal memory.
    """

    TYPE_FUSION, TYPE_CT, TYPE_WSI, TYPE_NOTE = 0, 1, 2, 3

    def __init__(self, d_model: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.text_to_image = GatedCrossAttention(d_model, heads, dropout)
        self.image_to_text = GatedCrossAttention(d_model, heads, dropout)
        self.type_embed = nn.Embedding(4, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.joint = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ct_tokens, wsi_tokens, wsi_present, note_tokens, note_pad_mask):
        b, n_ct, _ = ct_tokens.shape
        n_wsi = wsi_tokens.shape[1]
        device = ct_tokens.device
        visual = torch.cat([ct_tokens, wsi_tokens], dim=1)
        visual_pad = torch.cat([torch.zeros(b, n_ct, dtype=torch.bool, device=device),
                                (~wsi_present).unsqueeze(1).expand(b, n_wsi)], dim=1)
        note_fused, text_to_image_attn = self.text_to_image(note_tokens, visual, visual_pad)
        visual_fused, _ = self.image_to_text(visual, note_tokens, note_pad_mask)
        types = torch.cat([
            torch.full((n_ct,), self.TYPE_CT, device=device),
            torch.full((n_wsi,), self.TYPE_WSI, device=device)])
        visual_fused = visual_fused + self.type_embed(types)[None]
        note_fused = note_fused + self.type_embed.weight[self.TYPE_NOTE]
        cls = self.cls.expand(b, 1, ~0) + self.type_embed.weight[self.TYPE_FUSION]
        sequence = torch.cat([cls, visual_fused, note_fused], dim=1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=device), visual_pad, note_pad_mask], dim=1)
        fused = self.norm(self.joint(sequence, src_key_padding_mask=pad))
        layout = {"cls": (0, 1), "ct": (1, 1 + n_ct), "wsi": (1 + n_ct, 1 + n_ct + n_wsi),
                  "note": (1 + n_ct + n_wsi, sequence.shape[1])}
        return fused, pad, layout, text_to_image_attn


HISTOLOGY_CLASSES = ["granuloma", "hamartoma", "organising pneumonia", "nondiagnostic sample", "adenocarcinoma",
                     "squamous cell carcinoma", "small cell carcinoma", "large cell carcinoma", "carcinoid tumour"]
NODULE_TYPE_CLASSES = ["none", "solid", "part solid", "ground glass", "calcified"]
MARGIN_CLASSES = ["none", "smooth", "lobulated", "spiculated"]
LUNG_RADS_CLASSES = ["1", "2", "3", "4A", "4B", "4X"]


class ClinicalHeads(nn.Module):
    """Multitask prediction heads that also drive retrieval query construction."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_model * 3, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_model * 2, d_model), nn.GELU())
        self.presence = nn.Linear(d_model, 1)
        self.diameter = nn.Linear(d_model, 1)
        self.nodule_type = nn.Linear(d_model, len(NODULE_TYPE_CLASSES))
        self.margin = nn.Linear(d_model, len(MARGIN_CLASSES))
        self.lung_rads = nn.Linear(d_model, len(LUNG_RADS_CLASSES))
        self.malignancy = nn.Linear(d_model, 1)
        self.lymph = nn.Linear(d_model, 1)
        self.histology = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, len(HISTOLOGY_CLASSES)))
        # FiLM conditioned dense localisation head on the finest CT pyramid level
        self.film = nn.Linear(d_model, d_model * 2)
        self.loc = nn.Sequential(nn.Conv2d(d_model, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.loc[~0].bias, float(torch.log(torch.tensor(0.01 / 0.99))))

    def forward(self, fused, fused_pad, layout, ct_fine_map):
        cls = fused[:, 0]
        a, b = layout["ct"]
        ct_mean = fused[:, a:b].mean(1)
        c, d = layout["wsi"]
        wsi_valid = ~fused_pad[:, c:d]
        wsi_mean = (fused[:, c:d] * wsi_valid.unsqueeze(~0)).sum(1) / wsi_valid.sum(1, keepdim=True).clamp(min=1)
        h = self.trunk(torch.cat([cls, ct_mean, wsi_mean], dim=~0))
        gamma, beta = self.film(cls).chunk(2, dim=~0)
        loc_input = ct_fine_map * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return {
            "presence": self.presence(h).squeeze(~0), "log_diameter": self.diameter(h).squeeze(~0),
            "nodule_type": self.nodule_type(h), "margin": self.margin(h), "lung_rads": self.lung_rads(h),
            "malignancy": self.malignancy(h).squeeze(~0), "lymph": self.lymph(h).squeeze(~0),
            "histology": self.histology(torch.cat([h, wsi_mean], dim=~0)),
            "heatmap": self.loc(loc_input).squeeze(1),
        }


def _project(mha: nn.MultiheadAttention, x: torch.Tensor, which: int):
    """Apply the query (0), key (1) or value (2) input projection of an attention module."""
    weight = mha.in_proj_weight.chunk(3)[which]
    bias = mha.in_proj_bias.chunk(3)[which]
    return F.linear(x, weight, bias)


def _split_heads(x: torch.Tensor, heads: int):
    b, n, d = x.shape
    return x.reshape(b, n, heads, d // heads).transpose(1, 2)


def _attend(mha: nn.MultiheadAttention, q, k, v, key_pad=None):
    """Scaled dot product attention using cached keys and values."""
    scores = (q @ k.transpose(~1, ~0)) / (q.shape[~0] ** 0.5)
    if key_pad is not None:
        scores = scores.masked_fill(key_pad[:, None, None, :], torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=~0)
    out = (weights @ v).transpose(1, 2).flatten(2)
    return mha.out_proj(out), weights.mean(1)


class DecoderLayer(nn.Module):
    """Pre norm decoder layer: causal self attention, cross attention, feed forward."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * 4, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, self_pad, memory, memory_pad, need_weights: bool = False):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, attn_mask=causal_mask, key_padding_mask=self_pad,
                                            need_weights=False)[0])
        h = self.norm2(x)
        attended, weights = self.cross_attn(h, memory, memory, key_padding_mask=memory_pad, need_weights=need_weights)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x, weights

    def init_cache(self, memory, memory_pad):
        heads = self.cross_attn.num_heads
        return {"self_k": None, "self_v": None, "memory_pad": memory_pad,
                "cross_k": _split_heads(_project(self.cross_attn, memory, 1), heads),
                "cross_v": _split_heads(_project(self.cross_attn, memory, 2), heads)}

    def step(self, x, cache: dict):
        """Process one new position using cached keys and values (inference only)."""
        heads = self.self_attn.num_heads
        h = self.norm1(x)
        q = _split_heads(_project(self.self_attn, h, 0), heads)
        k = _split_heads(_project(self.self_attn, h, 1), heads)
        v = _split_heads(_project(self.self_attn, h, 2), heads)
        cache["self_k"] = k if cache["self_k"] is None else torch.cat([cache["self_k"], k], 2)
        cache["self_v"] = v if cache["self_v"] is None else torch.cat([cache["self_v"], v], 2)
        attended, _ = _attend(self.self_attn, q, cache["self_k"], cache["self_v"])
        x = x + attended
        h = self.norm2(x)
        q = _split_heads(_project(self.cross_attn, h, 0), heads)
        attended, weights = _attend(self.cross_attn, q, cache["cross_k"], cache["cross_v"], cache["memory_pad"])
        x = x + attended
        x = x + self.ffn(self.norm3(x))
        return x, weights[:, 0]


class RetrievalAugmentedDecoder(nn.Module):
    """Autoregressive clinical language model conditioned on fused multimodal
    memory and retrieved guidance passages (retrieval augmented generation)."""

    def __init__(self, vocab_size: int, d_model: int, heads: int, layers: int, max_len: int, dropout: float, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        for table in (self.token, self.position):
            nn.init.normal_(table.weight, std=0.02)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, memory, memory_pad, need_weights: bool = False):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.dropout(self.token(ids) + self.position(positions)[None])
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1)
        self_pad = ids.eq(self.pad_id)
        weights = None
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) + ~0
            x, w = layer(x, causal, self_pad, memory, memory_pad, need_weights and last)
            if w is not None:
                weights = w
        return self.lm_head(self.norm(x)), weights

    @torch.no_grad()
    def generate(self, prompt_ids, memory, memory_pad, max_new_tokens: int, eos_id: int):
        """Greedy decoding with key value caching.

        Every layer caches the projected keys and values of past positions and
        of the multimodal memory, so each new token costs one position of
        compute instead of a full sequence pass. Returns generated ids and the
        last layer cross attention over memory for every generated token.
        """
        caches = [layer.init_cache(memory, memory_pad) for layer in self.layers]
        batch = prompt_ids.shape[0]
        finished = torch.zeros(batch, dtype=torch.bool, device=prompt_ids.device)
        generated, attention = [], []
        limit = self.position.num_embeddings
        position = 0
        token = None
        steps = prompt_ids.shape[1] + max_new_tokens + ~0
        for position in range(min(steps, limit)):
            if position < prompt_ids.shape[1]:
                token = prompt_ids[:, position]
            x = self.token(token)[:, None] + self.position.weight[position]
            weights = None
            for layer, cache in zip(self.layers, caches):
                x, weights = layer.step(x, cache)
            if position + 1 < prompt_ids.shape[1]:
                continue
            next_token = self.lm_head(self.norm(x))[:, 0].argmax(~0)
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_id), next_token)
            generated.append(next_token)
            attention.append(weights)
            finished |= next_token.eq(eos_id)
            token = next_token
            if bool(finished.all()):
                break
        return torch.stack(generated, 1), torch.stack(attention, 1)
