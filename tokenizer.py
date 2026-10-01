"""Clinical word level tokenizer with digit decomposition and truecasing.

Design choices:
* Lower cased word tokens keep the vocabulary compact for a small model.
* Numbers are decomposed into single digits so measurements generalise to
  values never seen during training (12 mm becomes 1 2 mm).
* Detokenisation restores clinical acronyms (CT, PET, MDT, EGFR, Lung RADS)
  and sentence case so generated reports read naturally.
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path

from medfusion.utils import sub

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]", "[DOC]",
                  "[TASK_REPORT]", "[TASK_QA]"]
LETTERS = string.ascii_lowercase
COMPOUND = (r"(?:ki67|pdl1|g12c|t1mi|t[1234][abc]?|n[0123]|m[01][abc]?|ia[123]|4[abx])"
            r"(?![" + LETTERS + r"\d])")
TOKEN_PATTERN = re.compile(COMPOUND + r"|\d|[" + LETTERS + r"]+|[^\s" + LETTERS + r"\d]")
ACRONYMS = {
    "ct": "CT", "pet": "PET", "mri": "MRI", "mdt": "MDT", "rads": "RADS", "egfr": "EGFR", "alk": "ALK",
    "kras": "KRAS", "pdl1": "PDL1", "tps": "TPS", "copd": "COPD", "ecog": "ECOG", "ebus": "EBUS",
    "ki67": "Ki67", "pmh": "PMH", "fhx": "FHx", "sob": "SOB", "nsclc": "NSCLC", "g12c": "G12C",
    "findings": "FINDINGS", "pathology": "PATHOLOGY", "impression": "IMPRESSION",
    "recommendation": "RECOMMENDATION", "lung": "lung",
}
STAGE_TOKENS = {"ia": "IA", "ib": "IB", "iia": "IIA", "iib": "IIB", "iiia": "IIIA", "iv": "IV", "tis": "Tis"}


class ClinicalTokenizer:
    def __init__(self, vocab: list[str] | None = None):
        self.vocab = vocab or list(SPECIAL_TOKENS)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

    # Special token identifiers
    @property
    def pad_id(self):
        return self.token_to_id["[PAD]"]

    @property
    def unk_id(self):
        return self.token_to_id["[UNK]"]

    @property
    def bos_id(self):
        return self.token_to_id["[BOS]"]

    @property
    def eos_id(self):
        return self.token_to_id["[EOS]"]

    @property
    def sep_id(self):
        return self.token_to_id["[SEP]"]

    @property
    def cls_id(self):
        return self.token_to_id["[CLS]"]

    def __len__(self):
        return len(self.vocab)

    # Core API
    @staticmethod
    def split(text: str):
        return TOKEN_PATTERN.findall(text.lower())

    def fit(self, texts, min_freq: int = 2):
        counter = Counter()
        for text in texts:
            counter.update(self.split(text))
        words = sorted(w for w, c in counter.items() if c >= min_freq and w not in self.token_to_id)
        self.vocab = list(SPECIAL_TOKENS) + [str(d) for d in range(10) if str(d) not in words] + words
        self.vocab = list(dict.fromkeys(self.vocab))
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        return self

    def encode(self, text: str, max_len: int | None = None, add_bos: bool = False, add_eos: bool = False,
               add_cls: bool = False):
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self.split(text)]
        if add_cls:
            ids = [self.cls_id] + ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            if add_eos:
                ids[~0] = self.eos_id
        return ids

    def pad(self, ids: list[int], length: int):
        return ids[:length] + [self.pad_id] * max(0, sub(length, len(ids)))

    def decode_tokens(self, ids):
        tokens = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id, self.cls_id) or i >= len(self.vocab):
                continue
            tokens.append(self.vocab[i])
        return tokens

    def decode(self, ids):
        return detokenize(self.decode_tokens(ids))

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"vocab": self.vocab}, indent=0), encoding="utf8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf8"))["vocab"])


def detokenize(tokens: list[str]):
    """Join tokens into readable, truecased clinical text."""
    out: list[str] = []
    for tok in tokens:
        if tok.startswith("[") and tok.endswith("]"):
            continue
        if out and tok.isdigit() and out[~0].isdigit():
            out[~0] += tok
            continue
        if tok in {".", ",", ";", ":", "%", ")"} and out:
            out[~0] += tok
            continue
        if out and out[~0] == "(":
            out[~0] += tok
            continue
        out.append(tok)
    words = []
    capitalise_next = True
    for word in out:
        core = word.rstrip(".,;:%)")
        tail = word[len(core):]
        lead = ""
        if core.startswith("("):
            lead, core = "(", core[1:]
        if core in ACRONYMS:
            core = ACRONYMS[core]
        elif core in STAGE_TOKENS or re.fullmatch(r"ia[123]", core):
            core = STAGE_TOKENS.get(core, core.upper())
        elif re.fullmatch(r"t[12][abc]?|t1mi|n[0123]|m[01][abc]?|4[abx]", core):
            core = core.upper() if core.startswith("4") else core[0].upper() + core[1:]
        elif re.fullmatch(r"[rl][ul]l|rml", core):
            core = core.upper()
        if capitalise_next and core and core[0].isalpha():
            core = core[0].upper() + core[1:]
        words.append(lead + core + tail)
        capitalise_next = tail.endswith(".") or core in ("FINDINGS", "PATHOLOGY", "IMPRESSION", "RECOMMENDATION")
    return " ".join(words)
