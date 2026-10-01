"""Hybrid retrieval over the clinical guidance knowledge base.

Three complementary rankers are fused with reciprocal rank fusion (RRF):

1. Lexical: Okapi BM25 implemented from first principles.
2. Semantic: latent semantic analysis (TF IDF followed by truncated SVD) with
   cosine similarity, a lightweight dense retriever that needs no internet.
3. Graph: overlap between knowledge graph concepts detected in the case and
   the concepts each passage is tagged with, expanded one hop through the
   biomedical knowledge graph.

RRF is robust to the very different score scales of the three rankers and is
a widely used fusion method for hybrid search.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from medfusion.data.tokenizer import ClinicalTokenizer
from medfusion.utils import sub

STOPWORDS = set("a an and are as at be by for from has have in is it its of on or that the this to with which "
                "were was will their they there these than then also into such may should can".split())


def analyse(text: str):
    return [t for t in ClinicalTokenizer.split(text) if t.isalnum() and t not in STOPWORDS]


@dataclass
class RetrievalHit:
    doc_id: str
    score: float
    rank: int
    components: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, records: list[dict], graph=None, k1: float = 1.4, b: float = 0.7,
                 svd_dim: int = 32, rrf_k: int = 20, weights: dict | None = None):
        self.records = records
        self.doc_ids = [r["doc_id"] for r in records]
        self.index = {d: i for i, d in enumerate(self.doc_ids)}
        self.graph = graph
        self.k1, self.b, self.rrf_k = k1, b, rrf_k
        self.weights = weights or {"bm25": 1.0, "dense": 1.0, "graph": 1.3}
        self._fit_bm25()
        self._fit_dense(svd_dim)
        self.tags = [set(r["kg_tags"]) for r in records]

    @classmethod
    def from_jsonl(cls, path: str | Path, graph=None, **kwargs):
        with open(path, encoding="utf8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return cls(records, graph=graph, **kwargs)

    # Lexical ranker
    def _fit_bm25(self):
        self.doc_tokens = [analyse(r["text"]) for r in self.records]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean())
        self.term_freqs = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(self.records)
        self.idf = {term: math.log(1.0 + sub(n, freq) / (freq + 0.5) + 0.5 / (freq + 0.5)) for term, freq in df.items()}

    def bm25_scores(self, query: str):
        scores = np.zeros(len(self.records), dtype=np.float32)
        for term in set(analyse(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (sub(1.0, self.b) + self.b * self.doc_len[i] / self.avg_len)
                    scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores

    # Semantic ranker
    def _fit_dense(self, svd_dim: int):
        self.vectorizer = TfidfVectorizer(analyzer=analyse, sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform([r["text"] for r in self.records])
        dim = min(svd_dim, sub(min(tfidf.shape), 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        emb = self.svd.fit_transform(tfidf)
        self.doc_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1.0 / 1e9)

    def dense_scores(self, query: str):
        q = self.svd.transform(self.vectorizer.transform([query]))[0]
        q = q / (np.linalg.norm(q) + 1.0 / 1e9)
        return self.doc_emb @ q

    # Graph ranker
    def graph_scores(self, query_tags: set[str]):
        if not query_tags:
            return np.zeros(len(self.records), dtype=np.float32)
        expanded = {t: 1.0 for t in query_tags}
        if self.graph is not None:
            for tag in query_tags:
                for neighbour, weight in self.graph.neighbours(tag):
                    expanded[neighbour] = max(expanded.get(neighbour, 0.0), 0.35 * weight)
        scores = np.zeros(len(self.records), dtype=np.float32)
        for i, tags in enumerate(self.tags):
            overlap = sum(expanded.get(t, 0.0) for t in tags)
            scores[i] = overlap / math.sqrt(len(tags) + 1.0)
        return scores

    @staticmethod
    def _ranks(scores: np.ndarray):
        order = np.argsort(np.negative(scores), kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        return ranks

    def search(self, query: str, query_tags: set[str] | None = None, k: int = 3,
               management_slots: int = 1):
        """Return k passages: the best management passages first, then evidence.

        Slot based retrieval guarantees that the context always contains an
        actionable guidance passage alongside supporting knowledge passages.
        """
        components = {"bm25": self.bm25_scores(query), "dense": self.dense_scores(query),
                      "graph": self.graph_scores(query_tags or set())}
        fused = np.zeros(len(self.records), dtype=np.float64)
        for name, scores in components.items():
            if not np.any(scores > 0):
                continue
            fused += self.weights[name] / (self.rrf_k + self._ranks(scores) + 1.0)
        ranked = list(np.argsort(np.negative(fused), kind="stable"))
        management = [i for i in ranked if self.records[i]["recommendation_id"]][:management_slots]
        rest = [i for i in ranked if i not in management]
        order = (management + rest)[:k]
        hits = []
        for rank, i in enumerate(order, start=1):
            hits.append(RetrievalHit(self.doc_ids[i], float(fused[i]), rank,
                                     {n: float(max(s[i], 0.0)) for n, s in components.items()}))
        return hits


CATEGORY_DESCRIPTORS = {
    "1": "negative or benign appearance", "2": "benign behaviour", "3": "probably benign short interval",
    "4A": "suspicious", "4B": "very suspicious work up", "4X": "additional suspicious features",
}
TYPE_TAGS = {"solid": "solid_nodule", "part solid": "part_solid_nodule", "ground glass": "ground_glass_nodule",
             "calcified": "calcified_nodule"}


def build_query(findings: dict):
    """Compose a retrieval query and concept tags from (predicted) case findings.

    The query mirrors how a clinician searches guidance: a confirmed tissue
    diagnosis dominates; otherwise screening and higher risk nodules are framed
    by their Lung RADS category, and low risk incidental nodules by size and
    morphology. Expected keys: lung_rads_category, nodule_type, margin,
    nodule_present, mediastinal_lymphadenopathy, histology_subtype,
    referral_pathway.
    """
    category = str(findings.get("lung_rads_category", "1")).upper()
    ntype = findings.get("nodule_type", "none")
    present = int(findings.get("nodule_present", 0))
    subtype = findings.get("histology_subtype", "not sampled") or "not sampled"
    pathway = findings.get("referral_pathway", "screening programme")
    words: list[str] = []
    tags: set[str] = set()
    if present and ntype in TYPE_TAGS:
        words.append(f"{ntype} nodule")
        tags.add(TYPE_TAGS[ntype])
    if findings.get("margin") == "spiculated":
        words.append("spiculated margin")
        tags.add("spiculation")
    if int(findings.get("mediastinal_lymphadenopathy", 0)):
        words.append("enlarged mediastinal lymph nodes")
        tags.add("lymphadenopathy")
    if subtype != "not sampled":
        if subtype in ("granuloma", "hamartoma", "organising pneumonia"):
            words.append(f"benign concordant histology {subtype} matches imaging")
            tags.update({"benign", subtype.replace(" ", "_")})
        elif subtype == "nondiagnostic sample":
            words.append("nondiagnostic discordant sampling nonspecific inflammation")
            tags.add("nondiagnostic")
        else:
            words.append(f"histology confirms {subtype} management pathway referral")
            tags.update({"malignancy", subtype.replace(" ", "_")})
    elif pathway in ("screening programme", "surveillance") or category not in ("1", "2"):
        words.insert(0, f"Lung RADS category {category} {CATEGORY_DESCRIPTORS.get(category, '')}")
        tags.add(f"lung_rads_{category.lower()}")
        if pathway in ("screening programme", "surveillance"):
            words.append("screening round")
            tags.add("screening")
    else:
        if ntype == "calcified":
            words.append("benign pattern of calcification")
        else:
            words.append("incidentally detected small nodule in a low risk adult")
            tags.update({"incidental", "low_risk"})
    if not present:
        words.append("no lung nodule")
    return ". ".join(words), tags
