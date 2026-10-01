"""Tokenizer, retrieval and metric tests."""
import pandas as pd

from medfusion.data.generator import CohortGenerator
from medfusion.data.knowledge import RECOMMENDATIONS
from medfusion.data.tokenizer import ClinicalTokenizer
from medfusion.evaluation.metrics import clinical_efficacy, corpus_bleu, extract_entities, rouge_l
from medfusion.rag.graph import KnowledgeGraph
from medfusion.rag.retriever import HybridRetriever, build_query
from medfusion.utils import PROJECT_ROOT

FRAME = CohortGenerator(600, seed=3).generate()


def test_tokenizer_round_trip_is_lossless():
    tok = ClinicalTokenizer().fit(FRAME["integrated_report"], min_freq=1)
    for text in FRAME["integrated_report"]:
        assert tok.decode(tok.encode(text)) == text


def test_hybrid_retrieval_recall_with_reference_findings():
    graph = KnowledgeGraph.load(PROJECT_ROOT / "data" / "knowledge_graph")
    retriever = HybridRetriever.from_jsonl(PROJECT_ROOT / "data" / "knowledge_base" / "clinical_guidance_passages.jsonl", graph)
    hits = 0
    for row in FRAME.to_dict("records"):
        query, tags = build_query(row)
        top = retriever.search(query, tags, k=3)[0].doc_id
        hits += top in row["gold_guideline_doc_ids"].split("|")
    assert hits / len(FRAME) > 0.95


def test_metrics_on_identical_text():
    refs = list(FRAME["integrated_report"].head(50))
    assert corpus_bleu(refs, refs)["bleu_4"] > 0.999
    assert rouge_l(refs, refs) > 0.999
    assert clinical_efficacy(refs, refs, RECOMMENDATIONS)["clinical_entity_macro_accuracy"] == 1.0


def test_entity_extraction():
    row = FRAME[FRAME["histology_subtype"] == "adenocarcinoma"].iloc[0]
    ents = extract_entities(row["integrated_report"])
    assert ents["histology"] == "adenocarcinoma"
    assert ents["lung_rads"] == str(row["lung_rads_category"])
