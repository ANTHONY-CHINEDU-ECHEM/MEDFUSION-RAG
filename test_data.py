"""Dataset integrity tests."""
import pandas as pd
import pytest

from medfusion.data.generator import COLUMN_SPEC, CohortGenerator, lung_rads_category
from medfusion.data.knowledge import RECOMMENDATIONS, kb_records
from medfusion.utils import PROJECT_ROOT

CSV = PROJECT_ROOT / "data" / "medfusion_lung_cohort.csv"


@pytest.fixture(scope="module")
def frame():
    if not CSV.exists():
        pytest.skip("Run python run.py data first")
    return pd.read_csv(CSV, keep_default_na=False, na_values=[""])


def test_shape_meets_specification(frame):
    assert len(frame) >= 16000
    assert frame.shape[1] >= 35
    assert list(frame.columns) == list(COLUMN_SPEC)


def test_identifiers_unique_and_no_patient_leakage(frame):
    assert frame["case_id"].is_unique
    splits_per_patient = frame.groupby("patient_id")["split"].nunique()
    assert splits_per_patient.max() == 1


def test_clinical_consistency(frame):
    calcified = frame[frame["nodule_type"] == "calcified"]
    assert (calcified["malignancy_label"] == 0).all()
    absent = frame[frame["nodule_present"] == 0]
    assert (absent["lung_rads_category"].astype(str) == "1").all()
    sampled = frame[frame["histology_available"] == 1]
    assert (sampled["histology_subtype"] != "not sampled").all()
    assert frame["recommendation_text"].isin(RECOMMENDATIONS.values()).all()


def test_report_recommendation_grounded_in_knowledge_base(frame):
    kb_ids = {r["doc_id"] for r in kb_records()}
    for ids in frame["gold_guideline_doc_ids"].head(500):
        assert set(ids.split("|")) <= kb_ids


def test_generator_is_deterministic():
    a = CohortGenerator(200, seed=11).generate()
    b = CohortGenerator(200, seed=11).generate()
    pd.testing.assert_frame_equal(a, b)


def test_lung_rads_rules():
    assert lung_rads_category("solid", 5.0, None, False, "baseline") == "2"
    assert lung_rads_category("solid", 9.0, None, False, "baseline") == "4A"
    assert lung_rads_category("solid", 16.0, None, True, "baseline") == "4X"
    assert lung_rads_category("part solid", 12.0, 9.0, False, "baseline") == "4B"
    assert lung_rads_category("calcified", 9.0, None, False, "baseline") == "1"
    assert lung_rads_category("solid", 7.0, None, False, "stable") == "2"
