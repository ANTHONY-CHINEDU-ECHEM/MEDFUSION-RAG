"""Synthetic multimodal lung nodule cohort generator.

The generator simulates a realistic, longitudinal, multi site lung nodule
cohort that combines low dose screening, incidental findings and urgent
suspected cancer referrals. Each row is one imaging study and carries:

* patient demographics and risk factors (EHR structured fields)
* CT acquisition metadata and a deterministic key image specification
* nodule morphology, Brock style malignancy risk and a Lung RADS style category
* histopathology, grading, proliferation and biomarker results when sampled
* TNM staging, ICD 10 coding and a guideline grounded management decision
* a free text clinical note (model input) and an integrated radiology
  pathology report (generation target)

Clinical relationships follow published epidemiology at a coarse level but all
values are simulated. The cohort is enriched for malignancy so that models
receive enough positive examples. Nothing here describes a real person.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from medfusion.data.knowledge import RECOMMENDATIONS
from medfusion.utils import get_logger, neg, sub

LOGGER = get_logger("generator")

LOBES = ["RUL", "RML", "RLL", "LUL", "LLL"]
LOBE_PROBS = [0.30, 0.10, 0.18, 0.25, 0.17]
LOBE_NAMES = {"RUL": "right upper lobe", "RML": "right middle lobe", "RLL": "right lower lobe",
              "LUL": "left upper lobe", "LLL": "left lower lobe"}
# Approximate key image positions in normalised image coordinates (x, y).
# Radiological convention: patient right appears on the image left.
LOBE_CENTRES = {"RUL": (0.31, 0.38), "RML": (0.30, 0.52), "RLL": (0.30, 0.62),
                "LUL": (0.69, 0.40), "LLL": (0.70, 0.62)}
NODULE_TYPES = ["solid", "part solid", "ground glass", "calcified"]
NODULE_TYPE_PROBS = [0.64, 0.15, 0.13, 0.08]
CALCIFICATION_PATTERNS = ["central", "laminated", "popcorn", "diffuse"]
SITES = {
    "SITE01": ("Siemens Healthineers", "North West"), "SITE02": ("GE HealthCare", "North West"),
    "SITE03": ("Philips", "Yorkshire"), "SITE04": ("Canon Medical", "Midlands"),
    "SITE05": ("Siemens Healthineers", "Midlands"), "SITE06": ("GE HealthCare", "London"),
    "SITE07": ("Philips", "South West"), "SITE08": ("Canon Medical", "North East"),
}
SITE_WEIGHTS = [0.17, 0.15, 0.13, 0.12, 0.12, 0.13, 0.09, 0.09]
PATHWAY_PHRASES = {
    "screening programme": ["targeted lung health check programme", "lung cancer screening programme",
                            "community lung screening service"],
    "incidental finding": ["incidental finding pathway", "incidental nodule clinic"],
    "urgent suspected cancer": ["urgent suspected cancer pathway", "two week wait lung pathway"],
    "surveillance": ["nodule surveillance programme", "nodule follow up clinic"],
}
NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

COLUMN_SPEC = {
    "case_id": ("string", "Unique study identifier"),
    "patient_id": ("string", "Pseudonymised patient identifier; patients may have several studies"),
    "study_sequence": ("int", "Order of the study within the patient timeline, starting at 1"),
    "study_date": ("date", "Study date in YYYY/MM/DD format"),
    "interval_days_from_prior": ("int", "Days since the previous study for the same patient; blank at baseline"),
    "site_id": ("category", "Acquisition site code"),
    "region": ("category", "NHS region of the acquisition site"),
    "referral_pathway": ("category", "screening programme, incidental finding, urgent suspected cancer or surveillance"),
    "age_years": ("int", "Age at the time of the study"),
    "sex": ("category", "Recorded sex"),
    "smoking_status": ("category", "never, former or current"),
    "pack_years": ("float", "Cumulative tobacco exposure in pack years"),
    "years_since_quit": ("int", "Years since smoking cessation for former smokers"),
    "family_history_lung_cancer": ("binary", "First degree relative with lung cancer"),
    "copd_diagnosis": ("binary", "Recorded chronic obstructive pulmonary disease"),
    "prior_extrathoracic_cancer": ("binary", "Previous cancer outside the thorax"),
    "bmi": ("float", "Body mass index in kilograms per square metre; may be missing"),
    "ecog_status": ("int", "ECOG performance status 0 to 3; may be missing"),
    "presenting_symptom": ("category", "Dominant presenting symptom"),
    "scanner_vendor": ("category", "CT scanner manufacturer"),
    "slice_thickness_mm": ("float", "Reconstructed slice thickness"),
    "tube_voltage_kvp": ("int", "Tube voltage"),
    "ctdivol_mgy": ("float", "Volume CT dose index in milligray"),
    "contrast_enhanced": ("binary", "Intravenous contrast administered"),
    "emphysema_severity": ("category", "none, mild, moderate or severe"),
    "nodule_present": ("binary", "At least one pulmonary nodule identified"),
    "nodule_count": ("int", "Number of nodules identified"),
    "dominant_nodule_lobe": ("category", "Lobe of the dominant nodule (RUL, RML, RLL, LUL, LLL or none)"),
    "nodule_x_norm": ("float", "Horizontal centre of the dominant nodule on the key image (0 to 1)"),
    "nodule_y_norm": ("float", "Vertical centre of the dominant nodule on the key image (0 to 1)"),
    "nodule_diameter_mm": ("float", "Mean axial diameter of the dominant nodule"),
    "solid_component_mm": ("float", "Solid component diameter for part solid nodules"),
    "nodule_volume_mm3": ("float", "Estimated dominant nodule volume"),
    "nodule_type": ("category", "solid, part solid, ground glass, calcified or none"),
    "margin": ("category", "smooth, lobulated, spiculated or none"),
    "calcification_pattern": ("category", "Benign calcification pattern when calcified"),
    "pleural_contact": ("binary", "Dominant nodule abuts the pleura"),
    "mediastinal_lymphadenopathy": ("binary", "Enlarged mediastinal lymph nodes on CT"),
    "prior_diameter_mm": ("float", "Diameter of the same nodule on the prior study"),
    "growth_status": ("category", "baseline, growing, stable or decreasing"),
    "volume_doubling_time_days": ("int", "Volume doubling time when interval growth is present"),
    "brock_risk_pct": ("float", "Brock style malignancy probability in percent"),
    "lung_rads_category": ("category", "Lung RADS style category: 1, 2, 3, 4A, 4B or 4X"),
    "histology_available": ("binary", "Tissue sample and whole slide image available"),
    "sampling_procedure": ("category", "Tissue sampling method"),
    "histology_subtype": ("category", "Histopathological diagnosis"),
    "adenocarcinoma_pattern": ("category", "Predominant adenocarcinoma growth pattern"),
    "tumour_grade": ("category", "Histological grade or neuroendocrine category"),
    "mitoses_per_2mm2": ("int", "Mitotic count per 2 square millimetres"),
    "ki67_index_pct": ("int", "Ki67 proliferation index in percent"),
    "tumour_cellularity_pct": ("float", "Viable tumour cellularity of the sample in percent"),
    "nuclear_pleomorphism_score": ("int", "Nuclear pleomorphism score 1 to 3"),
    "necrosis_present": ("binary", "Tumour necrosis present in the sample"),
    "pdl1_tps_pct": ("int", "PDL1 tumour proportion score in percent"),
    "molecular_driver": ("category", "Actionable driver result"),
    "t_stage": ("category", "Clinical T descriptor"),
    "n_stage": ("category", "Clinical N descriptor"),
    "m_stage": ("category", "Clinical M descriptor"),
    "clinical_stage": ("category", "TNM 8th edition style stage group"),
    "icd10_code": ("category", "ICD 10 code for the encounter"),
    "malignancy_label": ("binary", "Reference standard malignancy of the dominant nodule"),
    "reference_standard": ("category", "How the malignancy label was established"),
    "recommendation_id": ("category", "Identifier of the guideline recommendation (R01 to R09)"),
    "gold_guideline_doc_ids": ("string", "Knowledge base passages that support the recommendation"),
    "ct_image_seed": ("int", "Seed for deterministic rendering of the CT key image"),
    "wsi_image_seed": ("int", "Seed for deterministic rendering of the whole slide image tile"),
    "clinical_note": ("text", "Free text referral and EHR summary (model input)"),
    "findings_text": ("text", "CT findings section of the integrated report"),
    "pathology_text": ("text", "Pathology section of the integrated report"),
    "impression_text": ("text", "Impression section of the integrated report"),
    "recommendation_text": ("text", "Guideline grounded recommendation sentence"),
    "integrated_report": ("text", "Complete integrated radiology pathology report (generation target)"),
    "split": ("category", "train, validation or test, assigned at patient level"),
}


def _sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(neg(x)))


def _logit(p: float):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / sub(1.0, p))


def brock_logit(age, female, family, emphysema, diameter, ntype, upper, count, spiculated):
    """PanCan (Brock) full model linear predictor for a non calcified nodule."""
    inverse_root = 1.0 / math.sqrt(max(diameter, 1.0) / 10.0)
    logit = 0.0287 * sub(age, 62) + 0.6011 * female + 0.2961 * family + 0.2953 * emphysema
    logit = sub(logit, 5.3854 * sub(inverse_root, 1.58113883))
    if ntype == "ground glass":
        logit = sub(logit, 0.1276)
    elif ntype == "part solid":
        logit += 0.377
    logit += 0.6581 * upper
    logit = sub(logit, 0.0824 * sub(count, 4))
    logit += 0.7729 * spiculated
    return sub(logit, 6.7892)


def lung_rads_category(ntype, diameter, solid_component, suspicious_feature, growth_status):
    """Simplified Lung RADS 2022 style assignment for the dominant nodule."""
    order = ["1", "2", "3", "4A", "4B"]
    if ntype in (None, "none", "calcified"):
        return "1"
    if ntype == "solid":
        category = "2" if diameter < 6 else "3" if diameter < 8 else "4A" if diameter < 15 else "4B"
    elif ntype == "part solid":
        if diameter < 6:
            category = "2"
        else:
            category = "3" if solid_component < 6 else "4A" if solid_component < 8 else "4B"
    else:
        category = "2" if diameter < 30 else "3"
    if growth_status == "growing":
        target = "4A" if diameter < 8 else "4B"
        if ntype == "ground glass":
            target = "3"
        category = max(category, target, key=order.index)
    elif growth_status in ("stable", "decreasing") and category in ("3", "4A"):
        category = "2"
    if category in ("3", "4A", "4B") and suspicious_feature:
        category = "4X"
    return category


class CohortGenerator:
    """Simulate a longitudinal multimodal lung nodule cohort."""

    def __init__(self, n_rows: int = 16000, seed: int = 20260101):
        self.n_rows = n_rows
        self.rng = np.random.default_rng(seed)
        self.case_counter = 0

    # Patient level simulation
    def _patient(self, index: int):
        rng = self.rng
        pathway = rng.choice(["screening programme", "incidental finding", "urgent suspected cancer"],
                             p=[0.56, 0.29, 0.15])
        if pathway == "screening programme":
            age = int(np.clip(rng.normal(66, 5.5), 55, 80))
        else:
            age = int(np.clip(rng.normal(64, 11), 35, 90))
        sex = "Female" if rng.random() < 0.47 else "Male"
        if pathway == "screening programme":
            smoking = rng.choice(["current", "former"], p=[0.42, 0.58])
        else:
            smoking = rng.choice(["current", "former", "never"], p=[0.27, 0.43, 0.30])
        pack_years = 0.0 if smoking == "never" else float(np.clip(rng.gamma(4.0, 9.0), 5, 120))
        years_quit = float(np.nan)
        if smoking == "former":
            years_quit = float(min(int(rng.integers(1, 26)), max(1, sub(age, 30))))
        if smoking != "never" and rng.random() < 0.04:
            pack_years = float(np.nan)
        family = int(rng.random() < 0.12)
        copd_p = 0.05 if smoking == "never" else min(0.65, 0.08 + (pack_years if pack_years == pack_years else 30) / 120)
        copd = int(rng.random() < copd_p)
        prior_cancer = int(rng.random() < 0.08)
        emph_score = (0 if smoking == "never" else (pack_years if pack_years == pack_years else 30) / 25) + copd * 1.2 + rng.normal(0, 0.6)
        emphysema = "none" if emph_score < 0.9 else "mild" if emph_score < 1.8 else "moderate" if emph_score < 2.7 else "severe"
        bmi = float(np.clip(rng.normal(27.2, 5.0), 16, 48))
        comorbidities = []
        if copd:
            comorbidities.append("COPD")
        if rng.random() < 0.38:
            comorbidities.append("hypertension")
        if rng.random() < 0.15:
            comorbidities.append("type 2 diabetes")
        if rng.random() < 0.12:
            comorbidities.append("ischaemic heart disease")
        if prior_cancer:
            comorbidities.append(str(rng.choice(["previous breast cancer", "previous colorectal cancer",
                                                 "previous prostate cancer", "previous melanoma"])))
        site = str(rng.choice(list(SITES), p=SITE_WEIGHTS))
        return {
            "patient_id": f"PT{index + 100000:06d}", "pathway": str(pathway), "age": age, "sex": sex,
            "smoking": str(smoking), "pack_years": pack_years, "years_quit": years_quit, "family": family,
            "copd": copd, "prior_cancer": prior_cancer, "emphysema": emphysema, "bmi": bmi,
            "comorbidities": comorbidities, "site": site,
            "start_date": date(2021, 1, 4) + timedelta(days=int(rng.integers(0, 1550))),
        }

    def _nodule(self, patient: dict):
        rng = self.rng
        presence_p = {"screening programme": 0.78, "incidental finding": 0.93, "urgent suspected cancer": 0.88}[patient["pathway"]]
        if rng.random() > presence_p:
            return None
        ntype = str(rng.choice(NODULE_TYPES, p=NODULE_TYPE_PROBS))
        scale = 1.55 if patient["pathway"] == "urgent suspected cancer" else 1.0
        if ntype == "solid":
            diameter = float(np.clip(rng.lognormal(math.log(7.2 * scale), 0.5), 3, 38))
        elif ntype == "part solid":
            diameter = float(np.clip(rng.lognormal(math.log(12.0 * scale), 0.4), 5, 38))
        elif ntype == "ground glass":
            diameter = float(np.clip(rng.lognormal(math.log(10.0), 0.5), 4, 40))
        else:
            diameter = float(np.clip(rng.lognormal(math.log(5.0), 0.35), 2, 15))
        solid_component = float(np.nan)
        if ntype == "part solid":
            solid_component = round(diameter * rng.uniform(0.2, 0.8), 1)
        lobe = str(rng.choice(LOBES, p=LOBE_PROBS))
        cx, cy = LOBE_CENTRES[lobe]
        x = float(np.clip(cx + rng.normal(0, 0.055), 0.16, 0.84))
        y = float(np.clip(cy + rng.normal(0, 0.06), 0.24, 0.78))
        if ntype == "calcified":
            margin = "smooth"
        elif ntype == "ground glass":
            margin = str(rng.choice(["smooth", "lobulated", "spiculated"], p=[0.8, 0.15, 0.05]))
        else:
            p_spic = float(np.clip(0.06 + 0.022 * sub(diameter, 5), 0.04, 0.55))
            p_lob = 0.27
            margin = str(rng.choice(["spiculated", "lobulated", "smooth"], p=[p_spic, p_lob, sub(sub(1.0, p_spic), p_lob)]))
        count = int(rng.choice([1, 2, 3, 4, 5], p=[0.56, 0.21, 0.12, 0.07, 0.04]))
        return {
            "type": ntype, "diameter": diameter, "solid_component": solid_component, "lobe": lobe,
            "x": x, "y": y, "margin": margin, "count": count,
            "calcification": str(rng.choice(CALCIFICATION_PATTERNS)) if ntype == "calcified" else "none",
            "pleural_contact": int(rng.random() < (0.22 if ntype != "calcified" else 0.08)),
        }

    def _malignancy(self, patient: dict, nodule: dict | None):
        if nodule is None:
            return 0, 0.0
        if nodule["type"] == "calcified":
            return 0, 0.1
        logit = brock_logit(patient["age"], int(patient["sex"] == "Female"), patient["family"],
                            int(patient["emphysema"] != "none"), nodule["diameter"], nodule["type"],
                            int(nodule["lobe"] in ("RUL", "LUL")), nodule["count"], int(nodule["margin"] == "spiculated"))
        brock = _sigmoid(logit)
        offset = 2.05 + (0.75 if patient["pathway"] == "urgent suspected cancer" else 0.0)
        if patient["smoking"] == "current":
            offset += 0.25
        malignant = int(self.rng.random() < _sigmoid(logit + offset))
        return malignant, round(100.0 * brock, 2)

    # Histopathology simulation
    def _histology(self, patient: dict, nodule: dict, malignant: int):
        rng = self.rng
        result = {"subtype": "not sampled", "pattern": "not applicable", "grade": "not applicable",
                  "mitoses": np.nan, "ki67": np.nan, "cellularity": np.nan, "pleomorphism": np.nan,
                  "necrosis": 0, "pdl1": np.nan, "driver": "not tested"}
        if malignant:
            if rng.random() < 0.04:
                result.update(subtype="nondiagnostic sample", cellularity=round(float(rng.uniform(2, 15)), 1),
                              pleomorphism=1.0)
                return result
            heavy = (patient["pack_years"] if patient["pack_years"] == patient["pack_years"] else 30) >= 40
            if nodule["type"] in ("ground glass", "part solid"):
                probs = [0.95, 0.03, 0.0, 0.01, 0.01]
            elif patient["smoking"] == "never":
                probs = [0.80, 0.05, 0.01, 0.02, 0.12]
            elif heavy:
                probs = [0.42, 0.31, 0.17, 0.05, 0.05]
            else:
                probs = [0.56, 0.24, 0.10, 0.04, 0.06]
            subtype = str(rng.choice(["adenocarcinoma", "squamous cell carcinoma", "small cell carcinoma",
                                      "large cell carcinoma", "carcinoid tumour"], p=probs))
            result["subtype"] = subtype
            if subtype == "adenocarcinoma":
                pattern_probs = [0.45, 0.30, 0.10, 0.08, 0.07] if nodule["type"] != "solid" else [0.08, 0.42, 0.14, 0.13, 0.23]
                pattern = str(rng.choice(["lepidic", "acinar", "papillary", "micropapillary", "solid"], p=pattern_probs))
                grade = {"lepidic": "grade 1", "acinar": "grade 2", "papillary": "grade 2",
                         "micropapillary": "grade 3", "solid": "grade 3"}[pattern]
                result.update(pattern=pattern, grade=grade)
                mit_range = {"grade 1": (1, 4), "grade 2": (3, 12), "grade 3": (8, 30)}[grade]
                ki_range = {"grade 1": (3, 12), "grade 2": (10, 30), "grade 3": (25, 55)}[grade]
                pleo = {"grade 1": 1.0, "grade 2": 2.0, "grade 3": 3.0}[grade]
            elif subtype == "squamous cell carcinoma":
                grade = str(rng.choice(["grade 2", "grade 3"], p=[0.55, 0.45]))
                result["grade"] = grade
                mit_range, ki_range, pleo = (10, 35), (30, 70), 2.0 if grade == "grade 2" else 3.0
            elif subtype == "small cell carcinoma":
                result["grade"] = "high grade"
                mit_range, ki_range, pleo = (50, 110), (65, 98), 3.0
            elif subtype == "large cell carcinoma":
                result["grade"] = "grade 3"
                mit_range, ki_range, pleo = (20, 60), (40, 80), 3.0
            else:
                atypical = rng.random() < 0.25
                result["grade"] = "atypical carcinoid" if atypical else "typical carcinoid"
                mit_range = (2, 9) if atypical else (0, 1)
                ki_range = (5, 20) if atypical else (1, 4)
                pleo = 1.0
            result["mitoses"] = float(rng.integers(mit_range[0], mit_range[1] + 1))
            result["ki67"] = float(rng.integers(ki_range[0], ki_range[1] + 1))
            result["cellularity"] = round(float(rng.uniform(35, 90)), 1)
            result["pleomorphism"] = pleo
            necrosis_p = {"adenocarcinoma": 0.2, "squamous cell carcinoma": 0.45, "small cell carcinoma": 0.7,
                          "large cell carcinoma": 0.6}.get(subtype, 0.0)
            if result["grade"] == "atypical carcinoid":
                necrosis_p = 0.5
            result["necrosis"] = int(rng.random() < necrosis_p)
            if subtype in ("adenocarcinoma", "squamous cell carcinoma", "large cell carcinoma"):
                band = rng.choice(["low", "mid", "high"], p=[0.34, 0.36, 0.30])
                result["pdl1"] = float(0 if band == "low" else rng.integers(1, 50) if band == "mid" else rng.integers(50, 101))
            if subtype == "adenocarcinoma":
                never = patient["smoking"] == "never"
                p_egfr = (0.5 if patient["sex"] == "Female" else 0.35) if never else 0.08
                p_alk = (0.12 if patient["age"] < 60 else 0.05) if never else 0.015
                p_kras = 0.03 if never else 0.14
                draw = rng.random()
                if draw < p_egfr:
                    result["driver"] = "EGFR mutation"
                elif draw < p_egfr + p_alk:
                    result["driver"] = "ALK rearrangement"
                elif draw < p_egfr + p_alk + p_kras:
                    result["driver"] = "KRAS G12C"
                else:
                    result["driver"] = "no driver detected"
        else:
            subtype = "granuloma" if nodule["type"] == "calcified" else str(
                rng.choice(["granuloma", "hamartoma", "organising pneumonia"], p=[0.48, 0.24, 0.28]))
            result.update(subtype=subtype, mitoses=0.0, ki67=float(rng.integers(1, 4)),
                          cellularity=round(float(rng.uniform(5, 30)), 1), pleomorphism=1.0,
                          necrosis=int(subtype == "granuloma" and rng.random() < 0.6))
        return result

    def _staging(self, nodule: dict, histology: dict, lymph: int):
        rng = self.rng
        malignant_types = {"adenocarcinoma", "squamous cell carcinoma", "small cell carcinoma",
                           "large cell carcinoma", "carcinoid tumour"}
        if histology["subtype"] not in malignant_types:
            return {"t": "not applicable", "n": "not applicable", "m": "not applicable", "stage": "not applicable"}
        invasive = nodule["solid_component"] if nodule["type"] == "part solid" else nodule["diameter"]
        if nodule["type"] == "ground glass" and histology["pattern"] == "lepidic" and nodule["diameter"] <= 30:
            t = "Tis"
        elif nodule["type"] == "part solid" and invasive <= 5:
            t = "T1mi"
        elif invasive <= 10:
            t = "T1a"
        elif invasive <= 20:
            t = "T1b"
        elif invasive <= 30:
            t = "T1c"
        elif invasive <= 40:
            t = "T2a"
        else:
            t = "T2b"
        if nodule["pleural_contact"] and t in ("T1a", "T1b", "T1c") and rng.random() < 0.3:
            t = "T2a"
        if lymph:
            n = "N2" if rng.random() < 0.6 else "N1"
        else:
            n = "N1" if rng.random() < 0.07 else "N0"
        m_p = 0.25 if histology["subtype"] == "small cell carcinoma" else 0.025
        m = "M1" if rng.random() < m_p else "M0"
        if t == "Tis":
            n, m = "N0", "M0"
        if m == "M1":
            stage = "IV"
        elif t == "Tis":
            stage = "0"
        elif n == "N2":
            stage = "IIIA"
        elif n == "N1":
            stage = "IIB"
        else:
            stage = {"T1mi": "IA1", "T1a": "IA1", "T1b": "IA2", "T1c": "IA3", "T2a": "IB", "T2b": "IIA"}[t]
        return {"t": t, "n": n, "m": m, "stage": stage}

    # Text generation
    def _note(self, p: dict, study: dict):
        rng = self.rng
        sex_word = "female" if p["sex"] == "Female" else "male"
        opener = str(rng.choice([
            f"{study['age']} year old {sex_word} referred via the {rng.choice(PATHWAY_PHRASES[study['pathway']])}.",
            f"{study['age']}yo {sex_word[0].upper()}, {rng.choice(PATHWAY_PHRASES[study['pathway']])} referral.",
            f"Referral from the {rng.choice(PATHWAY_PHRASES[study['pathway']])}: {study['age']} year old {sex_word}.",
        ]))
        parts = [opener]
        if p["smoking"] == "never":
            parts.append(str(rng.choice(["Never smoker.", "Lifelong non smoker.", "No smoking history."])))
        else:
            packs = "pack year history not recorded" if p["pack_years"] != p["pack_years"] else f"{int(round(p['pack_years']))} pack years"
            if p["smoking"] == "current":
                parts.append(str(rng.choice([f"Current smoker, {packs}.", f"Smokes daily, {packs}."])))
            else:
                quit_years = int(p["years_quit"])
                unit = "year" if quit_years == 1 else "years"
                parts.append(f"Ex smoker, stopped {quit_years} {unit} ago, {packs}.")
        if p["comorbidities"]:
            joined = ", ".join(p["comorbidities"])
            parts.append(str(rng.choice([f"PMH: {joined}.", f"Past medical history includes {joined}."])))
        else:
            parts.append("No significant past medical history.")
        symptom = study["symptom"]
        symptom_text = {
            "asymptomatic": ["Asymptomatic.", "No respiratory symptoms reported."],
            "persistent cough": ["Persistent cough for over three weeks.", "Reports a chronic cough, worse recently."],
            "haemoptysis": ["Two episodes of haemoptysis.", "Reports streaks of blood in sputum."],
            "weight loss": ["Unintentional weight loss over three months.", "Reports appetite and weight loss."],
            "breathlessness": ["Increasing breathlessness on exertion.", "SOB on exertion, progressive."],
            "chest pain": ["Intermittent pleuritic chest pain.", "Reports right sided chest discomfort."],
            "recurrent chest infection": ["Two chest infections treated in primary care this winter."],
        }[symptom]
        parts.append(str(rng.choice(symptom_text)))
        if p["family"]:
            relative = rng.choice(["Father", "Mother", "Brother", "Sister"])
            parts.append(f"FHx: {relative.lower()} diagnosed with lung cancer.")
        if study["ecog"] == study["ecog"]:
            parts.append(f"ECOG {int(study['ecog'])}.")
        if study["prior_diameter"] == study["prior_diameter"]:
            months = max(1, int(round(study["interval_days"] / 30.4)))
            parts.append(f"Prior CT {months} months ago showed a {int(round(study['prior_diameter']))} mm "
                         f"{study['type']} nodule in the {LOBE_NAMES[study['lobe']]}.")
        parts.append(str(rng.choice(["Please assess and advise on management.", "Query pulmonary nodule, please advise.",
                                     "For CT assessment and nodule management advice."])))
        return " ".join(parts)

    def _findings(self, study: dict):
        rng = self.rng
        if not study["nodule_present"]:
            sentences = [str(rng.choice(["No pulmonary nodule or mass is identified.",
                                         "No suspicious pulmonary nodule is seen."]))]
        else:
            d = int(round(study["diameter"]))
            lobe = LOBE_NAMES[study["lobe"]]
            ntype = study["type"]
            noun = {"solid": "solid nodule", "part solid": "part solid nodule",
                    "ground glass": "pure ground glass nodule", "calcified": "calcified nodule"}[ntype]
            if rng.random() < 0.5:
                lead = f"There is a {d} mm {noun} in the {lobe}"
            else:
                lead = f"A {noun} in the {lobe} measures {d} mm"
            details = []
            if ntype == "part solid":
                details.append(f"a {int(round(study['solid_component']))} mm solid component")
            if ntype == "calcified":
                details.append(f"{study['calcification']} calcification")
            elif study["margin"] != "none":
                details.append(f"{study['margin']} margins")
            descriptor = " with " + " and ".join(details) if details else ""
            if study["pleural_contact"]:
                descriptor += ", abutting the pleura"
            sentences = [lead + descriptor + "."]
            if study["growth"] == "growing":
                sentences.append(f"Previously {int(round(study['prior_diameter']))} mm, in keeping with interval growth; "
                                 f"estimated volume doubling time {int(round(study['vdt']))} days.")
            elif study["growth"] == "stable":
                sentences.append(f"Unchanged from the prior study at {int(round(study['prior_diameter']))} mm.")
            elif study["growth"] == "decreasing":
                sentences.append(f"Decreased from {int(round(study['prior_diameter']))} mm on the prior study.")
            extra = sub(study["count"], 1)
            if extra > 0:
                noun_extra = "nodule is" if extra == 1 else "nodules are"
                sentences.append(f"{NUMBER_WORDS[extra]} additional small {noun_extra} noted, below 4 mm.")
        sentences.append({"none": "No emphysema.", "mild": "Mild centrilobular emphysema.",
                          "moderate": "Moderate centrilobular emphysema.",
                          "severe": "Severe centrilobular emphysema."}[study["emphysema"]])
        sentences.append("Enlarged mediastinal lymph nodes are present." if study["lymph"]
                         else "No mediastinal or hilar lymphadenopathy.")
        return " ".join(sentences)

    @staticmethod
    def _pathology(study: dict):
        h = study["histology"]
        if not study["histology_available"]:
            return "No histology available."
        proc = study["procedure"][0].upper() + study["procedure"][1:]
        subtype = h["subtype"]
        if subtype == "nondiagnostic sample":
            return f"{proc} shows nonspecific inflammation only; the sample is nondiagnostic."
        if subtype in ("granuloma", "hamartoma", "organising pneumonia"):
            label = {"granuloma": "necrotising granulomatous inflammation" if h["necrosis"] else "granulomatous inflammation",
                     "hamartoma": "a pulmonary hamartoma", "organising pneumonia": "organising pneumonia"}[subtype]
            return f"{proc} shows {label} with no evidence of malignancy."
        text = f"{proc} shows {subtype}"
        if subtype == "adenocarcinoma":
            text += f", {h['pattern']} predominant, {h['grade']}"
        elif subtype in ("squamous cell carcinoma", "large cell carcinoma", "small cell carcinoma"):
            text += f", {h['grade']}"
        else:
            text += f", {h['grade']}"
        text += f". Ki67 {int(h['ki67'])}%."
        if h["pdl1"] == h["pdl1"]:
            text += f" PDL1 TPS {int(h['pdl1'])}%."
        if h["driver"] not in ("not tested",):
            text += f" Molecular testing: {h['driver']}."
        return text

    @staticmethod
    def _impression(study: dict):
        cat = study["lung_rads"]
        if not study["nodule_present"]:
            base = "No suspicious pulmonary nodule."
        elif study["type"] == "calcified":
            base = "Benign calcified nodule."
        else:
            base = {"1": "No suspicious pulmonary nodule.", "2": "Benign appearing nodule.",
                    "3": "Probably benign nodule.", "4A": "Suspicious nodule.",
                    "4B": "Highly suspicious nodule, concerning for primary lung malignancy.",
                    "4X": "Highly suspicious nodule, concerning for primary lung malignancy."}[cat]
        text = f"Lung RADS {cat}. {base}"
        h = study["histology"]
        if study["histology_available"]:
            subtype = h["subtype"]
            if subtype in ("granuloma", "hamartoma", "organising pneumonia"):
                text += f" Histology confirms benign {subtype}."
            elif subtype == "nondiagnostic sample":
                text += " Histology is nondiagnostic."
            else:
                s = study["staging"]
                text += f" Histology confirms {subtype}, clinical stage {s['stage']} ({s['t']} {s['n']} {s['m']})."
        return text

    @staticmethod
    def _recommendation_id(study: dict):
        cat = study["lung_rads"]
        if study["histology_available"]:
            subtype = study["histology"]["subtype"]
            if subtype == "small cell carcinoma":
                return "R08"
            if subtype in ("granuloma", "hamartoma", "organising pneumonia"):
                return "R07"
            if subtype == "nondiagnostic sample":
                return "R09"
            return "R06"
        if cat in ("4B", "4X"):
            return "R05"
        if cat == "4A":
            return "R04"
        if cat == "3":
            return "R03"
        if study["pathway"] in ("screening programme", "surveillance"):
            return "R01"
        return "R02"

    # Study timeline simulation
    def _symptom(self, pathway: str, malignant: int):
        rng = self.rng
        if pathway in ("screening programme", "surveillance"):
            return "asymptomatic" if rng.random() < 0.9 else "persistent cough"
        options = ["asymptomatic", "persistent cough", "haemoptysis", "weight loss", "breathlessness",
                   "chest pain", "recurrent chest infection"]
        if pathway == "urgent suspected cancer":
            probs = [0.02, 0.28, 0.18, 0.2, 0.14, 0.1, 0.08] if malignant else [0.04, 0.33, 0.1, 0.1, 0.18, 0.12, 0.13]
        else:
            probs = [0.55, 0.13, 0.03, 0.04, 0.11, 0.08, 0.06]
        return str(rng.choice(options, p=probs))

    def _site_protocol(self, site: str, pathway: str):
        rng = self.rng
        vendor, region = SITES[site]
        if pathway == "screening programme" or pathway == "surveillance":
            ctdi = float(np.clip(rng.normal(1.6, 0.5), 0.6, 3.0))
            contrast = 0
        else:
            ctdi = float(np.clip(rng.normal(7.5, 2.0), 3.5, 14.0))
            contrast = int(rng.random() < 0.62)
        return {"vendor": vendor, "region": region,
                "slice": float(rng.choice([0.625, 1.0, 1.25, 1.5], p=[0.2, 0.45, 0.25, 0.1])),
                "kvp": int(rng.choice([100, 120], p=[0.45, 0.55])), "ctdi": round(ctdi, 2), "contrast": contrast}

    def _simulate_patient(self, index: int):
        rng = self.rng
        p = self._patient(index)
        nodule = self._nodule(p)
        malignant, brock = self._malignancy(p, nodule)
        lymph_p = 0.32 if malignant else 0.05
        lymph = int(rng.random() < lymph_p) if nodule is not None else int(rng.random() < 0.02)
        if nodule is not None and malignant:
            if nodule["type"] == "solid":
                vdt = float(np.clip(rng.lognormal(math.log(190), 0.45), 45, 700))
            elif nodule["type"] == "part solid":
                vdt = float(np.clip(rng.lognormal(math.log(420), 0.4), 150, 1200))
            else:
                vdt = float(np.clip(rng.lognormal(math.log(780), 0.4), 300, 2000))
        else:
            vdt = float(np.nan)
        ecog = float(rng.choice([0, 1, 2, 3], p=[0.55, 0.3, 0.12, 0.03]))
        rows = []
        current_date = p["start_date"]
        prior_diameter = float(np.nan)
        diameter = nodule["diameter"] if nodule else float(np.nan)
        solid_component = nodule["solid_component"] if nodule else float(np.nan)
        interval_days = float(np.nan)
        for sequence in range(1, 4):
            pathway = p["pathway"] if sequence == 1 else "surveillance"
            growth = "baseline"
            vdt_observed = float(np.nan)
            if sequence > 1 and nodule is not None:
                if malignant:
                    factor = 2 ** (interval_days / (3.0 * vdt))
                    if nodule["type"] == "ground glass" and rng.random() < 0.5:
                        factor = 1.0
                    growth = "growing" if sub(diameter * factor, diameter) >= 1.5 else "stable"
                    diameter = diameter * factor
                    if nodule["type"] == "part solid":
                        solid_component = min(diameter, solid_component * factor * 1.15)
                    if growth == "growing":
                        vdt_observed = vdt * float(rng.uniform(0.85, 1.15))
                else:
                    if rng.random() < 0.15 and nodule["type"] != "calcified":
                        diameter = diameter * float(rng.uniform(0.55, 0.85))
                        growth = "decreasing"
                    else:
                        growth = "stable"
            age_now = p["age"] + int(sub(current_date, p["start_date"]).days // 365)
            suspicious = nodule is not None and (nodule["margin"] == "spiculated" or lymph)
            category = lung_rads_category(nodule["type"] if nodule else None, diameter,
                                          solid_component, suspicious, growth) if nodule else "1"
            p_biopsy = {"1": 0.0, "2": 0.0, "3": 0.02, "4A": 0.12, "4B": 0.82, "4X": 0.84}[category]
            if growth == "growing":
                p_biopsy = min(0.95, p_biopsy + 0.45)
            histology_available = int(nodule is not None and nodule["type"] != "calcified" and rng.random() < p_biopsy)
            histology = {"subtype": "not sampled", "pattern": "not applicable", "grade": "not applicable",
                         "mitoses": np.nan, "ki67": np.nan, "cellularity": np.nan, "pleomorphism": np.nan,
                         "necrosis": 0, "pdl1": np.nan, "driver": "not tested"}
            procedure = "not applicable"
            staging = {"t": "not applicable", "n": "not applicable", "m": "not applicable", "stage": "not applicable"}
            if histology_available:
                nod_now = dict(nodule, diameter=diameter, solid_component=solid_component)
                histology = self._histology(p, nod_now, malignant)
                procedure = self._procedure(lymph)
                staging = self._staging(nod_now, histology, lymph)
            study = {
                "pathway": pathway, "age": age_now, "symptom": self._symptom(pathway, malignant), "ecog": ecog,
                "nodule_present": int(nodule is not None), "type": nodule["type"] if nodule else "none",
                "diameter": diameter, "solid_component": solid_component, "lobe": nodule["lobe"] if nodule else "none",
                "margin": nodule["margin"] if nodule else "none",
                "calcification": nodule["calcification"] if nodule else "none",
                "pleural_contact": nodule["pleural_contact"] if nodule else 0,
                "count": nodule["count"] if nodule else 0, "emphysema": p["emphysema"], "lymph": lymph,
                "growth": growth if nodule else "baseline", "prior_diameter": prior_diameter, "vdt": vdt_observed,
                "interval_days": interval_days, "lung_rads": category, "histology_available": histology_available,
                "histology": histology, "procedure": procedure, "staging": staging,
            }
            if rng.random() < 0.08:
                study["ecog"] = float(np.nan)
            rec_id = self._recommendation_id(study)
            rows.append(self._assemble_row(p, nodule, study, malignant, brock, current_date, sequence, rec_id))
            follow_p = {"1": 0.1, "2": 0.12, "3": 0.82, "4A": 0.85, "4B": 0.08, "4X": 0.06}[category]
            if histology_available or rng.random() > follow_p:
                break
            interval_days = float({"1": 365, "2": 365, "3": 182, "4A": 91, "4B": 60, "4X": 60}[category]
                                  + int(rng.integers(0, 21)))
            prior_diameter = diameter
            current_date = current_date + timedelta(days=int(interval_days))
        return rows

    def _procedure(self, lymph: int):
        options = ["CT guided core biopsy", "EBUS guided needle aspiration", "surgical wedge resection"]
        probs = [0.55, 0.3, 0.15] if lymph else [0.62, 0.1, 0.28]
        return str(self.rng.choice(options, p=probs))

    def _assemble_row(self, p, nodule, study, malignant, brock, current_date, sequence, rec_id):
        rng = self.rng
        self.case_counter += 1
        protocol = self._site_protocol(p["site"], study["pathway"])
        h = study["histology"]
        findings = self._findings(study)
        pathology = self._pathology(study)
        impression = self._impression(study)
        recommendation = RECOMMENDATIONS[rec_id]
        report = (f"FINDINGS: {findings} PATHOLOGY: {pathology} IMPRESSION: {impression} "
                  f"RECOMMENDATION: {recommendation}")
        if study["histology_available"] and h["subtype"] not in ("granuloma", "hamartoma", "organising pneumonia",
                                                                   "nondiagnostic sample"):
            lobe = study["lobe"]
            icd = "C34.1" if lobe in ("RUL", "LUL") else "C34.2" if lobe == "RML" else "C34.3"
        elif h["subtype"] == "hamartoma":
            icd = "D14.3"
        elif study["nodule_present"]:
            icd = "R91"
        else:
            icd = "Z12.2" if study["pathway"] in ("screening programme", "surveillance") else "Z03.1"
        if study["histology_available"]:
            reference = "histology"
        elif malignant:
            reference = "clinical follow up"
        else:
            reference = "imaging stability"
        diameter = study["diameter"]
        volume = math.pi / 6.0 * diameter ** 3 if diameter == diameter else float(np.nan)
        note = self._note(p, study)
        return {
            "case_id": f"MF{self.case_counter:06d}", "patient_id": p["patient_id"], "study_sequence": sequence,
            "study_date": current_date.strftime("%Y/%m/%d"), "interval_days_from_prior": study["interval_days"],
            "site_id": p["site"], "region": protocol["region"], "referral_pathway": study["pathway"],
            "age_years": study["age"], "sex": p["sex"], "smoking_status": p["smoking"],
            "pack_years": round(p["pack_years"], 1) if p["pack_years"] == p["pack_years"] else np.nan,
            "years_since_quit": p["years_quit"], "family_history_lung_cancer": p["family"],
            "copd_diagnosis": p["copd"], "prior_extrathoracic_cancer": p["prior_cancer"],
            "bmi": round(p["bmi"], 1) if rng.random() > 0.035 else np.nan, "ecog_status": study["ecog"],
            "presenting_symptom": study["symptom"], "scanner_vendor": protocol["vendor"],
            "slice_thickness_mm": protocol["slice"], "tube_voltage_kvp": protocol["kvp"],
            "ctdivol_mgy": protocol["ctdi"], "contrast_enhanced": protocol["contrast"],
            "emphysema_severity": study["emphysema"], "nodule_present": study["nodule_present"],
            "nodule_count": study["count"], "dominant_nodule_lobe": study["lobe"],
            "nodule_x_norm": round(nodule["x"], 4) if nodule else np.nan,
            "nodule_y_norm": round(nodule["y"], 4) if nodule else np.nan,
            "nodule_diameter_mm": round(diameter, 1) if diameter == diameter else np.nan,
            "solid_component_mm": round(study["solid_component"], 1) if study["solid_component"] == study["solid_component"] else np.nan,
            "nodule_volume_mm3": round(volume, 1) if volume == volume else np.nan,
            "nodule_type": study["type"], "margin": study["margin"], "calcification_pattern": study["calcification"],
            "pleural_contact": study["pleural_contact"], "mediastinal_lymphadenopathy": study["lymph"],
            "prior_diameter_mm": round(study["prior_diameter"], 1) if study["prior_diameter"] == study["prior_diameter"] else np.nan,
            "growth_status": study["growth"],
            "volume_doubling_time_days": round(study["vdt"], 0) if study["vdt"] == study["vdt"] else np.nan,
            "brock_risk_pct": brock if study["nodule_present"] else np.nan,
            "lung_rads_category": study["lung_rads"], "histology_available": study["histology_available"],
            "sampling_procedure": study["procedure"], "histology_subtype": h["subtype"],
            "adenocarcinoma_pattern": h["pattern"], "tumour_grade": h["grade"], "mitoses_per_2mm2": h["mitoses"],
            "ki67_index_pct": h["ki67"], "tumour_cellularity_pct": h["cellularity"],
            "nuclear_pleomorphism_score": h["pleomorphism"], "necrosis_present": int(h["necrosis"]),
            "pdl1_tps_pct": h["pdl1"], "molecular_driver": h["driver"], "t_stage": study["staging"]["t"],
            "n_stage": study["staging"]["n"], "m_stage": study["staging"]["m"],
            "clinical_stage": study["staging"]["stage"], "icd10_code": icd,
            "malignancy_label": int(malignant and study["nodule_present"]), "reference_standard": reference,
            "recommendation_id": rec_id, "gold_guideline_doc_ids": "",
            "ct_image_seed": int(rng.integers(1, 2 ** 31)),
            "wsi_image_seed": float(rng.integers(1, 2 ** 31)) if study["histology_available"] else np.nan,
            "clinical_note": note, "findings_text": findings, "pathology_text": pathology,
            "impression_text": impression, "recommendation_text": recommendation, "integrated_report": report,
            "split": "",
        }

    def generate(self):
        rows: list[dict] = []
        index = 0
        while len(rows) < self.n_rows:
            rows.extend(self._simulate_patient(index))
            index += 1
        frame = pd.DataFrame(rows[: self.n_rows])
        frame = self._assign_splits(frame)
        frame = self._attach_gold_documents(frame)
        integer_like = ["interval_days_from_prior", "years_since_quit", "ecog_status", "volume_doubling_time_days",
                        "mitoses_per_2mm2", "ki67_index_pct", "nuclear_pleomorphism_score", "pdl1_tps_pct",
                        "wsi_image_seed"]
        for column in integer_like:
            frame[column] = frame[column].round().astype("Int64")
        LOGGER.info("Generated %d studies from %d patients", len(frame), frame["patient_id"].nunique())
        return frame[list(COLUMN_SPEC)]

    def _assign_splits(self, frame: pd.DataFrame):
        patients = frame["patient_id"].unique()
        order = self.rng.permutation(len(patients))
        n_train = int(0.8 * len(patients))
        n_val = int(0.1 * len(patients))
        mapping = {}
        for rank, idx in enumerate(order):
            mapping[patients[idx]] = "train" if rank < n_train else "validation" if rank < n_train + n_val else "test"
        frame["split"] = frame["patient_id"].map(mapping)
        return frame

    @staticmethod
    def _attach_gold_documents(frame: pd.DataFrame):
        from medfusion.data.knowledge import KB_PASSAGES
        by_rec: dict[str, list[str]] = {}
        for doc_id, _title, _family, rec_id, _tags, _body in KB_PASSAGES:
            if rec_id:
                by_rec.setdefault(rec_id, []).append(doc_id)
        frame["gold_guideline_doc_ids"] = frame["recommendation_id"].map(lambda r: "|".join(by_rec[r]))
        return frame


def data_dictionary():
    return pd.DataFrame([{"column": k, "type": v[0], "description": v[1]} for k, v in COLUMN_SPEC.items()])
