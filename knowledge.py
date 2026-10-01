"""Clinical knowledge base (for retrieval) and biomedical knowledge graph (for XAI).

All passages are original paraphrased summaries written for research and
education. They are inspired by the structure of widely used public frameworks
(Lung RADS version 2022, the British Thoracic Society 2015 pulmonary nodule
guideline, the Fleischner Society 2017 recommendations, the WHO 2021 thoracic
tumour classification and the eighth edition TNM staging system) but they are
simplified and must never be used for real clinical decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Canonical management recommendations. Every report recommendation in the
# synthetic dataset is copied verbatim from one of these sentences, which makes
# retrieval augmented generation measurable: the model must retrieve the right
# guidance passage and ground its recommendation in it.
RECOMMENDATIONS = {
    "R01": "Continue annual low dose CT screening in 12 months.",
    "R02": "No further imaging follow up is required for this finding.",
    "R03": "Arrange low dose CT in 6 months to assess stability.",
    "R04": "Arrange low dose CT in 3 months; PET CT may be considered if the solid component is 8 mm or larger.",
    "R05": "Arrange PET CT and image guided tissue sampling with lung multidisciplinary team review.",
    "R06": "Refer to the lung cancer multidisciplinary team for staging review and treatment planning.",
    "R07": "Benign histology is concordant with imaging; confirm stability with CT in 6 months.",
    "R08": "Urgent oncology referral is advised; complete staging with PET CT and brain MRI.",
    "R09": "The sample is nondiagnostic and discordant with suspicious imaging; repeat tissue sampling is advised.",
}

KB_PASSAGES = [
    # Management passages (each carries a recommendation identifier)
    ("KB001", "Lung RADS category 1 negative or benign appearance", "Lung RADS 2022 summary", "R01",
     ["lung_rads_1", "calcified_nodule", "screening"],
     "No lung nodule, or a nodule with a benign pattern of calcification or fat, is classed as category 1 in a screening round. The probability of malignancy is below one percent and routine annual screening continues."),
    ("KB002", "Lung RADS category 2 benign behaviour", "Lung RADS 2022 summary", "R01",
     ["lung_rads_2", "solid_nodule", "ground_glass_nodule", "screening"],
     "Small solid nodules below 6 mm at baseline, part solid nodules below 6 mm in total diameter and ground glass nodules below 30 mm are classed as category 2. Category 3 or 4A nodules that remain stable over the surveillance interval are also downgraded to category 2."),
    ("KB003", "Lung RADS category 3 probably benign", "Lung RADS 2022 summary", "R03",
     ["lung_rads_3", "solid_nodule", "part_solid_nodule"],
     "Solid nodules from 6 mm to below 8 mm at baseline, part solid nodules of 6 mm or more with a solid component below 6 mm, and ground glass nodules of 30 mm or more are classed as category 3. The estimated malignancy probability is one to two percent and short interval imaging is used to confirm stability."),
    ("KB004", "Lung RADS category 4A suspicious", "Lung RADS 2022 summary", "R04",
     ["lung_rads_4a", "solid_nodule", "part_solid_nodule", "action_ct_3_months"],
     "Solid nodules from 8 mm to below 15 mm at baseline, and part solid nodules whose solid component measures 6 mm to below 8 mm, are classed as category 4A with an estimated malignancy probability of five to fifteen percent. A growing nodule below 8 mm is also managed as category 4A."),
    ("KB005", "Lung RADS category 4B very suspicious", "Lung RADS 2022 summary", "R05",
     ["lung_rads_4b", "solid_nodule", "action_pet_ct", "action_tissue_sampling"],
     "Solid nodules of 15 mm or more, part solid nodules with a solid component of 8 mm or more, and nodules showing interval growth at 8 mm or above are classed as category 4B. Malignancy probability exceeds fifteen percent and diagnostic work up is indicated rather than surveillance."),
    ("KB006", "Lung RADS category 4X additional suspicious features", "Lung RADS 2022 summary", "R05",
     ["lung_rads_4x", "spiculation", "lymphadenopathy", "action_pet_ct", "action_tissue_sampling"],
     "Category 3 or 4 nodules with additional features that raise suspicion, such as spiculation, enlarged mediastinal lymph nodes or a rapidly enlarging solid component, are upgraded to category 4X. They should be managed with the same urgency as category 4B lesions."),
    ("KB007", "Incidental small solid nodule in a low risk patient", "Fleischner 2017 summary", "R02",
     ["solid_nodule", "incidental", "low_risk"],
     "For an incidentally detected solitary solid nodule below 6 mm in an adult without strong risk factors, routine imaging follow up is not required because the malignancy risk is well below one percent. Clinical judgement may still favour optional follow up in higher risk individuals."),
    ("KB008", "Benign calcification patterns", "Thoracic imaging teaching summary", "R02",
     ["calcified_nodule", "granuloma", "hamartoma", "benign"],
     "Diffuse, central, laminated and popcorn patterns of calcification indicate a benign nodule, most often a healed granuloma or a hamartoma. Nodules with these patterns do not need surveillance imaging."),
    ("KB009", "Very small nodules below 5 mm", "BTS 2015 summary", "R02",
     ["solid_nodule", "incidental", "low_risk"],
     "Nodules below 5 mm in maximum diameter or below 80 cubic millimetres in volume carry a very low malignancy risk and do not require further imaging follow up outside a screening programme."),
    ("KB010", "Brock risk of ten percent or more", "BTS 2015 summary", "R05",
     ["brock_high", "action_pet_ct", "action_tissue_sampling", "solid_nodule"],
     "When a nodule of 8 mm or more has a Brock model malignancy risk of ten percent or higher, PET CT is recommended. The combined Brock and Herder assessment then guides the choice between tissue sampling, surgical excision and continued surveillance."),
    ("KB011", "Brock risk below ten percent", "BTS 2015 summary", "R03",
     ["brock_low", "solid_nodule", "action_ct_surveillance"],
     "Nodules with a Brock risk below ten percent can be managed with CT surveillance. Volumetric measurement at three and twelve months allows the volume doubling time to be calculated and growth to be detected early."),
    ("KB012", "Confirmed non small cell lung cancer", "Lung cancer pathway summary", "R06",
     ["malignancy", "adenocarcinoma", "squamous_cell_carcinoma", "large_cell_carcinoma", "action_mdt"],
     "When histology confirms non small cell lung cancer, the case is discussed at the lung cancer multidisciplinary team. Staging completeness, performance status, lung function and biomarker results together determine whether surgery, radiotherapy or systemic therapy is offered."),
    ("KB013", "Small cell lung cancer pathway", "Lung cancer pathway summary", "R08",
     ["small_cell_carcinoma", "malignancy", "action_oncology_urgent", "high_ki67"],
     "Small cell lung cancer grows rapidly and disseminates early, so referral to oncology is urgent. Staging with PET CT and brain MRI is completed promptly because treatment is usually systemic chemotherapy combined with immunotherapy or radiotherapy."),
    ("KB014", "Benign concordant histology", "Radiology pathology correlation summary", "R07",
     ["benign", "granuloma", "hamartoma", "organising_pneumonia"],
     "A specific benign diagnosis such as granuloma, hamartoma or organising pneumonia that matches the imaging appearance is considered concordant. A short interval CT is still advised to confirm stability or resolution before discharge."),
    ("KB015", "Nondiagnostic or discordant sampling", "Radiology pathology correlation summary", "R09",
     ["nondiagnostic", "action_tissue_sampling", "malignancy"],
     "A biopsy that shows only nonspecific inflammation or normal lung in a nodule with suspicious imaging is regarded as nondiagnostic. Such discordance should not be taken as reassurance and repeat sampling or surgical excision is considered."),
    ("KB016", "Pulmonary carcinoid tumour", "WHO 2021 thoracic tumours summary", "R06",
     ["carcinoid_tumour", "malignancy", "low_ki67", "action_mdt"],
     "Typical carcinoid tumours show fewer than 2 mitoses per 2 square millimetres and no necrosis, while atypical carcinoids show 2 to 10 mitoses or focal necrosis. Surgical resection with lymph node assessment is the usual treatment after multidisciplinary review."),
    # Imaging feature passages
    ("KB017", "Spiculated margins", "Thoracic imaging teaching summary", None,
     ["spiculation", "malignancy", "adenocarcinoma"],
     "Spiculation describes linear strands radiating from the nodule margin. It reflects desmoplastic reaction and local infiltration and is one of the strongest morphological predictors of malignancy in the Brock model."),
    ("KB018", "Part solid nodules", "Thoracic imaging teaching summary", None,
     ["part_solid_nodule", "adenocarcinoma", "lepidic_pattern"],
     "Part solid nodules contain both ground glass and solid components. The size of the solid component correlates with the invasive part of an adenocarcinoma and drives both staging and management."),
    ("KB019", "Pure ground glass nodules", "Thoracic imaging teaching summary", None,
     ["ground_glass_nodule", "adenocarcinoma", "lepidic_pattern"],
     "Persistent pure ground glass nodules may represent atypical adenomatous hyperplasia, adenocarcinoma in situ or minimally invasive adenocarcinoma. They grow slowly and surveillance can extend to five years."),
    ("KB020", "Volume doubling time", "BTS 2015 summary", None,
     ["rapid_growth", "malignancy", "volume_doubling_time"],
     "A volume doubling time below 400 days in a solid nodule strongly suggests malignancy, values between 400 and 600 days are indeterminate, and stability over two years for solid nodules indicates benign behaviour. Subsolid lesions grow more slowly."),
    ("KB021", "Emphysema as a risk factor", "Lung cancer risk summary", None,
     ["emphysema", "malignancy", "smoking"],
     "Visual emphysema on CT is an independent risk factor for lung cancer beyond smoking exposure and is included as a variable in the Brock model."),
    ("KB022", "Upper lobe location", "Lung cancer risk summary", None,
     ["upper_lobe", "malignancy"],
     "Primary lung cancers occur more frequently in the upper lobes, and upper lobe location increases the predicted malignancy probability of an indeterminate nodule."),
    ("KB023", "Mediastinal lymphadenopathy", "TNM 8th edition summary", None,
     ["lymphadenopathy", "n_positive", "malignancy"],
     "Mediastinal lymph nodes above 10 mm in short axis are considered enlarged on CT. Ipsilateral hilar involvement corresponds to N1 and ipsilateral mediastinal involvement to N2, both requiring pathological confirmation by endobronchial ultrasound when it alters management."),
    ("KB024", "Pleural contact", "Thoracic imaging teaching summary", None,
     ["pleural_contact", "malignancy"],
     "A nodule abutting the pleura may indicate visceral pleural invasion, which upstages a small tumour to T2a. Pleural tags and retraction support the possibility of invasion."),
    ("KB025", "Number of nodules", "Lung cancer risk summary", None,
     ["multiple_nodules", "benign"],
     "In the Brock model a greater number of nodules slightly lowers the malignancy probability of each individual nodule, reflecting the frequency of benign granulomatous or inflammatory nodules."),
    ("KB026", "Brock model variables", "BTS 2015 summary", None,
     ["brock_high", "brock_low", "spiculation", "upper_lobe", "emphysema"],
     "The Brock model combines age, sex, family history of lung cancer, emphysema, nodule size, nodule type, upper lobe location, nodule count and spiculation to estimate malignancy probability for nodules detected on CT."),
    # Histopathology passages
    ("KB027", "Invasive adenocarcinoma grading", "WHO 2021 thoracic tumours summary", None,
     ["adenocarcinoma", "lepidic_pattern", "gland_formation", "egfr_mutation"],
     "Invasive nonmucinous adenocarcinoma is graded by predominant pattern and high grade components: lepidic predominant tumours are grade 1, acinar or papillary predominant tumours are grade 2, and solid or micropapillary rich tumours are grade 3."),
    ("KB028", "Squamous cell carcinoma", "WHO 2021 thoracic tumours summary", None,
     ["squamous_cell_carcinoma", "keratinisation", "smoking"],
     "Squamous cell carcinoma shows keratinisation, keratin pearls or intercellular bridges and is typically positive for p40. It is strongly associated with heavy smoking and often arises centrally."),
    ("KB029", "Small cell carcinoma morphology", "WHO 2021 thoracic tumours summary", None,
     ["small_cell_carcinoma", "high_ki67", "high_mitoses", "smoking"],
     "Small cell carcinoma consists of densely packed small cells with scant cytoplasm, finely granular chromatin and nuclear moulding. The mitotic rate is very high and the Ki67 index usually exceeds fifty percent."),
    ("KB030", "Large cell carcinoma", "WHO 2021 thoracic tumours summary", None,
     ["large_cell_carcinoma", "malignancy"],
     "Large cell carcinoma is a diagnosis of exclusion for undifferentiated non small cell carcinoma lacking glandular, squamous or neuroendocrine differentiation on morphology and immunohistochemistry."),
    ("KB031", "Neuroendocrine proliferation markers", "WHO 2021 thoracic tumours summary", None,
     ["carcinoid_tumour", "low_ki67", "high_ki67", "small_cell_carcinoma"],
     "Proliferation markers help separate neuroendocrine tumours: carcinoid tumours usually have a low Ki67 index, whereas small cell and large cell neuroendocrine carcinomas have very high proliferation."),
    ("KB032", "Granulomatous inflammation", "Thoracic pathology teaching summary", None,
     ["granuloma", "benign", "calcified_nodule"],
     "Granulomas are compact aggregates of epithelioid histiocytes, often with multinucleated giant cells and central necrosis, and commonly calcify as they heal. Infection should be excluded with special stains and culture."),
    ("KB033", "Pulmonary hamartoma", "Thoracic pathology teaching summary", None,
     ["hamartoma", "benign", "calcified_nodule"],
     "Hamartomas are benign mesenchymal tumours containing cartilage, fat and entrapped respiratory epithelium. Fat density or popcorn calcification on CT is characteristic."),
    ("KB034", "Organising pneumonia", "Thoracic pathology teaching summary", None,
     ["organising_pneumonia", "benign"],
     "Organising pneumonia shows intra alveolar plugs of granulation tissue and can mimic a nodule or mass. It often improves on follow up imaging or after corticosteroid treatment."),
    # Biomarker passages
    ("KB035", "EGFR mutation testing", "Biomarker testing summary", None,
     ["egfr_mutation", "adenocarcinoma", "never_smoker"],
     "EGFR activating mutations are most frequent in adenocarcinoma arising in never smokers and in women. Patients with sensitising mutations benefit from EGFR tyrosine kinase inhibitors."),
    ("KB036", "ALK rearrangement", "Biomarker testing summary", None,
     ["alk_rearrangement", "adenocarcinoma", "never_smoker"],
     "ALK rearrangements occur in a small proportion of adenocarcinomas, typically in younger never or light smokers, and predict response to ALK inhibitors."),
    ("KB037", "KRAS G12C mutation", "Biomarker testing summary", None,
     ["kras_g12c", "adenocarcinoma", "smoking"],
     "KRAS G12C mutations are associated with smoking related adenocarcinoma and are targetable with specific KRAS G12C inhibitors after first line therapy."),
    ("KB038", "PDL1 tumour proportion score", "Biomarker testing summary", None,
     ["pdl1_high", "action_mdt"],
     "The PDL1 tumour proportion score is reported as the percentage of viable tumour cells with membrane staining. A score of fifty percent or more identifies patients most likely to benefit from first line immunotherapy."),
    # Staging and protocol passages
    ("KB039", "T descriptor by tumour size", "TNM 8th edition summary", None,
     ["t_stage", "malignancy", "part_solid_nodule"],
     "T1a tumours measure 10 mm or less, T1b more than 10 mm up to 20 mm, T1c more than 20 mm up to 30 mm and T2a more than 30 mm up to 40 mm. For part solid tumours the solid component measures the invasive size used for T staging."),
    ("KB040", "Stage grouping", "TNM 8th edition summary", None,
     ["t_stage", "n_positive", "malignancy"],
     "Node negative T1a, T1b and T1c tumours correspond to stage IA1, IA2 and IA3, T2a to stage IB, and T2b to stage IIA. Hilar nodal disease with a T1 or T2 tumour gives stage IIB, ipsilateral mediastinal disease stage IIIA, and distant metastasis stage IV."),
    ("KB041", "Low dose CT protocol", "Screening protocol summary", None,
     ["screening", "radiation_dose"],
     "Screening examinations use low dose technique with a volume CT dose index typically below 3 mGy, thin reconstructed slices of 1.25 mm or less and no intravenous contrast, which supports accurate volumetry."),
    ("KB042", "Integrated radiology pathology review", "Radiology pathology correlation summary", None,
     ["action_mdt", "benign", "malignancy", "nondiagnostic"],
     "Every biopsy result should be correlated with the imaging appearance. Concordant malignant or specific benign results guide treatment, whereas discordant results prompt repeat sampling to avoid false reassurance."),
]

KG_NODES = [
    # node_id, label, node_type
    ("spiculation", "Spiculated margin", "imaging_finding"),
    ("lobulated_margin", "Lobulated margin", "imaging_finding"),
    ("solid_nodule", "Solid nodule", "imaging_finding"),
    ("part_solid_nodule", "Part solid nodule", "imaging_finding"),
    ("ground_glass_nodule", "Ground glass nodule", "imaging_finding"),
    ("calcified_nodule", "Calcified nodule", "imaging_finding"),
    ("upper_lobe", "Upper lobe location", "imaging_finding"),
    ("pleural_contact", "Pleural contact", "imaging_finding"),
    ("lymphadenopathy", "Mediastinal lymphadenopathy", "imaging_finding"),
    ("emphysema", "Emphysema", "imaging_finding"),
    ("rapid_growth", "Volume doubling time below 400 days", "imaging_finding"),
    ("large_nodule", "Nodule 15 mm or larger", "imaging_finding"),
    ("multiple_nodules", "Multiple nodules", "imaging_finding"),
    ("smoking", "Heavy smoking exposure", "risk_factor"),
    ("never_smoker", "Never smoker", "risk_factor"),
    ("family_history", "Family history of lung cancer", "risk_factor"),
    ("gland_formation", "Gland formation", "histology_feature"),
    ("lepidic_pattern", "Lepidic growth pattern", "histology_feature"),
    ("keratinisation", "Keratinisation", "histology_feature"),
    ("high_mitoses", "High mitotic count", "histology_feature"),
    ("high_ki67", "High Ki67 index", "histology_feature"),
    ("low_ki67", "Low Ki67 index", "histology_feature"),
    ("malignancy", "Primary lung malignancy", "diagnosis"),
    ("benign", "Benign process", "diagnosis"),
    ("adenocarcinoma", "Adenocarcinoma", "diagnosis"),
    ("squamous_cell_carcinoma", "Squamous cell carcinoma", "diagnosis"),
    ("small_cell_carcinoma", "Small cell carcinoma", "diagnosis"),
    ("large_cell_carcinoma", "Large cell carcinoma", "diagnosis"),
    ("carcinoid_tumour", "Carcinoid tumour", "diagnosis"),
    ("granuloma", "Granuloma", "diagnosis"),
    ("hamartoma", "Hamartoma", "diagnosis"),
    ("organising_pneumonia", "Organising pneumonia", "diagnosis"),
    ("nondiagnostic", "Nondiagnostic sample", "diagnosis"),
    ("egfr_mutation", "EGFR mutation", "biomarker"),
    ("alk_rearrangement", "ALK rearrangement", "biomarker"),
    ("kras_g12c", "KRAS G12C mutation", "biomarker"),
    ("pdl1_high", "PDL1 TPS 50 percent or more", "biomarker"),
    ("lung_rads_1", "Lung RADS 1", "risk_category"),
    ("lung_rads_2", "Lung RADS 2", "risk_category"),
    ("lung_rads_3", "Lung RADS 3", "risk_category"),
    ("lung_rads_4a", "Lung RADS 4A", "risk_category"),
    ("lung_rads_4b", "Lung RADS 4B", "risk_category"),
    ("lung_rads_4x", "Lung RADS 4X", "risk_category"),
    ("action_ct_surveillance", "CT surveillance", "action"),
    ("action_ct_3_months", "CT at 3 months", "action"),
    ("action_pet_ct", "PET CT", "action"),
    ("action_tissue_sampling", "Tissue sampling", "action"),
    ("action_mdt", "Multidisciplinary team review", "action"),
    ("action_oncology_urgent", "Urgent oncology referral", "action"),
]

KG_EDGES = [
    # source, target, relation, weight, evidence passage
    ("spiculation", "malignancy", "suggests", 0.85, "KB017"),
    ("lobulated_margin", "malignancy", "suggests", 0.45, "KB017"),
    ("part_solid_nodule", "adenocarcinoma", "suggests", 0.75, "KB018"),
    ("ground_glass_nodule", "adenocarcinoma", "suggests", 0.55, "KB019"),
    ("ground_glass_nodule", "lepidic_pattern", "correlates_with", 0.7, "KB019"),
    ("part_solid_nodule", "lepidic_pattern", "correlates_with", 0.6, "KB018"),
    ("calcified_nodule", "benign", "indicates", 0.95, "KB008"),
    ("calcified_nodule", "granuloma", "suggests", 0.7, "KB032"),
    ("calcified_nodule", "hamartoma", "suggests", 0.5, "KB033"),
    ("upper_lobe", "malignancy", "increases_risk_of", 0.5, "KB022"),
    ("emphysema", "malignancy", "increases_risk_of", 0.45, "KB021"),
    ("smoking", "malignancy", "increases_risk_of", 0.7, "KB021"),
    ("smoking", "squamous_cell_carcinoma", "associated_with", 0.7, "KB028"),
    ("smoking", "small_cell_carcinoma", "associated_with", 0.75, "KB029"),
    ("smoking", "kras_g12c", "associated_with", 0.5, "KB037"),
    ("never_smoker", "egfr_mutation", "associated_with", 0.6, "KB035"),
    ("never_smoker", "alk_rearrangement", "associated_with", 0.4, "KB036"),
    ("family_history", "malignancy", "increases_risk_of", 0.4, "KB026"),
    ("pleural_contact", "malignancy", "suggests", 0.45, "KB024"),
    ("lymphadenopathy", "malignancy", "suggests", 0.7, "KB023"),
    ("rapid_growth", "malignancy", "suggests", 0.85, "KB020"),
    ("large_nodule", "malignancy", "suggests", 0.7, "KB005"),
    ("multiple_nodules", "benign", "suggests", 0.3, "KB025"),
    ("gland_formation", "adenocarcinoma", "indicates", 0.9, "KB027"),
    ("lepidic_pattern", "adenocarcinoma", "indicates", 0.8, "KB027"),
    ("keratinisation", "squamous_cell_carcinoma", "indicates", 0.9, "KB028"),
    ("high_mitoses", "small_cell_carcinoma", "suggests", 0.75, "KB029"),
    ("high_ki67", "small_cell_carcinoma", "suggests", 0.8, "KB031"),
    ("low_ki67", "carcinoid_tumour", "suggests", 0.75, "KB031"),
    ("adenocarcinoma", "malignancy", "is_a", 1.0, "KB012"),
    ("squamous_cell_carcinoma", "malignancy", "is_a", 1.0, "KB012"),
    ("small_cell_carcinoma", "malignancy", "is_a", 1.0, "KB013"),
    ("large_cell_carcinoma", "malignancy", "is_a", 1.0, "KB030"),
    ("carcinoid_tumour", "malignancy", "is_a", 1.0, "KB016"),
    ("granuloma", "benign", "is_a", 1.0, "KB032"),
    ("hamartoma", "benign", "is_a", 1.0, "KB033"),
    ("organising_pneumonia", "benign", "is_a", 1.0, "KB034"),
    ("adenocarcinoma", "egfr_mutation", "tested_for", 0.8, "KB035"),
    ("adenocarcinoma", "alk_rearrangement", "tested_for", 0.8, "KB036"),
    ("adenocarcinoma", "kras_g12c", "tested_for", 0.7, "KB037"),
    ("malignancy", "pdl1_high", "tested_for", 0.6, "KB038"),
    ("malignancy", "action_mdt", "managed_by", 0.9, "KB012"),
    ("small_cell_carcinoma", "action_oncology_urgent", "managed_by", 0.95, "KB013"),
    ("nondiagnostic", "action_tissue_sampling", "managed_by", 0.9, "KB015"),
    ("benign", "action_ct_surveillance", "managed_by", 0.6, "KB014"),
    ("lung_rads_3", "action_ct_surveillance", "managed_by", 0.9, "KB003"),
    ("lung_rads_4a", "action_ct_3_months", "managed_by", 0.9, "KB004"),
    ("lung_rads_4b", "action_pet_ct", "managed_by", 0.9, "KB005"),
    ("lung_rads_4b", "action_tissue_sampling", "managed_by", 0.85, "KB005"),
    ("lung_rads_4x", "action_pet_ct", "managed_by", 0.9, "KB006"),
    ("lung_rads_4x", "action_tissue_sampling", "managed_by", 0.85, "KB006"),
    ("spiculation", "lung_rads_4x", "upgrades_to", 0.8, "KB006"),
    ("lymphadenopathy", "lung_rads_4x", "upgrades_to", 0.8, "KB006"),
    ("large_nodule", "lung_rads_4b", "classifies_as", 0.85, "KB005"),
    ("lung_rads_4b", "malignancy", "suggests", 0.6, "KB005"),
    ("lung_rads_4x", "malignancy", "suggests", 0.7, "KB006"),
    ("lung_rads_4a", "malignancy", "suggests", 0.35, "KB004"),
    ("lung_rads_1", "benign", "suggests", 0.9, "KB001"),
    ("lung_rads_2", "benign", "suggests", 0.85, "KB002"),
    ("lung_rads_3", "benign", "suggests", 0.7, "KB003"),
]


def kb_records():
    """Return knowledge base passages as dictionaries with full text."""
    records = []
    for doc_id, title, family, rec_id, tags, body in KB_PASSAGES:
        if rec_id:
            text = f"{title}. Recommendation: {RECOMMENDATIONS[rec_id]} Rationale: {body}"
        else:
            text = f"{title}. Key point: {body}"
        records.append({
            "doc_id": doc_id, "title": title, "source_family": family,
            "recommendation_id": rec_id or "", "kg_tags": tags, "text": text,
            "provenance": "Original synthetic paraphrase for research use only",
        })
    return records


def write_knowledge_assets(data_dir: Path):
    """Persist the knowledge base and knowledge graph to disk."""
    kb_dir = data_dir / "knowledge_base"
    kg_dir = data_dir / "knowledge_graph"
    kb_dir.mkdir(parents=True, exist_ok=True)
    kg_dir.mkdir(parents=True, exist_ok=True)
    records = kb_records()
    with open(kb_dir / "clinical_guidance_passages.jsonl", "w", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(kb_dir / "recommendations.json", "w", encoding="utf8") as handle:
        json.dump(RECOMMENDATIONS, handle, indent=2)
    pd.DataFrame(KG_NODES, columns=["node_id", "label", "node_type"]).to_csv(kg_dir / "kg_nodes.csv", index=False)
    pd.DataFrame(KG_EDGES, columns=["source", "target", "relation", "weight", "evidence_doc_id"]).to_csv(
        kg_dir / "kg_edges.csv", index=False)
    return {"passages": len(records), "nodes": len(KG_NODES), "edges": len(KG_EDGES)}
