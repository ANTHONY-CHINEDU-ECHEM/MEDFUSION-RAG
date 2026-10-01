# MedFusion RAG

**A retrieval augmented vision language model for integrated lung nodule radiology and histopathology reporting**

MedFusion RAG reads a CT key image, a histology whole slide image tile and the free text clinical note for a lung nodule study. It writes a complete integrated radiology and pathology report, answers clinical questions about the case, and backs every recommendation with retrieved guidance and an explainable chain of evidence. The repository contains everything needed to reproduce the work from scratch: a 16,000 study synthetic multimodal cohort, a curated clinical knowledge base and knowledge graph, the model, the training pipeline, a full evaluation suite and an explainability module.

> **Important.** Every patient, image, note, report and guidance passage in this repository is synthetic. The guidance passages are original paraphrased summaries written for research and education. Nothing here is a medical device, and nothing here should inform a real clinical decision.

## Project brief

Lung cancer remains the leading cause of cancer death in the United Kingdom, and outcomes depend heavily on the stage at diagnosis. Targeted lung health checks and incidental findings on routine CT now generate very large numbers of pulmonary nodules, almost all of them benign. Each one still has to be measured, categorised, risk assessed and given a management decision consistent with Lung RADS, British Thoracic Society or Fleischner guidance. When a nodule is sampled, the radiologist and pathologist must reconcile imaging and histology, decide whether the result is concordant, and route the patient into the correct pathway. Today this reasoning is spread across separate systems, separate reports and separate professionals. That fragmentation is a known source of delay and discordance.

Vision language models can, in principle, close that gap. A single model can look at the CT and the slide, read the referral note, and draft the integrated report a multidisciplinary team needs. In practice, two problems block adoption. First, generative models are fluent but not reliably grounded: a report can sound authoritative while recommending the wrong follow up interval. Second, clinicians cannot trust a recommendation they cannot interrogate. Attention heatmaps alone do not explain why a nodule was judged suspicious, nor which guideline the recommendation came from.

MedFusion RAG addresses both problems by design. A hierarchical vision transformer and a clinical text encoder are fused through gated bidirectional cross attention, giving a shared multimodal representation. Clinical heads read this representation to localise the nodule, estimate malignancy, assign a Lung RADS category and classify the histology. Those structured predictions then drive a hybrid retriever that fetches the relevant guidance passage. The language model decoder writes its report while attending to both the fused image and text memory and the retrieved passages. Because every recommendation in the training data is copied verbatim from a guidance passage, the benefit of retrieval can be measured directly rather than asserted.

Real paired CT, whole slide imaging, clinical notes and integrated reports at this scale are not openly available, and sharing them would raise serious governance issues. The project therefore begins by building a realistic synthetic cohort whose internal relationships follow published clinical knowledge: the Brock (PanCan) malignancy model, Lung RADS 2022 categories, eighth edition TNM staging, WHO 2021 histological grading and known biomarker epidemiology. The result is a fully reproducible, privacy safe testbed for multimodal clinical AI engineering. Every architectural and evaluation decision can be audited end to end, and the same code can later be pointed at governed real data.

## Highlights

* **16,000 study synthetic cohort, 73 columns.** Longitudinal (3,306 patients with repeat studies), multi site, with patient level splits, realistic missingness, growth and volume doubling time, histology, biomarkers, staging, ICD 10 coding, free text notes and integrated reports.
* **Deterministic image rendering.** CT key images (128 by 128) and H and E style histology tiles (64 by 64) are rendered on demand from the geometry and seed stored in each row. The CSV stays compact while the pixels remain causally tied to the labels.
* **Enhanced architecture.** Dual hierarchical MedViT encoders with feature pyramids, a BERT style text encoder, gated bidirectional cross attention fusion, multitask clinical heads with a FiLM conditioned localisation head, and a retrieval augmented autoregressive decoder.
* **Hybrid, graph aware RAG.** BM25, latent semantic retrieval and knowledge graph concept matching fused with reciprocal rank fusion, with slot based retrieval that always supplies an actionable guidance passage.
* **Instruction tuned clinical language model.** One model handles full report generation and eight families of clinical questions.
* **Explainability.** Localisation heatmaps, Grad CAM, decoder cross attention maps and knowledge graph evidence chains, each edge citing its supporting passage.
* **Rigorous evaluation.** Classification, calibration, localisation, retrieval, natural language and clinical efficacy metrics, a nearest neighbour baseline, a retrieval ablation and subgroup analysis.
* **Strict typographic policy.** No hyphen, minus sign, en dash or em dash appears anywhere in the repository, enforced by an automated audit and a unit test.

## Architecture

```
  CT key image           WSI tile              Clinical note (EHR)         Guidance knowledge base
  128 x 128 x 1          64 x 64 x 3           free text                   42 passages
       |                     |                       |                           |
       v                     v                       v                           |
 +===============+    +===============+    +==================+                  |
 | Hierarchical  |    | Hierarchical  |    | Text encoder     |                  |
 | MedViT (CT)   |    | MedViT (WSI)  |    | BERT style,      |                  |
 | conv stem     |    | conv stem     |    | segment aware    |                  |
 | ECB x2        |    | ECB x2        |    +==================+                  |
 | ECB + LTB     |    | ECB + LTB     |              |                           |
 | LTB x2        |    | LTB x2        |              |                           |
 | FPN P1 P2 P3  |    | FPN P1 P2 P3  |              |                           |
 +===============+    +===============+              |                           |
   |  128 tokens          |  32 tokens               |  note tokens              |
   |  + fine map P1       |                          |                           |
   v                      v                          v                           |
 +===========================================================================+   |
 |  Multimodal fusion                                                        |   |
 |  1. note tokens attend to image tokens       (gated cross attention)      |   |
 |  2. image tokens attend to note tokens       (gated cross attention)      |   |
 |  3. joint transformer over [FUSION CLS | CT | WSI | NOTE] + type embeds  |   |
 +===========================================================================+   |
          |                                    |                                 |
          v                                    v                                 |
 +=========================+        +========================+                   |
 | Clinical heads          |        | Fused multimodal       |                   |
 | localisation heatmap    |        | memory                 |                   |
 | presence, diameter      |        +========================+                   |
 | type, margin            |                   |                                 |
 | Lung RADS, malignancy   |                   |                                 |
 | lymph nodes, histology  |                   |                                 |
 +=========================+                   |                                 |
          |  structured findings               |                                 |
          v                                    |                                 |
 +=========================+                   |                                 |
 | Hybrid retriever        | <=================|=================================+
 | BM25 + LSA + graph, RRF |                   |
 +=========================+                   |
          |  top 3 passages (1 management slot)|
          v                                    v
 +===========================================================================+
 |  Retrieval augmented decoder (clinical language model)                    |
 |  causal self attention, cross attention over [fused memory | passages]    |
 |  instruction tokens: [TASK_REPORT] or [TASK_QA] + question                |
 +===========================================================================+
          |
          v
  Integrated report or answer   +   XAI: heatmap, Grad CAM, cross attention,
                                     knowledge graph evidence chains, citations
```

### Enhancements over the reference design

<table>
  <tr><th>Component</th><th>Reference diagram</th><th>This implementation</th></tr>
  <tr><td>Image encoder</td><td>Single ViT variant with feature pyramids</td><td>Two modality specific hierarchical MedViT encoders (CT and WSI). Efficient convolution blocks capture local texture, local transformer blocks with spatial reduction attention capture context, and a top down FPN merges three scales. Two pyramid levels are exposed as tokens with fixed 2D sinusoidal positions and learned level embeddings.</td></tr>
  <tr><td>Text encoder</td><td>BERT or RoBERTa backbone</td><td>Compact BERT style encoder with segment embeddings, shared between clinical notes, guidance passages and findings text. A clinical tokenizer splits numbers into digits so measurements generalise, and its round trip is lossless.</td></tr>
  <tr><td>Fusion</td><td>Cross attention feeding a decoder head</td><td>Bidirectional cross attention with learnable tanh gates (Flamingo style residual gating), followed by joint multimodal transformer blocks with modality type embeddings. Modality dropout during training makes the model robust to missing histology or notes.</td></tr>
  <tr><td>Decoder</td><td>Decoder head</td><td>Autoregressive clinical language model with tied embeddings, instruction tokens, and cross attention over fused memory plus retrieved passages. Generation uses key value caching, verified against full recomputation by a unit test.</td></tr>
  <tr><td>Knowledge</td><td>Knowledge graph and LLM pretraining for XAI</td><td>Retrieval augmented generation over a 42 passage guidance base, using hybrid lexical, semantic and graph ranking fused with reciprocal rank fusion. A 49 node, 60 edge biomedical knowledge graph supports graph aware retrieval and evidence chain explanations.</td></tr>
  <tr><td>Supervision</td><td>Report generation</td><td>Multitask learning: language modelling, CenterNet style focal heatmap loss, nodule presence, diameter, type, margin, Lung RADS, malignancy, lymphadenopathy, masked histology classification, and CLIP style image to findings contrastive alignment.</td></tr>
  <tr><td>Robustness</td><td>Not specified</td><td>Retrieval noise injection during training, passage dropout, modality dropout, patient level splits and subgroup evaluation by scanner vendor, sex and referral pathway.</td></tr>
  <tr><td>Explainability</td><td>Visual attention maps</td><td>Four complementary views: localisation heatmap, Grad CAM for malignancy, decoder cross attention while writing the findings, and strength ranked knowledge graph evidence chains with passage citations.</td></tr>
</table>

## The dataset

The cohort lives in `data/medfusion_lung_cohort.csv`: **16,000 rows and 73 columns** describing **12,351 synthetic patients** across eight NHS style acquisition sites. The full schema, missingness rates and distributions are in [`docs/dataset_card.md`](docs/dataset_card.md), and a machine readable dictionary is in `data/data_dictionary.csv`.

### How it was generated

The generator (`medfusion/data/generator.py`) simulates each patient as a timeline rather than as independent rows.

1. **Patient and referral.** Age, sex, smoking history, pack years, years since quitting, family history, COPD, emphysema, BMI, ECOG and comorbidities are sampled jointly with the referral pathway (screening programme, incidental finding, urgent suspected cancer). The draws are correlated: for example, emphysema depends on pack years and COPD.
2. **Acquisition.** Each site has a scanner vendor. Screening and surveillance studies use low dose protocols (CTDIvol around 1.6 mGy, no contrast), while diagnostic studies use higher dose and frequent contrast. Image noise in the rendered key image scales with dose.
3. **Nodule morphology.** Type (solid, part solid, ground glass, calcified), diameter, solid component, lobe, position, margin, calcification pattern, count and pleural contact. Spiculation probability rises with size.
4. **Malignancy.** The published Brock (PanCan) full model linear predictor is computed from the simulated features. A documented enrichment offset raises prevalence so the model sees enough positive cases (34.7 percent of nodules are malignant).
5. **Longitudinal follow up.** Category 3 and 4A nodules usually return for surveillance at 6 or 3 months. Malignant nodules grow according to a sampled volume doubling time, while benign nodules stay stable or regress. Growth upgrades the category and stability downgrades it, following simplified Lung RADS rules.
6. **Tissue sampling.** Category 4B and 4X lesions and growing nodules are usually sampled. Histology follows WHO 2021 categories with smoking and morphology dependent subtype probabilities, IASLC style adenocarcinoma grading, mitotic count, Ki67, PDL1 TPS, and EGFR, ALK and KRAS G12C prevalence by smoking status and sex. About 4 percent of malignant biopsies are deliberately nondiagnostic.
7. **Staging and coding.** Eighth edition TNM descriptors (invasive size, pleural invasion, nodal and metastatic disease) and stage groups, ICD 10 codes and the reference standard used for the label.
8. **Text.** Clinical notes are written in UK NHS referral style with varied templates and abbreviations (PMH, FHx, SOB, ECOG). The integrated report has four sections: FINDINGS, PATHOLOGY, IMPRESSION and RECOMMENDATION. The recommendation is copied verbatim from the guidance passage chosen by a deterministic management policy, and the supporting passage identifiers are stored in `gold_guideline_doc_ids`.

### Selected columns

<table>
  <tr><th>Group</th><th>Columns</th></tr>
  <tr><td>Identity and timeline</td><td>case_id, patient_id, study_sequence, study_date, interval_days_from_prior, split</td></tr>
  <tr><td>Patient and EHR</td><td>age_years, sex, smoking_status, pack_years, years_since_quit, family_history_lung_cancer, copd_diagnosis, prior_extrathoracic_cancer, bmi, ecog_status, presenting_symptom, referral_pathway</td></tr>
  <tr><td>Acquisition</td><td>site_id, region, scanner_vendor, slice_thickness_mm, tube_voltage_kvp, ctdivol_mgy, contrast_enhanced, ct_image_seed</td></tr>
  <tr><td>Imaging findings</td><td>nodule_present, nodule_count, dominant_nodule_lobe, nodule_x_norm, nodule_y_norm, nodule_diameter_mm, solid_component_mm, nodule_volume_mm3, nodule_type, margin, calcification_pattern, pleural_contact, mediastinal_lymphadenopathy, emphysema_severity, prior_diameter_mm, growth_status, volume_doubling_time_days</td></tr>
  <tr><td>Risk and triage</td><td>brock_risk_pct, lung_rads_category, malignancy_label, reference_standard</td></tr>
  <tr><td>Pathology</td><td>histology_available, sampling_procedure, histology_subtype, adenocarcinoma_pattern, tumour_grade, mitoses_per_2mm2, ki67_index_pct, tumour_cellularity_pct, nuclear_pleomorphism_score, necrosis_present, pdl1_tps_pct, molecular_driver, wsi_image_seed</td></tr>
  <tr><td>Staging and coding</td><td>t_stage, n_stage, m_stage, clinical_stage, icd10_code</td></tr>
  <tr><td>Text and grounding</td><td>clinical_note, findings_text, pathology_text, impression_text, recommendation_text, integrated_report, recommendation_id, gold_guideline_doc_ids</td></tr>
</table>

### Example record

**Clinical note (model input).** Referral from the nodule surveillance programme: 66 year old male. No smoking history. No significant past medical history. Asymptomatic. ECOG 1. Prior CT 3 months ago showed a 14 mm solid nodule in the right lower lobe. Please assess and advise on management.

**Integrated report (generation target).** FINDINGS: A solid nodule in the right lower lobe measures 16 mm with smooth margins. Previously 14 mm, in keeping with interval growth; estimated volume doubling time 142 days. One additional small nodule is noted, below 4 mm. Mild centrilobular emphysema. No mediastinal or hilar lymphadenopathy. PATHOLOGY: CT guided core biopsy shows adenocarcinoma, solid predominant, grade 3. Ki67 43%. PDL1 TPS 0%. Molecular testing: no driver detected. IMPRESSION: Lung RADS 4B. Highly suspicious nodule, concerning for primary lung malignancy. Histology confirms adenocarcinoma, clinical stage IV (T1b N0 M1). RECOMMENDATION: Refer to the lung cancer multidisciplinary team for staging review and treatment planning.

### Rendered images

![Rendered examples](docs/figures/rendered_examples.png)

The CT renderer draws the body outline, lung fields, mediastinum, vertebra, vascular markings that act as realistic nodule mimics, emphysematous low attenuation areas, enlarged mediastinal nodes, satellite nodules and the dominant nodule. The dominant nodule is drawn with type specific appearance (solid, part solid with a dense core, ground glass, calcified) and margin specific shape (smooth, lobulated, spiculated). The histology renderer uses an H and E colour model with subtype specific architecture: glands with lumens for adenocarcinoma, keratin pearls for squamous carcinoma, dense small nuclei for small cell carcinoma, granulomas with optional necrosis, and cartilage islands for hamartoma. Nuclear size and density follow the recorded pleomorphism, cellularity and mitotic count.

## Knowledge base and knowledge graph

`data/knowledge_base/clinical_guidance_passages.jsonl` holds 42 original passages across seven source families: Lung RADS 2022, BTS 2015, Fleischner 2017, WHO 2021 thoracic tumours, TNM eighth edition, biomarker testing, and radiology pathology correlation. Sixteen passages are management passages carrying one of nine canonical recommendations (R01 to R09). The rest cover imaging features, histology, biomarkers and staging, and give the decoder supporting evidence.

`data/knowledge_graph/` holds 49 typed nodes (imaging findings, risk factors, histology features, diagnoses, biomarkers, risk categories, actions) and 60 weighted, typed edges such as *suggests*, *indicates*, *increases risk of*, *managed by* and *upgrades to*. Every edge cites the passage that justifies it.

### Retrieval design

Retrieval is built from the case findings, not from the raw note. A confirmed tissue diagnosis dominates the query. Without one, screening and higher risk nodules are framed by their Lung RADS category, and low risk incidental nodules by size and morphology. Three rankers score every passage:

* **BM25**, implemented from first principles;
* **latent semantic analysis**, a dense, offline semantic ranker (TF IDF followed by truncated SVD);
* a **graph ranker** that matches case concepts, expanded one hop through the knowledge graph, against passage concept tags.

Reciprocal rank fusion combines the three rankings. Slot based selection then guarantees that the top ranked management passage always occupies the first slot.

<table>
  <tr><th>Ranker (reference findings, test split)</th><th>Recall at 1</th></tr>
  <tr><td>BM25 only</td><td>80.5%</td></tr>
  <tr><td>Latent semantic only</td><td>75.3%</td></tr>
  <tr><td>Knowledge graph only</td><td>99.7%</td></tr>
  <tr><td>Hybrid reciprocal rank fusion (used)</td><td>98.6%</td></tr>
</table>

The graph ranker alone scores slightly higher than the hybrid on this benchmark. That is expected: the queries are built from the same structured findings that the passage tags describe, so concept matching is almost perfect. The hybrid is still the deployed choice for robustness. Lexical and semantic ranking keep working when a query contains concepts that have no graph tag, when tags are incomplete, or when free text is added to the query, and fusion costs just over one point here. The ablation configuration in `configs/default.json` (`retrieval.weights`) lets the graph only setting be selected with no code change.

## Training

The model has **3.03 million parameters** and was trained from scratch on CPU for 4 epochs (1,600 optimisation steps, batch size 32, about 85 minutes on a single core). It uses AdamW with linear warmup, cosine decay and gradient clipping.

Every training example pairs the images and note with one instruction:

* `[TASK_REPORT]` asks for the full integrated report (55 percent of examples);
* `[TASK_QA]` asks one of eight questions covering Lung RADS category, nodule location, size, morphology, lymphadenopathy, histology, impression and next step.

The loss is computed on answer tokens only, with label smoothing.

The total objective weights language modelling, localisation focal loss, the eight clinical heads and contrastive image to findings alignment. Three regularisers target the failure modes that matter in deployment:

* **Retrieval noise.** 15 percent of training cases receive passages retrieved from perturbed findings, so the decoder learns to use guidance without blindly copying it.
* **Passage dropout.** 10 percent of passages are hidden, so the model degrades gracefully if retrieval fails.
* **Modality dropout.** 10 percent of cases lose their histology or note, so the model copes with incomplete records.

Training language modelling loss is higher than validation loss because training uses label smoothing, dropout, modality dropout, passage dropout and noisy retrieval, while validation uses none of them.

<table>
  <tr><th>Epoch</th><th>Train LM loss</th><th>Validation LM loss</th><th>Validation total loss</th></tr>
  <tr><td>1</td><td>2.430</td><td>0.305</td><td>4.166</td></tr>
  <tr><td>2</td><td>0.776</td><td>0.222</td><td>3.434</td></tr>
  <tr><td>3</td><td>0.722</td><td>0.188</td><td>2.825</td></tr>
  <tr><td>4</td><td>0.701</td><td>0.180</td><td>2.736</td></tr>
</table>

![Training curves](outputs/figures/training_curves.png)

## Results

All results are on the held out test split: **1,598 studies from 1,236 patients**, none of whom appear in training or validation.

### Perception and risk

<table>
  <tr><th>Task</th><th>Metric</th><th>Result</th></tr>
  <tr><td>Malignancy (all studies)</td><td>AUROC</td><td>0.883</td></tr>
  <tr><td>Malignancy (all studies)</td><td>AUPRC</td><td>0.832</td></tr>
  <tr><td>Malignancy (all studies)</td><td>Sensitivity / specificity at 0.5</td><td>61.3% / 97.0%</td></tr>
  <tr><td>Malignancy (all studies)</td><td>Brier score / ECE</td><td>0.097 / 0.018</td></tr>
  <tr><td>Malignancy (nodule studies)</td><td>AUROC (MedFusion, images and note)</td><td>0.875</td></tr>
  <tr><td>Malignancy (nodule studies)</td><td>AUROC (Brock reference, true measured features)</td><td>0.864</td></tr>
  <tr><td>Nodule presence</td><td>AUROC</td><td>0.963</td></tr>
  <tr><td>Lung RADS category (6 classes)</td><td>Accuracy / within one category</td><td>74.5% / 94.2%</td></tr>
  <tr><td>Lung RADS category (6 classes)</td><td>Macro F1</td><td>0.717</td></tr>
  <tr><td>Nodule type (5 classes)</td><td>Accuracy / macro F1</td><td>72.8% / 0.431</td></tr>
  <tr><td>Margin (4 classes)</td><td>Accuracy / macro F1</td><td>65.2% / 0.511</td></tr>
  <tr><td>Mediastinal lymphadenopathy</td><td>AUROC</td><td>0.996</td></tr>
  <tr><td>Histology subtype (9 classes, 292 sampled studies)</td><td>Accuracy / macro F1</td><td>87.0% / 0.539</td></tr>
  <tr><td>Localisation</td><td>Median peak error</td><td>4.3 mm</td></tr>
  <tr><td>Localisation</td><td>Hit within 10 mm (all / 8 mm and larger / below 6 mm)</td><td>91.5% / 93.8% / 88.4%</td></tr>
  <tr><td>Guidance retrieval (predicted findings)</td><td>Recall at 1 / MRR</td><td>87.0% / 0.892</td></tr>
  <tr><td>Guidance retrieval (reference findings)</td><td>Recall at 1 / MRR</td><td>98.6% / 0.987</td></tr>
</table>

Three findings stand out. First, on nodule studies the model reaches an AUROC of 0.875 from pixels and free text alone. That is slightly above the Brock reference (0.864), which is computed from the true, perfectly measured features that generated the labels, so the network has learned to recover clinically meaningful risk signals from the raw inputs. Second, the malignancy probabilities are well calibrated (ECE 0.018 across all studies), so the operating threshold can be chosen deliberately. At the default threshold of 0.5 the model favours specificity (96.8 percent on nodule studies) over sensitivity (61.3 percent), and a screening deployment would lower the threshold. Third, the weaknesses are informative. Morphology distinctions that are subtle on a 128 pixel key image (ground glass versus part solid, lobulated versus smooth margins) have low balanced accuracy. Rare histology classes pull the macro F1 down even though overall histology accuracy is 87.0 percent.

![Test performance](outputs/figures/test_performance.png)

### Report generation

Generated reports are scored with standard language metrics and with **clinical efficacy**: the clinically important fields are parsed from generated and reference reports and compared field by field. A model can score well on BLEU yet get the Lung RADS category or the recommendation wrong, so clinical efficacy is the more meaningful measure. The baseline copies the report of the most similar training case, matched by TF IDF similarity of the clinical notes. This classic retrieval baseline is strong on fluency but blind to the images.

<table>
  <tr><th>Metric (800 test reports)</th><th>Nearest neighbour baseline</th><th>MedFusion RAG</th></tr>
  <tr><td>BLEU 1</td><td>0.617</td><td><b>0.822</b></td></tr>
  <tr><td>BLEU 4</td><td>0.439</td><td><b>0.687</b></td></tr>
  <tr><td>ROUGE L</td><td>0.599</td><td><b>0.808</b></td></tr>
  <tr><td>Clinical efficacy (macro over 10 fields)</td><td>53.4%</td><td><b>75.1%</b></td></tr>
  <tr><td>Lung RADS category correct</td><td>35.4%</td><td><b>74.9%</b></td></tr>
  <tr><td>Nodule presence correct</td><td>76.0%</td><td><b>91.8%</b></td></tr>
  <tr><td>Lobe correct</td><td>27.1%</td><td><b>37.2%</b></td></tr>
  <tr><td>Nodule type correct</td><td>42.2%</td><td><b>72.8%</b></td></tr>
  <tr><td>Margin correct</td><td>37.5%</td><td><b>60.5%</b></td></tr>
  <tr><td>Lymphadenopathy correct</td><td>84.6%</td><td><b>99.1%</b></td></tr>
  <tr><td>Emphysema grade correct</td><td><b>38.0%</b></td><td>34.8%</td></tr>
  <tr><td>Histology correct</td><td>71.1%</td><td><b>97.6%</b></td></tr>
  <tr><td>Clinical stage correct</td><td>77.6%</td><td><b>94.1%</b></td></tr>
  <tr><td>Recommendation correct</td><td>44.5%</td><td><b>88.2%</b></td></tr>
  <tr><td>Size within 2 mm</td><td>39.1%</td><td><b>70.4%</b></td></tr>
  <tr><td>Size MAE (mm)</td><td>5.08</td><td><b>2.57</b></td></tr>
</table>

MedFusion RAG beats the image blind baseline on fifteen of the sixteen measures. The clinically decisive fields improve the most: recommendation accuracy rises from 44.5 to 88.3 percent, Lung RADS category from 35.4 to 74.9 percent, histology from 71.1 to 97.6 percent, and size within 2 mm from 39.1 to 70.4 percent. Two weaknesses remain and are the priority for the next iteration. The lobe is correct in only 37.3 percent of reports, and the emphysema grade in only 34.8 percent. When image evidence is weak, the decoder tends to fall back on the most frequent phrasing (for example the right upper lobe and a 10 mm nodule). This happens even though the localisation head places its peak within 10 mm of the true nodule in 91.5 percent of cases. In other words, the information exists inside the network but is not yet routed into the text. Feeding the head outputs into the decoder as a structured findings prompt is the planned fix (see the roadmap).

### Does retrieval help?

The same 300 test cases were decoded three ways: with no retrieved passages, with passages retrieved from the model's own predicted findings (the deployed configuration), and with passages retrieved from the reference findings (an upper bound that isolates the retriever's contribution).

<table>
  <tr><th>Metric</th><th>No retrieval</th><th>Predicted findings retrieval</th><th>Oracle findings retrieval</th></tr>
  <tr><td>Recommendation correct</td><td>62.0%</td><td>88.3%</td><td>90.7%</td></tr>
  <tr><td>Lung RADS category correct</td><td>71.0%</td><td>77.0%</td><td>85.0%</td></tr>
  <tr><td>Clinical efficacy (macro)</td><td>70.9%</td><td>75.0%</td><td>78.5%</td></tr>
  <tr><td>BLEU 4</td><td>0.617</td><td>0.689</td><td>0.701</td></tr>
  <tr><td>ROUGE L</td><td>0.736</td><td>0.810</td><td>0.830</td></tr>
</table>

Retrieval is the single largest contributor to getting the management decision right. Without passages, the decoder must reconstruct the recommendation from its own parameters and is correct 62.0 percent of the time. With passages retrieved from its own predicted findings, this rises to 88.3 percent, a gain of 26.3 points; perfect findings would add only a further 2.3 points. Retrieval also makes the stated Lung RADS category more consistent (71.0 to 77.0 percent), because the impression and the retrieved category passage reinforce each other. On the full test split, guidance recall at 1 is 87.0 percent from predicted findings against 98.6 percent from reference findings. That gap shows the remaining errors begin upstream in perception, not in the retriever.

### Clinical question answering

<table>
  <tr><th>Question family (400 cases each)</th><th>Exact match</th></tr>
  <tr><td>Lung RADS category</td><td>77.0%</td></tr>
  <tr><td>Nodule location</td><td>37.5%</td></tr>
  <tr><td>Nodule size</td><td>32.2%</td></tr>
  <tr><td>Morphology</td><td>51.7%</td></tr>
  <tr><td>Lymphadenopathy</td><td>98.8%</td></tr>
  <tr><td>Recommended next step</td><td>87.0%</td></tr>
  <tr><td><b>Mean</b></td><td>64.0%</td></tr>
</table>

Question answering mirrors the report results. Questions answered from strong signals are reliable: lymphadenopathy (98.8 percent), next step (87.0 percent) and Lung RADS category (77.0 percent). Exact location and size are the weakest, for the routing reason described above. Size is scored by exact match, so answering 9 mm for a 10 mm nodule counts as wrong.

### Subgroup analysis (malignancy AUROC, nodule studies)

<table>
  <tr><th>Attribute</th><th>Group</th><th>Studies</th><th>AUROC</th></tr>
  <tr><td>scanner vendor</td><td>Canon Medical</td><td>271</td><td>0.807</td></tr>
  <tr><td>scanner vendor</td><td>GE HealthCare</td><td>390</td><td>0.899</td></tr>
  <tr><td>scanner vendor</td><td>Philips</td><td>298</td><td>0.914</td></tr>
  <tr><td>scanner vendor</td><td>Siemens Healthineers</td><td>369</td><td>0.862</td></tr>
  <tr><td>sex</td><td>Female</td><td>636</td><td>0.865</td></tr>
  <tr><td>sex</td><td>Male</td><td>692</td><td>0.886</td></tr>
  <tr><td>referral pathway</td><td>incidental finding</td><td>342</td><td>0.874</td></tr>
  <tr><td>referral pathway</td><td>screening programme</td><td>497</td><td>0.828</td></tr>
  <tr><td>referral pathway</td><td>surveillance</td><td>333</td><td>0.882</td></tr>
  <tr><td>referral pathway</td><td>urgent suspected cancer</td><td>156</td><td>0.935</td></tr>
</table>

### Sample generated report

A near exact output (only the emphysema grade differs):

**MF000374 reference.** FINDINGS: No suspicious pulmonary nodule is seen. Severe centrilobular emphysema. No mediastinal or hilar lymphadenopathy. PATHOLOGY: No histology available. IMPRESSION: Lung RADS 1. No suspicious pulmonary nodule. RECOMMENDATION: No further imaging follow up is required for this finding.

**MF000374 generated** (retrieved KB007, KB009, KB001; malignancy probability 0.11). FINDINGS: No suspicious pulmonary nodule is seen. No emphysema. No mediastinal or hilar lymphadenopathy. PATHOLOGY: No histology available. IMPRESSION: Lung RADS 1. No suspicious pulmonary nodule. RECOMMENDATION: No further imaging follow up is required for this finding.

A typical imperfect output, shown for balance:

**MF000107 reference.** FINDINGS: A solid nodule in the left lower lobe measures 9 mm with spiculated margins. One additional small nodule is noted, below 4 mm. Severe centrilobular emphysema. No mediastinal or hilar lymphadenopathy. PATHOLOGY: Surgical wedge resection shows granulomatous inflammation with no evidence of malignancy. IMPRESSION: Lung RADS 4X. Highly suspicious nodule, concerning for primary lung malignancy. Histology confirms benign granuloma. RECOMMENDATION: Benign histology is concordant with imaging; confirm stability with CT in 6 months.

**MF000107 generated** (retrieved KB014, KB008, KB017; malignancy probability 0.02). FINDINGS: A solid nodule in the right upper lobe measures 10 mm with spiculated margins. One additional small nodule is noted, below 4 mm. No emphysema. No mediastinal or hilar lymphadenopathy. PATHOLOGY: CT guided core biopsy shows granulomatous inflammation with no evidence of malignancy. IMPRESSION: Lung RADS 4X. Highly suspicious nodule, concerning for primary lung malignancy. Histology confirms benign granuloma. RECOMMENDATION: Benign histology is concordant with imaging; confirm stability with CT in 6 months.

In the second case the model correctly reads the histology tile as granulomatous inflammation and assigns a very low malignancy probability (0.02). It keeps the Lung RADS 4X category that the spiculated imaging warrants, recognises the result as concordant benign histology and retrieves the matching guidance (KB014), so the recommendation is exactly right. However, the lobe, size, sampling procedure and emphysema grade are wrong. This is the characteristic error pattern discussed above: decision level content is reliable, while fine descriptive detail still falls back on common phrasing.

## Explainability

`python run.py explain` writes a four view panel, a JSON record and a Markdown summary for a diverse set of test cases to `outputs/explanations/`.

![Explanation example](outputs/explanations/MF000359.png)

The panel above shows test case MF000359, a spiculated adenocarcinoma, through five views.

* **Reference CT.** The green ring marks the true nodule.
* **Localisation head.** It fires exactly on the nodule (peak confidence 1.00).
* **Grad CAM.** For the malignancy logit (probability 0.99), it concentrates on the nodule and its immediate surroundings.
* **Decoder cross attention.** While writing the findings, the decoder attends mostly to the lower left hemithorax region that contains the lesion.
* **Histology tile.** It is correctly classified as adenocarcinoma.

The knowledge graph module then explains the conclusion with ranked evidence chains, each citing its source passage:

* spiculated margin suggests primary lung malignancy (strength 0.85, KB017);
* Lung RADS 4X suggests malignancy (0.70, KB006);
* upper lobe location increases risk (0.50, KB022);
* family history increases risk (0.40, KB026).

The recommendation is grounded in the retrieved passage KB012 (confirmed non small cell lung cancer).

The same case shows why explanations matter. The generated report calls the lesion a 10 mm right upper lobe nodule, while the image and the localisation head clearly show a larger lesion in the left lower lobe. Because the panel puts these side by side, a reviewer can catch the discrepancy at a glance. The full summary for all six explained cases is in [`outputs/explanations/explanations_summary.md`](outputs/explanations/explanations_summary.md).

## Quick start

```bash
python tools/install_requirements.py
python run.py all
```

The first command installs everything listed in `requirements.txt` through pip, and `run.py all` runs every stage in order:

<table>
  <tr><th>Command</th><th>What it does</th></tr>
  <tr><td><code>python run.py data</code></td><td>Generates the cohort, data dictionary, knowledge base and knowledge graph</td></tr>
  <tr><td><code>python run.py prepare</code></td><td>Fits the tokenizer, renders and caches all images, precomputes retrieval</td></tr>
  <tr><td><code>python run.py train</code></td><td>Trains MedFusion VLM and saves the best checkpoint</td></tr>
  <tr><td><code>python run.py evaluate</code></td><td>Runs the full test evaluation, baselines, ablations and figures</td></tr>
  <tr><td><code>python run.py explain</code></td><td>Produces XAI panels and evidence chains</td></tr>
  <tr><td><code>python run.py demo</code></td><td>Runs the clinical assistant, including a never seen study rendered on demand</td></tr>
  <tr><td><code>python run.py audit</code></td><td>Verifies that no hyphen or dash character exists anywhere</td></tr>
  <tr><td><code>python tools/build_dataset_card.py</code></td><td>Regenerates the dataset card</td></tr>
  <tr><td><code>pytest tests</code></td><td>Runs the test suite</td></tr>
</table>

Configuration lives in `configs/default.json`. Any value can be overridden with dotted `key=value` pairs, for example `python run.py train training.epochs=8 model.d_model=192`.

### Using the assistant in code

```python
from medfusion.utils import load_config
from medfusion.inference.assistant import MedFusionAssistant

assistant = MedFusionAssistant(load_config())
case = "MF000058"
print(assistant.analyse(case)["report"])
print(assistant.ask(case, "What should happen next?")["answer"])
```

The demo transcript is in [`outputs/demo_transcript.md`](outputs/demo_transcript.md).

## Repository structure

```
configs/default.json              all hyperparameters and paths
data/
  medfusion_lung_cohort.csv       16,000 by 73 synthetic cohort
  data_dictionary.csv             column definitions
  knowledge_base/                 guidance passages and canonical recommendations
  knowledge_graph/                typed nodes and evidence cited edges
docs/
  dataset_card.md                 schema, missingness and distributions
  figures/                        rendered image examples
medfusion/
  data/generator.py               longitudinal cohort simulator
  data/renderer.py                deterministic CT and histology rendering
  data/knowledge.py               knowledge base and knowledge graph content
  data/tokenizer.py               clinical tokenizer with lossless round trip
  data/dataset.py                 asset preparation and instruction dataset
  models/image_encoder.py         hierarchical MedViT with feature pyramid
  models/components.py            text encoder, fusion, heads, cached decoder
  models/vlm.py                   full retrieval augmented VLM
  rag/retriever.py                BM25, LSA and graph rankers with RRF
  rag/graph.py                    knowledge graph and evidence paths
  training/losses.py              multitask objective
  training/trainer.py             training loop and checkpointing
  inference/engine.py             two pass retrieval augmented inference
  inference/assistant.py          clinical assistant and demo
  evaluation/metrics.py           BLEU, ROUGE L, clinical efficacy and more
  evaluation/evaluate.py          full evaluation suite
  xai/explain.py                  heatmaps, Grad CAM, attention, evidence chains
outputs/
  artifacts/                      trained checkpoint and tokenizer
  metrics/                        test metrics, training history, samples
  figures/                        evaluation and training figures
  explanations/                   XAI panels and summaries
tests/                            data, retrieval, model and policy tests
tools/
  install_requirements.py         installs requirements.txt through pip
  hyphen_audit.py                 repository wide typographic audit
  build_dataset_card.py           dataset card generator
run.py                            command line entry point
```

## Engineering notes

**No hyphen policy.** No hyphen, minus sign, en dash or em dash appears anywhere: not in code, data, configuration, documentation, file names or generated outputs. `tools/hyphen_audit.py` declares the forbidden characters by code point and scans every text artefact, and a unit test fails the build on any violation. Inside the code the policy is honoured without loss of clarity:

* subtraction uses `operator.sub`, `torch.sub`, `torch.rsub` or `numpy.subtract`;
* reverse indexing uses the bitwise complement (`x[~0]` is the last element, `dim=~0` the last axis);
* small constants are written as decimals or reciprocals;
* dates use slashes;
* regular expression ranges are spelled out from `string.ascii_lowercase`;
* return type arrows are omitted;
* JSON outputs are rounded so scientific notation and negative zero never appear;
* Markdown tables are written in HTML.

**Reproducibility.** The cohort, renders, splits, retrieval assignments and training are all seeded. The CSV stores render seeds rather than pixels, so the 16,000 images are bit for bit reproducible from the 22 MB CSV.

**Leakage control.** Splits are assigned per patient, so repeat studies never straddle splits. Validation uses retrieval from reference findings, while test evaluation uses retrieval driven only by the model's own predictions. The tokenizer is fitted on training text and the knowledge base only.

**Tests.** The 13 tests in `tests/` cover:

* dataset shape and schema;
* patient level leakage;
* clinical consistency rules;
* grounding of every recommendation in the knowledge base;
* generator determinism and the Lung RADS rules;
* the lossless tokenizer round trip and hybrid retrieval recall;
* metric sanity checks;
* a full forward and backward pass;
* exact equivalence of cached and uncached decoding;
* the repository wide hyphen audit.

All 13 pass.

## Limitations and responsible use

* **Synthetic realism has limits.** The images are stylised 2D key images and tiles, not volumetric CT or gigapixel slides. Relationships follow published knowledge at a coarse level, so performance here says nothing about performance on real patients.
* **Templated language.** Notes and reports come from varied but finite templates, so language metrics are higher than real world report generation would achieve. Clinical efficacy and the retrieval ablation are the more informative results.
* **Simplified guidance.** The management policy compresses Lung RADS, BTS and Fleischner logic into nine recommendations. Real guidance has more nuance, including Herder scoring, patient preference and comorbidity.
* **Enriched prevalence.** Malignancy is enriched relative to screening populations, so predictive values and calibration would need recalibration for any target population.
* **Compute constrained.** The model was trained on a single CPU core. A larger model, longer training and pretrained encoders would be expected to improve perception in particular.
* **Not a medical device.** Any real deployment would require governed data, prospective validation, human oversight, bias auditing across demographic groups and regulatory approval.

## Roadmap

* Volumetric 3D encoders with nodule level tokens and multi instance learning over whole slides.
* Pretrained initialisation from biomedical vision language checkpoints, with parameter efficient fine tuning.
* Feed the clinical head outputs (peak location, lobe, diameter, emphysema grade) into the decoder as a structured findings prompt, so information the network already has reaches the text.
* Constrained decoding that guarantees the recommendation matches a retrieved management passage.
* Uncertainty estimation and selective prediction, so low confidence cases are routed to a human first.
* A FHIR and DICOM structured reporting export.

## Licence

Released under the MIT licence. See `LICENSE`.
