# Dataset card: MedFusion synthetic lung nodule cohort

All records are synthetic. No real patient data was used at any stage.

## Overview

<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Studies (rows)</td><td>16,000</td></tr>
  <tr><td>Columns</td><td>73</td></tr>
  <tr><td>Patients</td><td>12,351</td></tr>
  <tr><td>Patients with repeat studies</td><td>3,306</td></tr>
  <tr><td>Studies with a nodule</td><td>13,811</td></tr>
  <tr><td>Malignancy prevalence among nodules</td><td>34.7%</td></tr>
  <tr><td>Studies with histology and WSI</td><td>2,992</td></tr>
  <tr><td>Mean report length (words)</td><td>57.5</td></tr>
  <tr><td>Mean note length (words)</td><td>36.7</td></tr>
</table>

## Splits (assigned at patient level)

<table>
  <tr><th>split</th><th>Studies</th><th>Share</th></tr>
  <tr><td>train</td><td>12,777</td><td>79.9%</td></tr>
  <tr><td>validation</td><td>1,625</td><td>10.2%</td></tr>
  <tr><td>test</td><td>1,598</td><td>10.0%</td></tr>
</table>

## Key distributions

<table>
  <tr><th>referral_pathway</th><th>Studies</th><th>Share</th></tr>
  <tr><td>screening programme</td><td>6,905</td><td>43.2%</td></tr>
  <tr><td>surveillance</td><td>3,649</td><td>22.8%</td></tr>
  <tr><td>incidental finding</td><td>3,614</td><td>22.6%</td></tr>
  <tr><td>urgent suspected cancer</td><td>1,832</td><td>11.4%</td></tr>
</table>

<table>
  <tr><th>lung_rads_category</th><th>Studies</th><th>Share</th></tr>
  <tr><td>2</td><td>6,346</td><td>39.7%</td></tr>
  <tr><td>1</td><td>3,100</td><td>19.4%</td></tr>
  <tr><td>4X</td><td>2,141</td><td>13.4%</td></tr>
  <tr><td>4A</td><td>1,771</td><td>11.1%</td></tr>
  <tr><td>3</td><td>1,591</td><td>9.9%</td></tr>
  <tr><td>4B</td><td>1,051</td><td>6.6%</td></tr>
</table>

<table>
  <tr><th>nodule_type</th><th>Studies</th><th>Share</th></tr>
  <tr><td>solid</td><td>9,208</td><td>66.7%</td></tr>
  <tr><td>part solid</td><td>2,151</td><td>15.6%</td></tr>
  <tr><td>ground glass</td><td>1,541</td><td>11.2%</td></tr>
  <tr><td>calcified</td><td>911</td><td>6.6%</td></tr>
</table>

<table>
  <tr><th>histology_subtype</th><th>Studies</th><th>Share</th></tr>
  <tr><td>adenocarcinoma</td><td>1,535</td><td>51.3%</td></tr>
  <tr><td>squamous cell carcinoma</td><td>358</td><td>12.0%</td></tr>
  <tr><td>granuloma</td><td>350</td><td>11.7%</td></tr>
  <tr><td>organising pneumonia</td><td>183</td><td>6.1%</td></tr>
  <tr><td>hamartoma</td><td>178</td><td>5.9%</td></tr>
  <tr><td>small cell carcinoma</td><td>145</td><td>4.8%</td></tr>
  <tr><td>carcinoid tumour</td><td>94</td><td>3.1%</td></tr>
  <tr><td>nondiagnostic sample</td><td>89</td><td>3.0%</td></tr>
  <tr><td>large cell carcinoma</td><td>60</td><td>2.0%</td></tr>
</table>

<table>
  <tr><th>recommendation_id</th><th>Studies</th><th>Share</th></tr>
  <tr><td>R01</td><td>7,008</td><td>43.8%</td></tr>
  <tr><td>R02</td><td>2,438</td><td>15.2%</td></tr>
  <tr><td>R06</td><td>2,047</td><td>12.8%</td></tr>
  <tr><td>R03</td><td>1,559</td><td>9.7%</td></tr>
  <tr><td>R04</td><td>1,545</td><td>9.7%</td></tr>
  <tr><td>R07</td><td>711</td><td>4.4%</td></tr>
  <tr><td>R05</td><td>458</td><td>2.9%</td></tr>
  <tr><td>R08</td><td>145</td><td>0.9%</td></tr>
  <tr><td>R09</td><td>89</td><td>0.6%</td></tr>
</table>

<table>
  <tr><th>growth_status</th><th>Studies</th><th>Share</th></tr>
  <tr><td>baseline</td><td>12,565</td><td>78.5%</td></tr>
  <tr><td>stable</td><td>2,619</td><td>16.4%</td></tr>
  <tr><td>growing</td><td>457</td><td>2.9%</td></tr>
  <tr><td>decreasing</td><td>359</td><td>2.2%</td></tr>
</table>

## Numeric summary

<table>
  <tr><th>Variable</th><th>Non missing</th><th>Mean</th><th>Median</th><th>5th percentile</th><th>95th percentile</th></tr>
  <tr><td>age_years</td><td>16,000</td><td>64.7</td><td>65.0</td><td>50.0</td><td>78.0</td></tr>
  <tr><td>pack_years</td><td>15,414</td><td>31.0</td><td>29.5</td><td>0.0</td><td>67.9</td></tr>
  <tr><td>bmi</td><td>15,423</td><td>27.2</td><td>27.2</td><td>19.1</td><td>35.4</td></tr>
  <tr><td>nodule_diameter_mm</td><td>13,811</td><td>9.6</td><td>8.4</td><td>3.6</td><td>20.1</td></tr>
  <tr><td>brock_risk_pct</td><td>13,811</td><td>10.1</td><td>4.8</td><td>0.1</td><td>40.0</td></tr>
  <tr><td>ctdivol_mgy</td><td>16,000</td><td>3.6</td><td>2.0</td><td>0.9</td><td>9.7</td></tr>
  <tr><td>volume_doubling_time_days</td><td>457</td><td>203.7</td><td>162.0</td><td>87.0</td><td>500.2</td></tr>
  <tr><td>ki67_index_pct</td><td>2,903</td><td>23.9</td><td>17.0</td><td>1.0</td><td>69.0</td></tr>
  <tr><td>pdl1_tps_pct</td><td>1,953</td><td>32.2</td><td>23.0</td><td>0.0</td><td>92.0</td></tr>
</table>

## Data dictionary

<table>
  <tr><th>Column</th><th>Type</th><th>Description</th><th>Missing</th></tr>
  <tr><td>case_id</td><td>string</td><td>Unique study identifier</td><td>0.0%</td></tr>
  <tr><td>patient_id</td><td>string</td><td>Pseudonymised patient identifier; patients may have several studies</td><td>0.0%</td></tr>
  <tr><td>study_sequence</td><td>int</td><td>Order of the study within the patient timeline, starting at 1</td><td>0.0%</td></tr>
  <tr><td>study_date</td><td>date</td><td>Study date in YYYY/MM/DD format</td><td>0.0%</td></tr>
  <tr><td>interval_days_from_prior</td><td>int</td><td>Days since the previous study for the same patient; blank at baseline</td><td>77.2%</td></tr>
  <tr><td>site_id</td><td>category</td><td>Acquisition site code</td><td>0.0%</td></tr>
  <tr><td>region</td><td>category</td><td>NHS region of the acquisition site</td><td>0.0%</td></tr>
  <tr><td>referral_pathway</td><td>category</td><td>screening programme, incidental finding, urgent suspected cancer or surveillance</td><td>0.0%</td></tr>
  <tr><td>age_years</td><td>int</td><td>Age at the time of the study</td><td>0.0%</td></tr>
  <tr><td>sex</td><td>category</td><td>Recorded sex</td><td>0.0%</td></tr>
  <tr><td>smoking_status</td><td>category</td><td>never, former or current</td><td>0.0%</td></tr>
  <tr><td>pack_years</td><td>float</td><td>Cumulative tobacco exposure in pack years</td><td>3.7%</td></tr>
  <tr><td>years_since_quit</td><td>int</td><td>Years since smoking cessation for former smokers</td><td>48.3%</td></tr>
  <tr><td>family_history_lung_cancer</td><td>binary</td><td>First degree relative with lung cancer</td><td>0.0%</td></tr>
  <tr><td>copd_diagnosis</td><td>binary</td><td>Recorded chronic obstructive pulmonary disease</td><td>0.0%</td></tr>
  <tr><td>prior_extrathoracic_cancer</td><td>binary</td><td>Previous cancer outside the thorax</td><td>0.0%</td></tr>
  <tr><td>bmi</td><td>float</td><td>Body mass index in kilograms per square metre; may be missing</td><td>3.6%</td></tr>
  <tr><td>ecog_status</td><td>int</td><td>ECOG performance status 0 to 3; may be missing</td><td>8.2%</td></tr>
  <tr><td>presenting_symptom</td><td>category</td><td>Dominant presenting symptom</td><td>0.0%</td></tr>
  <tr><td>scanner_vendor</td><td>category</td><td>CT scanner manufacturer</td><td>0.0%</td></tr>
  <tr><td>slice_thickness_mm</td><td>float</td><td>Reconstructed slice thickness</td><td>0.0%</td></tr>
  <tr><td>tube_voltage_kvp</td><td>int</td><td>Tube voltage</td><td>0.0%</td></tr>
  <tr><td>ctdivol_mgy</td><td>float</td><td>Volume CT dose index in milligray</td><td>0.0%</td></tr>
  <tr><td>contrast_enhanced</td><td>binary</td><td>Intravenous contrast administered</td><td>0.0%</td></tr>
  <tr><td>emphysema_severity</td><td>category</td><td>none, mild, moderate or severe</td><td>0.0%</td></tr>
  <tr><td>nodule_present</td><td>binary</td><td>At least one pulmonary nodule identified</td><td>0.0%</td></tr>
  <tr><td>nodule_count</td><td>int</td><td>Number of nodules identified</td><td>0.0%</td></tr>
  <tr><td>dominant_nodule_lobe</td><td>category</td><td>Lobe of the dominant nodule (RUL, RML, RLL, LUL, LLL or none)</td><td>0.0%</td></tr>
  <tr><td>nodule_x_norm</td><td>float</td><td>Horizontal centre of the dominant nodule on the key image (0 to 1)</td><td>13.7%</td></tr>
  <tr><td>nodule_y_norm</td><td>float</td><td>Vertical centre of the dominant nodule on the key image (0 to 1)</td><td>13.7%</td></tr>
  <tr><td>nodule_diameter_mm</td><td>float</td><td>Mean axial diameter of the dominant nodule</td><td>13.7%</td></tr>
  <tr><td>solid_component_mm</td><td>float</td><td>Solid component diameter for part solid nodules</td><td>86.6%</td></tr>
  <tr><td>nodule_volume_mm3</td><td>float</td><td>Estimated dominant nodule volume</td><td>13.7%</td></tr>
  <tr><td>nodule_type</td><td>category</td><td>solid, part solid, ground glass, calcified or none</td><td>0.0%</td></tr>
  <tr><td>margin</td><td>category</td><td>smooth, lobulated, spiculated or none</td><td>0.0%</td></tr>
  <tr><td>calcification_pattern</td><td>category</td><td>Benign calcification pattern when calcified</td><td>0.0%</td></tr>
  <tr><td>pleural_contact</td><td>binary</td><td>Dominant nodule abuts the pleura</td><td>0.0%</td></tr>
  <tr><td>mediastinal_lymphadenopathy</td><td>binary</td><td>Enlarged mediastinal lymph nodes on CT</td><td>0.0%</td></tr>
  <tr><td>prior_diameter_mm</td><td>float</td><td>Diameter of the same nodule on the prior study</td><td>78.5%</td></tr>
  <tr><td>growth_status</td><td>category</td><td>baseline, growing, stable or decreasing</td><td>0.0%</td></tr>
  <tr><td>volume_doubling_time_days</td><td>int</td><td>Volume doubling time when interval growth is present</td><td>97.1%</td></tr>
  <tr><td>brock_risk_pct</td><td>float</td><td>Brock style malignancy probability in percent</td><td>13.7%</td></tr>
  <tr><td>lung_rads_category</td><td>category</td><td>Lung RADS style category: 1, 2, 3, 4A, 4B or 4X</td><td>0.0%</td></tr>
  <tr><td>histology_available</td><td>binary</td><td>Tissue sample and whole slide image available</td><td>0.0%</td></tr>
  <tr><td>sampling_procedure</td><td>category</td><td>Tissue sampling method</td><td>0.0%</td></tr>
  <tr><td>histology_subtype</td><td>category</td><td>Histopathological diagnosis</td><td>0.0%</td></tr>
  <tr><td>adenocarcinoma_pattern</td><td>category</td><td>Predominant adenocarcinoma growth pattern</td><td>0.0%</td></tr>
  <tr><td>tumour_grade</td><td>category</td><td>Histological grade or neuroendocrine category</td><td>0.0%</td></tr>
  <tr><td>mitoses_per_2mm2</td><td>int</td><td>Mitotic count per 2 square millimetres</td><td>81.9%</td></tr>
  <tr><td>ki67_index_pct</td><td>int</td><td>Ki67 proliferation index in percent</td><td>81.9%</td></tr>
  <tr><td>tumour_cellularity_pct</td><td>float</td><td>Viable tumour cellularity of the sample in percent</td><td>81.3%</td></tr>
  <tr><td>nuclear_pleomorphism_score</td><td>int</td><td>Nuclear pleomorphism score 1 to 3</td><td>81.3%</td></tr>
  <tr><td>necrosis_present</td><td>binary</td><td>Tumour necrosis present in the sample</td><td>0.0%</td></tr>
  <tr><td>pdl1_tps_pct</td><td>int</td><td>PDL1 tumour proportion score in percent</td><td>87.8%</td></tr>
  <tr><td>molecular_driver</td><td>category</td><td>Actionable driver result</td><td>0.0%</td></tr>
  <tr><td>t_stage</td><td>category</td><td>Clinical T descriptor</td><td>0.0%</td></tr>
  <tr><td>n_stage</td><td>category</td><td>Clinical N descriptor</td><td>0.0%</td></tr>
  <tr><td>m_stage</td><td>category</td><td>Clinical M descriptor</td><td>0.0%</td></tr>
  <tr><td>clinical_stage</td><td>category</td><td>TNM 8th edition style stage group</td><td>0.0%</td></tr>
  <tr><td>icd10_code</td><td>category</td><td>ICD 10 code for the encounter</td><td>0.0%</td></tr>
  <tr><td>malignancy_label</td><td>binary</td><td>Reference standard malignancy of the dominant nodule</td><td>0.0%</td></tr>
  <tr><td>reference_standard</td><td>category</td><td>How the malignancy label was established</td><td>0.0%</td></tr>
  <tr><td>recommendation_id</td><td>category</td><td>Identifier of the guideline recommendation (R01 to R09)</td><td>0.0%</td></tr>
  <tr><td>gold_guideline_doc_ids</td><td>string</td><td>Knowledge base passages that support the recommendation</td><td>0.0%</td></tr>
  <tr><td>ct_image_seed</td><td>int</td><td>Seed for deterministic rendering of the CT key image</td><td>0.0%</td></tr>
  <tr><td>wsi_image_seed</td><td>int</td><td>Seed for deterministic rendering of the whole slide image tile</td><td>81.3%</td></tr>
  <tr><td>clinical_note</td><td>text</td><td>Free text referral and EHR summary (model input)</td><td>0.0%</td></tr>
  <tr><td>findings_text</td><td>text</td><td>CT findings section of the integrated report</td><td>0.0%</td></tr>
  <tr><td>pathology_text</td><td>text</td><td>Pathology section of the integrated report</td><td>0.0%</td></tr>
  <tr><td>impression_text</td><td>text</td><td>Impression section of the integrated report</td><td>0.0%</td></tr>
  <tr><td>recommendation_text</td><td>text</td><td>Guideline grounded recommendation sentence</td><td>0.0%</td></tr>
  <tr><td>integrated_report</td><td>text</td><td>Complete integrated radiology pathology report (generation target)</td><td>0.0%</td></tr>
  <tr><td>split</td><td>category</td><td>train, validation or test, assigned at patient level</td><td>0.0%</td></tr>
</table>
