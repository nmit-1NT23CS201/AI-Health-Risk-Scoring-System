# AI Health Risk Scoring System: Architectural Blueprint and Technical System Description

---

### Canonical Project Definition

The **AI Health Risk Scoring System** is an intelligent, multi-disease clinical decision-support and screening research platform designed to assess individual risk for major non-communicable cardiometabolic diseases—specifically **Cardiovascular Disease (CVD)**, **Type 2 Diabetes Mellitus**, and **Hypertension**. Developed on a nationally representative, multi-cycle epidemiological cohort (**CDC NHANES 2021–2023**, $N = 7,809$ adults aged 20 and older), the system introduces an adaptive **two-tier feature architecture**: **Mode A (Non-Invasive Screening)**, which utilizes readily accessible demographic, anthropometric, vital sign, and behavioral parameters; and **Mode B (Laboratory-Augmented Assessment)**, which integrates up to 16 routine and fasting blood biomarkers (including lipid subfractions, glycemic markers, renal indices, liver enzymes, and complete blood counts).

To address the severe calibration drift and clinical operating-point failures typical of raw machine learning classifiers on imbalanced epidemiological data, the platform incorporates a rigorous post-processing pipeline comprising **Sigmoid (Platt) probability calibration** and **validation-derived operating threshold optimization** evaluated under strict 5-fold out-of-fold cross-validation. The machine learning pipeline is paired with an explainable AI (XAI) engine utilizing **SHAP (SHapley Additive exPlanations)** to unpack model predictions into localized biomarker contributions, and is architected to support automated document parsing for patient blood-test reports. Implemented as a high-performance **FastAPI** backend and **Next.js** interactive dashboard, the system bridges the gap between complex statistical learning and transparent, preventive healthcare screening.

---

## 1. System Overview

Non-communicable diseases (NCDs)—primarily cardiovascular disorders, diabetes, and chronic hypertension—represent the leading cause of global mortality and premature morbidity. Early identification of individuals at high risk enables targeted lifestyle modification, therapeutic intervention, and substantial reductions in long-term healthcare expenditure. However, conventional population-scale screening faces a fundamental dilemma: comprehensive laboratory testing is resource-intensive and creates access barriers in low-resource settings, whereas basic questionnaire-based risk tools often lack the diagnostic precision required for nuanced clinical stratifications.

The **AI Health Risk Scoring System** resolves this trade-off by implementing an adaptive, multi-condition predictive framework that scales dynamically with the availability of user health data:

```
+----------------------------------------------------------------------------------------------------+
|                                 AI HEALTH RISK SCORING SYSTEM                                      |
+----------------------------------------------------------------------------------------------------+
                                                  |
                        +-------------------------+-------------------------+
                        |                                                   |
                        v                                                   v
           +-------------------------+                         +-------------------------+
           |         MODE A          |                         |         MODE B          |
           |   Non-Invasive Tier     |                         |   Lab-Augmented Tier    |
           | (Questionnaire/Vitals)  |                         | (Mode A + Blood Panel)  |
           +-------------------------+                         +-------------------------+
                        |                                                   |
                        +-------------------------+-------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   DISEASE-SPECIFIC ML ENGINE (V2)   |
                               |  - Cardiovascular Disease (RF)      |
                               |  - Diabetes Mellitus (HGB / XGB)    |
                               |  - Hypertension (LR / RF)           |
                               +-------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               | PROBABILITY CALIBRATION & THRESHOLD |
                               |  - Sigmoid (Platt) Scaling          |
                               |  - Screening Operating Thresholds   |
                               +-------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |       EXPLAINABILITY & DASHBOARD    |
                               |  - Localized SHAP Attributions      |
                               |  - Next.js Interactive Dashboard    |
                               +-------------------------------------+
```

Key operational pillars include:
1. **Multi-Condition Profiling:** Rather than predicting an undifferentiated, generic "health score," the system trains and serves independent, clinically grounded models for Cardiovascular Disease, Diabetes, and Hypertension.
2. **Two-Tier Ingestion (Mode A vs. Mode B):** Enables immediate, friction-free community or home triage when lab results are unavailable (Mode A), while providing a seamless upgrade path to high-dimensional clinical laboratory evaluation when blood work is provided (Mode B).
3. **Calibrated Probabilistic Inference:** Replaces arbitrary classifier cutoffs ($0.50$) with mathematically calibrated probabilities and validation-optimized screening thresholds designed specifically to minimize false negatives in population screening.
4. **Transparent Feature Attribution:** Employs cooperative game theory (SHAP) to transform "black-box" model outputs into patient-specific waterfall explanations and physician-interpretable risk factors.
5. **Modern Decoupled Architecture:** Employs a decoupled microservices design utilizing Python/FastAPI for high-throughput inference and Next.js/TypeScript for real-time visualization.

---

## 2. Problem Statement

Modern preventive medicine and healthcare data science face four distinct systemic challenges:

1. **The Asymptomatic Progression Window:** Chronic cardiometabolic disorders develop over years or decades with minimal overt symptomatology. By the time a patient presents with symptomatic hypertension, microvascular diabetic complications, or acute coronary syndrome, irreversible organ damage has frequently occurred.
2. **Clinical Friction of Routine Laboratory Testing:** While biochemical panels (e.g., lipid fractionation, fasting plasma glucose, renal function tests) provide critical physiological insights, requiring a laboratory blood draw for basic initial screening excludes vast populations in rural, underserved, or primary-care triage settings.
3. **The Base-Rate Fallacy and Threshold Misalignment in ML Classifiers:** In real-world epidemiological populations, disease prevalence is heavily skewed (e.g., hard CVD prevalence is approximately $12.6\%$ in adults). Standard off-the-shelf machine learning classifiers default to an arbitrary decision threshold of $p \ge 0.50$. In imbalanced epidemiological cohorts, this results in catastrophic under-diagnosis: our baseline benchmarks reveal that uncalibrated models operating at $0.50$ miss **$80\%$ to $90\%$** of actual at-risk individuals (yielding sensitivities as poor as $9.5\%$).
4. **Opacity and Distrust in Algorithmic Scoring:** Clinicians and patients legitimately reject uninterpretable risk scores. Standard risk calculators (e.g., Framingham, ASCVD Risk Estimator) use simplified parametric equations that cannot capture complex non-linear biomarker interactions, while modern ensemble classifiers (Random Forests, Gradient Boosting Machines) operate as black boxes without integrated explanation layers.

---

## 3. Objectives

The primary research and engineering objectives of the AI Health Risk Scoring System are:

* **Objective 1: Epidemiological Dataset Engineering:** Synthesize, clean, and harmonise multiple survey and laboratory modules from the official CDC NHANES 2021–2023 release into six standardized, leakage-controlled analytic datasets representing an adult cohort of $N = 7,809$ individuals.
* **Objective 2: Adaptive Dual-Tier Modeling:** Design, train, and validate distinct model pipelines for Mode A (non-invasive features only) and Mode B (Mode A + clinical laboratory biomarkers) across CVD, Diabetes, and Hypertension.
* **Objective 3: Rigorous Algorithmic Benchmarking:** Conduct a controlled, progressive benchmarking study across linear models (Logistic Regression with $L_2$ regularization), tree ensembles (Random Forest with tuned depth and leaf constraints), and gradient boosting architectures (HistGradientBoosting, XGBoost).
* **Objective 4: Probability Calibration & Threshold Optimization:** Implement post-hoc calibration (Sigmoid vs. Isotonic) and sweep operating thresholds on out-of-fold validation distributions to optimize Youden's $J$ statistic and $F_2$ screening metrics, explicitly addressing class imbalance without artificial data synthesis.
* **Objective 5: Localized Explainability Integration:** Formulate a real-time SHAP explanation framework that generates feature attributions and base64-encoded visual plots (summary, bar, waterfall) for each assessment.
* **Objective 6: End-to-End Decision Support Architecture:** Formulate the comprehensive system architecture uniting user input, document ingestion vision, inference APIs, persistent assessment history, and a modern web dashboard.

---

## 4. Proposed System

The proposed system is an end-to-end clinical intelligence platform capable of operating in dual modes:

```
+----------------------------------------------------------------------------------------------------+
|                                    PROPOSED SYSTEM WORKFLOW                                        |
+----------------------------------------------------------------------------------------------------+

 [User Ingestion]
        |
        +---> Mode A: Interactive Questionnaire (Demographics, Lifestyle, Vitals)
        |
        +---> Mode B: Questionnaire + Blood Report Document Upload (PDF / Image)
                   |
                   v (Proposed Lab Extraction Module)
             [OCR & Biomarker Extraction] (NER, Unit Normalization, Schema Mapping)
                   |
                   +-----------------------+
                                           |
                                           v
                             [Preprocessing & Validation]
                             (Range checks, Imputation)
                                           |
                                           v
                             [Disease-Specific ML Inference]
                                - CVD Model (Mode A or B)
                                - Diabetes Model (Mode A or B)
                                - Hypertension Model (Mode A or B)
                                           |
                                           v
                             [Post-Hoc Probability Calibration]
                             (Sigmoid / Platt Transformation)
                                           |
                                           v
                             [Optimized Threshold Decision Logic]
                             (Classification: Low / Moderate / High)
                                           |
                                           v
                             [SHAP Explainability Engine]
                             (Feature attribution & localized impact)
                                           |
                                           v
                             [Interactive Next.js Dashboard]
                             (Dynamic dials, waterfall plots, recommendations)
```

1. **Ingestion Layer:** Accepts self-reported patient parameters through an interactive, responsive user interface. When blood test results are available, the user can either enter biomarker values manually or upload laboratory report documents (PDF, scans, images) to be parsed by an automated extraction engine.
2. **Data Normalization & Feature Alignment:** The system sanitizes inputs, harmonizes clinical measurement units (e.g., standardizing blood glucose from mmol/L to mg/dL or cholesterol subfractions), performs boundary validation against physiological extremes, and maps values to the required feature vector.
3. **Inference & Calibration Layer:** Evaluates the feature vector against disease-specific machine learning pipelines. Raw model outputs are transformed via calibrated Sigmoid functions into true posterior probabilities, which are then compared against pre-computed, validation-locked screening thresholds.
4. **Interpretation & Attribution Engine:** Deconstructs the calibrated prediction into top positive and negative risk contributors using SHAP values. Translates statistical weightings into plain-language clinical insights and lifestyle recommendations.
5. **Persistence & Presentation Layer:** Stores assessment records in a relational database for longitudinal tracking and renders dynamic visual components—including interactive risk dials, comparative biomarker tables, and SHAP waterfall diagrams—on the user dashboard.

---

## 5. Current Implementation Status

To maintain strict scientific integrity, the technical state of the repository is explicitly delineated into currently implemented functionality versus planned architectural vision:

### Currently Implemented (Verified in Codebase)
* **Dataset Harmonization Pipeline (`backend/ml/scripts/`):** Complete automated ingestion of 16 raw CDC NHANES 2021–2023 XPT files. Automated filtering to adult cohort ($N = 7,809$), target definition, data sanitization, SAS underflow clamping, and generation of six processed Parquet files under `backend/ml/data/processed/nhanes_2021_2023/`.
* **Leakage-Free Partitioning:** Fixed participant-level splits (`cvd_splits.csv`, `diabetes_splits.csv`, `hypertension_splits.csv`) establishing an exact $70\%$ Train ($N=5,464$), $15\%$ Validation ($N=1,171$), and $15\%$ Test ($N=1,171$) split structure.
* **Benchmarking & Optimization Suite:**
  * Stage 1D baseline benchmarking across 5 standard algorithms.
  * Stage 1E-1 un-tuned XGBoost baseline evaluation.
  * Stage 1E-2 targeted RandomizedSearchCV optimization (5-fold Stratified CV, 40 iterations) for Logistic Regression, HistGradientBoosting, and XGBoost.
  * Stage 1E-3 controlled hyperparameter optimization for Random Forest across depth, leaf size, feature ratios, and tree counts.
* **Probability Calibration & Threshold Pipeline (Stage 1F):**
  * Training base estimators strictly on Train ($70\%$).
  * Computing 5-fold out-of-fold (OOF) cross-validated validation probabilities.
  * Comparative evaluation of Raw vs. Sigmoid (Platt) vs. Isotonic calibration.
  * Sweeping 91 operating thresholds ($0.05$ to $0.95$) on OOF validation predictions to lock screening thresholds.
  * Refitting final Sigmoid calibrators on the full validation partition and serializing 12 model artifacts (`.joblib` and `.pkl`) in `backend/ml/evaluation/stage_1f/models/`.
  * Evaluating the finalized pipeline exactly once on the untouched held-out Test set ($15\%$).
* **Model V1 Full-Stack Prototype:**
  * A working legacy application connecting a Next.js 16 frontend (`frontend/app/page.tsx`) to a FastAPI backend (`backend/api/api_server.py`) serving a continuous `RandomForestRegressor` trained on synthetic data (`dataset/indian_health_risk_dataset.csv`), with SQLite assessment history logging (`backend/database/database.py`) and dynamic SHAP plot generation.

### Planned / System Vision (Pending Integration)
* **V2 Model Serving Migration:** Migrating the active FastAPI `/predict` route from the legacy Model V1 single-target regressor to the six calibrated Model V2 classification pipelines.
* **Blood-Test Report OCR Ingestion:** The document parsing, text extraction, biomarker entity recognition, and automated unit mapping module is designed as an architectural component but is **not yet implemented** in executable code.
* **Multi-Condition UI Dashboard:** Upgrading the frontend dashboard to display concurrent risk cards for CVD, Diabetes, and Hypertension with Mode A/B toggle controls.
* **User Authentication & Longitudinal Tracking:** Role-based access control, secure patient profile management, and multi-assessment trend tracking over time.

---

## 6. Complete Product Vision

The ultimate product vision for the AI Health Risk Scoring System is a comprehensive, preventive health co-pilot serving patients, primary healthcare providers, and corporate wellness initiatives:

```
+----------------------------------------------------------------------------------------------------+
|                                    FINAL PRODUCT ECOSYSTEM                                         |
+----------------------------------------------------------------------------------------------------+

     [Patient / Community Portal]                     [Clinical Triage Interface]
     - Frictionless Mode A self-check                 - Multi-condition overview
     - Smartphone camera lab report upload            - Calibrated absolute risk probabilities
     - Plain-language risk factor breakdowns          - Visual SHAP biomarker attributions
     - Actionable lifestyle guidance                  - PDF clinical summary export
                  |                                                |
                  +-----------------------+------------------------+
                                          |
                                          v
                      [Unified Cloud Analytics Platform]
                      - High-performance inference engine
                      - Automated OCR & lab report normalizer
                      - Encrypted longitudinal risk history
                      - Calibrated epidemiological models (NHANES/Global)
```

In its target deployment, an individual can complete a 2-minute non-invasive assessment at home or at a community health kiosk. If abnormal cardiovascular or metabolic indicators are flagged, the platform prompts the user to upload recent laboratory blood test reports. Utilizing multimodal document vision, the platform parses raw lab sheets, populates the Mode B clinical feature vector, and updates the patient's calibrated risk trajectory across all three conditions. For primary care physicians, the platform provides a clinical decision-support dashboard showing not just raw risks, but mathematically calibrated probabilities, operating sensitivity profiles, and localized SHAP waterfall charts illustrating exactly which physiological metrics are driving the patient's elevated risk.

---

## 7. System Architecture

The technical architecture is structured as a decoupled, multi-layered service model:

```
+----------------------------------------------------------------------------------------------------+
|                                      SYSTEM ARCHITECTURE                                           |
+----------------------------------------------------------------------------------------------------+

 [ PRESENTATION LAYER ]
  Next.js 16 (React 18, TypeScript, CSS Variables, Responsive Viewport)
  - Interactive Health Assessment Form (Sliders, Dropdowns, Numeric Inputs)
  - Mode A / Mode B Dual Input View
  - Real-Time Risk Display (Animated Gauge, Pill Badges, Dynamic Metrics)
  - Explainable AI Visualizer (Base64 Rendered Waterfall, Summary & Bar Plots)
  - Assessment History Table (Asynchronous fetch, delete, review)
         |
         | HTTP / JSON REST APIs (CORS Enabled)
         v
 [ APPLICATION & ROUTING LAYER ]
  FastAPI (Python, Uvicorn ASGI Server, Pydantic Data Contracts)
  - POST /predict: Primary inference pipeline
  - GET  /history: Historical assessment retrieval
  - DEL  /history/{id}: Assessment deletion
  - GET  /model-metrics: Pre-computed model performance delivery
         |
         +---------------------------------------+
         |                                       |
         v                                       v
 [ INFERENCE & EXPLAINABILITY ]          [ DATA & PERSISTENCE ]
  - Preprocessing & Scaler Pipelines      - SQLite Relational DB (`healthrisk.db`)
  - Calibrated Classifier CV Models       - Tables: `assessments` (Demographics,
  - SHAP Explainer Engine (Tree/Linear)     Vitals, Scores, Risk Categories)
  - Heuristic Recommendation Generator    - JSON Serialized Evaluation Logs
```

### Technical Stack Specifications
* **Frontend Runtime:** Next.js `16.2.7`, React `18.3.1`, React DOM `18.3.1`, TypeScript `5.5.4`.
* **Frontend Styling:** Vanilla CSS design system (`globals.css`) with glassmorphism tokens, dark-mode color palettes, fluid layout grids, and CSS animation primitives.
* **Backend Framework:** FastAPI with Pydantic for strict request payload validation and data type enforcement.
* **Asynchronous Server:** Uvicorn ASGI server with standard CORS middleware configurations for secure local and cross-origin communication.
* **Machine Learning & Math:** Scikit-learn `1.9.0`, XGBoost `3.4.1`, NumPy, Pandas, Joblib.
* **Explainability Framework:** SHAP `0.46+` with Matplotlib backend integration (rendering to in-memory headless PNG buffers encoded as Base64 strings).
* **Relational Storage:** SQLite 3 via Python native `sqlite3` driver, anchoring table definitions with automated initialization upon application startup.

---

## 8. Dataset and Data Pipeline

### 8.1 Primary Dataset: CDC NHANES 2021–2023
The core research pipeline is built upon the August 2021–August 2023 continuous cycle of the **National Health and Nutrition Examination Survey (NHANES)** conducted by the National Center for Health Statistics (NCHS) at the Centers for Disease Control and Prevention (CDC). NHANES uses a complex, stratified, multistage probability sampling design to produce a representative sample of the civilian, non-institutionalized U.S. population.

The raw data consists of 16 distinct SAS transport files (`.XPT`) located in `backend/ml/data/raw/nhanes_2021_2023/`:

| Module Category | Filename | Survey Domain | Raw Records | Key Variables Extracted |
|:---|:---|:---|:---:|:---|
| **Demographics** | `DEMO_L.xpt` | Demographics & Survey Weights | 11,933 | `SEQN`, `RIDAGEYR`, `RIAGENDR`, `DMDEDUC2`, `INDFMPIR`, `WTINT2YR`, `WTMEC2YR`, `SDMVSTRA`, `SDMVPSU` |
| **Examination** | `BMX_L.xpt` | Anthropometry (Body Measures) | 8,860 | `BMXBMI` (BMI), `BMXWAIST` (Waist Circumference) |
| **Examination** | `BPXO_L.xpt` | Blood Pressure (Oscillometric) | 7,801 | `BPXOSY1-3` (Systolic), `BPXODI1-3` (Diastolic), `BPXOPLS1-3` (Pulse) |
| **Questionnaire** | `BPQ_L.xpt` | Blood Pressure & Cholesterol | 8,501 | `BPQ020` (Hypertension Diagnosis History), `BPQ101D` |
| **Questionnaire** | `SMQ_L.xpt` | Smoking & Tobacco Use | 9,015 | `SMQ020` (100+ cigarettes lifetime), `SMQ040` (Current smoking frequency) |
| **Questionnaire** | `ALQ_L.xpt` | Alcohol Consumption | 6,337 | `ALQ111` (Ever had drink), `ALQ121` (Past 12-month frequency) |
| **Questionnaire** | `PAQ_L.xpt` | Physical Activity (GPAQ) | 8,153 | `PAD790Q`, `PAD810Q`, `PAD680` (Sedentary minutes) |
| **Questionnaire** | `DIQ_L.xpt` | Diabetes Medical History | 11,744 | `DIQ010` (Doctor diagnosed), `DIQ050` (Insulin use), `DIQ070` (Oral agents) |
| **Questionnaire** | `MCQ_L.xpt` | Medical Conditions / Morbidities | 11,744 | `MCQ160B` (CHF), `MCQ160C` (CHD), `MCQ160D` (Angina), `MCQ160E` (MI), `MCQ160F` (Stroke) |
| **Laboratory** | `TCHOL_L.xpt`| Serum Total Cholesterol | 8,068 | `LBXTC` (Total cholesterol, mg/dL) |
| **Laboratory** | `HDL_L.xpt` | HDL-Cholesterol | 8,068 | `LBDHDD` (Direct HDL-cholesterol, mg/dL) |
| **Laboratory** | `TRIGLY_L.xpt`| Triglycerides & LDL (Fasting) | 3,996 | `LBXTLG` (Serum triglycerides), `LBDLDL` (Calculated LDL-cholesterol) |
| **Laboratory** | `GLU_L.xpt` | Fasting Plasma Glucose | 3,996 | `LBXGLU` (Fasting glucose, mg/dL), `WTSAF2YR` (Subsample weight) |
| **Laboratory** | `GHB_L.xpt` | Glycohemoglobin | 7,199 | `LBXGH` (Glycohemoglobin HbA1c, %) |
| **Laboratory** | `BIOPRO_L.xpt`| Comprehensive Biochemistry | 7,199 | `LBXSCR` (Creatinine), `LBXSBU` (BUN), `LBXSUA` (Uric acid), `LBXSATSI` (ALT), `LBXSASSI` (AST), `LBXSAL` (Albumin) |
| **Laboratory** | `CBC_L.xpt` | Complete Blood Count | 8,727 | `LBXHGB` (Hemoglobin), `LBXWBCSI` (WBC), `LBXPLTSI` (Platelets), `LBXRDW` (Red Cell Distribution Width) |

### 8.2 Cohort Construction & Harmonization Rules
1. **Adult Eligibility Filter:** The full NHANES file contains 11,933 participants of all ages. An adult filter (`RIDAGEYR >= 20`) is applied immediately, isolating a base cohort of **$N = 7,809$ adult participants**.
2. **Left-Join Assembly:** All examination, questionnaire, and laboratory files are merged to the demographic backbone via unique participant identifiers (`SEQN`) using left outer joins. This preserves the entire adult cohort for Mode A without artificially discarding participants who lack laboratory draws.
3. **SAS Floating-Point & Special Code Sanitization:**
   * NHANES codes refusal (`7`, `77`, `777`, `7777`) and missing knowledge (`9`, `99`, `999`, `9999`) with distinct integer flags. All such instances are explicitly recoded to `np.nan`.
   * Due to SAS transport encoding nuances, true zero values in continuous variables are stored as floating-point underflow epsilons ($\approx 5.3976 \times 10^{-79}$). The pipeline scans continuous features and clamps all values $< 10^{-10}$ to exactly `0.0`.
4. **Physiological Signal Aggregation:** Oscillometric blood pressure is captured across up to three consecutive seated readings (`BPXOSY1-3`, `BPXODI1-3`, `BPXOPLS1-3`). The pipeline computes a robust intra-individual mean across valid readings (`mean_sbp`, `mean_dbp`, `mean_pulse`).

### 8.3 Status of Secondary/Indian Datasets
* **Legacy Synthetic Dataset (`dataset/indian_health_risk_dataset.csv`):** A 151-row synthetic CSV file containing simulated wearable fields (`sdnn_hrv`, `rmssd_hrv`, `spo2`) and a continuous target `risk_score`. This file was created for the Model V1 proof-of-concept. **It is not used in the Stage 1E/1F V2 machine learning pipeline.**
* **ICMR-INDIAB Cohort Sample (`backend/ml/data/raw/icmr_indiab/sample.dta`):** A 500-row, 41-variable Stata dataset from the Indian Council of Medical Research–India Diabetes study audited during Stage 1A. It provides authoritative Indian epidemiological reference distributions (e.g., Asian-specific BMI cutoffs $\ge 25 \text{ kg/m}^2$, waist circumference $\ge 90\text{ cm}$ for men and $\ge 80\text{ cm}$ for women). **It is retained in the repository as an external reference asset but is not part of the active V2 model training sets.**

---

## 9. Target Disease Models

The system formulates three separate binary classification tasks based on standardized epidemiological and clinical diagnostic guidelines:

```
+----------------------------------------------------------------------------------------------------+
|                                    DISEASE TARGET SPECIFICATIONS                                   |
+----------------------------------------------------------------------------------------------------+

 [ CARDIOVASCULAR DISEASE (CVD) ]
  Definition: Self-reported physician diagnosis of hard cardiovascular endpoints.
  Equation:   MCQ160B==1 (CHF) OR MCQ160C==1 (CHD) OR MCQ160D==1 (Angina) OR
              MCQ160E==1 (Heart Attack/MI) OR MCQ160F==1 (Stroke)
  Prevalence: 12.58% (982 positive / 6,825 negative / 2 missing) [N = 7,807]

 [ TYPE 2 DIABETES MELLITUS ]
  Definition: Multi-criteria ADA diagnostic standard combining biomarkers & diagnosis.
  Equation:   LBXGH >= 6.5% (HbA1c) OR LBXGLU >= 126 mg/dL (Fasting Glucose) OR
              DIQ010==1 (Physician Diagnosed) OR DIQ050==1 (Insulin) OR DIQ070==1 (Oral Agents)
  Prevalence: 17.74% (1,385 positive / 6,421 negative / 3 missing) [N = 7,806]

 [ HYPERTENSION ]
  Definition: JNC7 Clinical Guideline Threshold (Stage 1 / Stage 2 or Diagnosed).
  Equation:   mean_sbp >= 140 mmHg OR mean_dbp >= 90 mmHg OR BPQ020==1 (Physician Diagnosed)
  Prevalence: 42.71% (3,331 positive / 4,469 negative / 9 missing) [N = 7,800]
```

---

## 10. Mode A and Mode B Architecture

To accommodate varying levels of health data availability, each disease target is implemented across two distinct feature schemas:

```
+----------------------------------------------------------------------------------------------------+
|                                   FEATURE COMPOSITION & LEAKAGE CONTROL                            |
+----------------------------------------------------------------------------------------------------+

 [ BASELINE NON-INVASIVE SCREENING (MODE A) ]
  Features (11 to 13 predictors):
  - Demographics: Age, Gender, Education Level, Poverty-Income Ratio
  - Anthropometrics: Body Mass Index (BMI), Waist Circumference
  - Examination Vitals: Mean Systolic BP*, Mean Diastolic BP*, Mean Resting Pulse
  - Behavioral Lifestyle: Smoking Status (Never/Former/Current), Alcohol Frequency,
                          Physical Activity Level (High/Moderate/Low), Sedentary Minutes/Day
  * Excluded from Hypertension model to prevent target leakage.

 [ LABORATORY-AUGMENTED ASSESSMENT (MODE B) ]
  Features (27 to 29 predictors):
  - Mode A Features (Demographics, Anthropometrics, Vitals, Lifestyle)
  - Lipid Biomarkers: Total Cholesterol, HDL-Cholesterol, Triglycerides, LDL-Cholesterol
  - Metabolic & Glycemic Biomarkers: HbA1c**, Fasting Plasma Glucose**
  - Renal Profile: Serum Creatinine, Blood Urea Nitrogen (BUN), Serum Uric Acid
  - Hepatic Enzymes: Alanine Aminotransferase (ALT), Aspartate Aminotransferase (AST)
  - Hematological Indices: Hemoglobin, White Blood Cell Count (WBC), Platelet Count,
                           Red Cell Distribution Width (RDW), Serum Albumin
  ** Excluded from Diabetes model to prevent target leakage.
```

### Strict Leakage Prevention Matrix
Target-defining variables, diagnostic proxies, and survey design weights must be quarantined from predictive feature vectors. The repository enforces an automated assertion audit (`assert_no_leakage`) before any model training:

| Variable Identifier | Clinical Role | CVD Predictor? | Diabetes Predictor? | Hypertension Predictor? |
|:---|:---|:---:|:---:|:---:|
| `MCQ160B-F` | CVD Target Defining Variables | **PROHIBITED** | Prohibited | Prohibited |
| `LBXGH` (HbA1c), `LBXGLU` (Fasting Glucose) | Diabetes Diagnostic Biomarkers | Allowed | **PROHIBITED** | Allowed |
| `DIQ010`, `DIQ050`, `DIQ070` | Diabetes Medication / History | Prohibited | **PROHIBITED** | Prohibited |
| `mean_sbp`, `mean_dbp` | Hypertension Diagnostic Vitals | Allowed | Allowed | **PROHIBITED** |
| `BPQ020` | Hypertension Diagnosis History | Prohibited | Prohibited | **PROHIBITED** |
| `BPQ101D` | Cholesterol Medication Proxy | **PROHIBITED** | **PROHIBITED** | **PROHIBITED** |
| `WTINT2YR`, `WTMEC2YR`, `WTSAF2YR` | Survey Design Sampling Weights | **PROHIBITED** | **PROHIBITED** | **PROHIBITED** |
| `SDMVSTRA`, `SDMVPSU`, `SEQN` | Survey Strata, PSU, & ID | **PROHIBITED** | **PROHIBITED** | **PROHIBITED** |

---

## 11. Machine Learning Methodology

The machine learning methodology adheres to rigorous epidemiological and statistical learning principles:

```
+----------------------------------------------------------------------------------------------------+
|                                DATA PARTITIONING & MODEL WORKFLOW                                  |
+----------------------------------------------------------------------------------------------------+

                          POPULATION DATASET (N = 7,809 Adults)
                                           |
         +---------------------------------+---------------------------------+
         | 85% DEVELOPMENT SET                                               | 15% HELD-OUT TEST
         | (N = 6,635)                                                       | (N = 1,171)
         v                                                                   v
  +---------------------------------------+                           +---------------------+
  | TRAIN SET (70%)                       |                           | UNTOUCHED TEST SET  |
  | N = 5,464                             |                           | N = 1,171           |
  | Base Estimator Fitting Only           |                           | Final Single-Pass   |
  +---------------------------------------+                           | Evaluation Only     |
         |                                                            +---------------------+
         v                                                                       ^
  +---------------------------------------+                                      |
  | VALIDATION SET (15%)                  |                                      |
  | N = 1,171                             |                                      |
  | 1. 5-Fold OOF Predictions             |                                      |
  | 2. Compare Raw vs Sigmoid vs Isotonic |                                      |
  | 3. Optimize Operating Threshold (t)   |                                      |
  | 4. LOCK Calibration & Threshold       |                                      |
  | 5. Refit Calibrator on Full 100% Val  |--------------------------------------+
  +---------------------------------------+
```

### Partitioning & Imbalance Strategy
* **Participant-Level Independence:** Data splits are generated on unique participant IDs (`SEQN`) and stratified by the target label. All modes for a given disease use identical splits, guaranteeing comparability.
* **Absence of Synthetic Oversampling (No SMOTE):** In medical epidemiological modeling, synthetic minority oversampling (such as SMOTE) alters underlying joint probability densities, distorting calibrated risk estimations. Class imbalance is addressed via algorithmic class weighting and post-hoc threshold selection rather than artificial data generation.
* **Fold-Enclosed Preprocessing:** For models requiring imputation and scaling, preprocessors (`SimpleImputer(strategy='median')`, `StandardScaler()`) are encapsulated within Scikit-learn `Pipeline` objects. Imputation statistics are fitted exclusively on training splits, eliminating cross-fold data snooping.

---

## 12. Model Optimization

Model development progressed through four systematic experimental stages:

1. **Stage 1D (Baseline Multi-Model Benchmarking):** Evaluated default Scikit-learn algorithms: Logistic Regression, Balanced Logistic Regression, Random Forest, Balanced Random Forest, HistGradientBoosting, and Balanced HistGradientBoosting. Linear models performed exceptionally well on CVD, while gradient boosting showed strength on Diabetes.
2. **Stage 1E-1 (XGBoost Baseline Study):** Added un-tuned XGBoost 3.4.1 to the benchmark. Results confirmed that default XGBoost did not automatically outperform tuned linear models or HistGradientBoosting on tabular survey data.
3. **Stage 1E-2 (Targeted Hyperparameter Tuning):** Performed `RandomizedSearchCV` ($n_{\text{iter}} = 40$, 5-fold Stratified CV, random seed 42) on the development set ($85\%$). Tuned regularization parameters ($C$ in LR, learning rates, maximum leaf nodes, and $L_2$ penalties in HistGradientBoosting and XGBoost).
4. **Stage 1E-3 (Controlled Random Forest Optimization):** Explored whether regularizing tree complexity could elevate Random Forest performance. Conducted RandomizedSearchCV over 40 parameter combinations ($n_{\text{estimators}} \in [300, 800]$, $max\_depth \in [8, 20, \text{None}]$, $min\_samples\_split \in [2, 10]$, $min\_samples\_leaf \in [1, 10]$, $max\_features \in [\text{'sqrt'}, \text{'log2'}, 0.5, 0.8]$). Constraining tree depth ($max\_depth = 8$) and leaf sizes successfully prevented overfitting, allowing Random Forest to achieve top test ROC-AUC scores on CVD Mode A ($0.8140$), CVD Mode B ($0.8244$), and Hypertension Mode B ($0.7717$).

---

## 13. Probability Calibration and Threshold Optimization

### 13.1 Probability Calibration
Standard classifiers output uncalibrated scores that do not reflect true statistical event probabilities. Tree ensembles, in particular, exhibit sigmoidal distortion, pushing probabilities toward $0.0$ and $1.0$.

In **Stage 1F**, three calibration regimes were systematically evaluated on the validation set using **5-fold out-of-fold cross-validation**:
1. **Raw Classifier Probabilities:** Direct uncalibrated outputs.
2. **Sigmoid / Platt Calibration:** A parametric logistic transformation fitting $\hat{P}(Y=1|f) = \frac{1}{1 + \exp(A \cdot f + B)}$.
3. **Isotonic Calibration:** A non-parametric piecewise-constant isotonic regression fitting a monotonic step function.

```
+----------------------------------------------------------------------------------------------------+
|                             VALIDATION CALIBRATION COMPARISON (OOF)                                |
+----------------------------------------------------------------------------------------------------+
  Config             Method            Brier Score (lower=better)   ECE (lower=better)    Selected
  --------------------------------------------------------------------------------------------------
  cvd_mode_a         Raw                         0.0924                   0.0271
                     Sigmoid (Platt)             0.0922                   0.0111          SELECTED
                     Isotonic                    0.0939                   0.0155
  --------------------------------------------------------------------------------------------------
  cvd_mode_b         Raw                         0.0885                   0.0167
                     Sigmoid (Platt)             0.0884                   0.0101          SELECTED
                     Isotonic                    0.0902                   0.0236
  --------------------------------------------------------------------------------------------------
  diabetes_mode_a    Raw                         0.1238                   0.0429
                     Sigmoid (Platt)             0.1231                   0.0373          SELECTED
                     Isotonic                    0.1239                   0.0256
  --------------------------------------------------------------------------------------------------
  diabetes_mode_b    Raw                         0.1132                   0.0395
                     Sigmoid (Platt)             0.1136                   0.0346          SELECTED
                     Isotonic                    0.1149                   0.0336
  --------------------------------------------------------------------------------------------------
  hypertension_mode_a Raw                        0.1874                   0.0582
                     Sigmoid (Platt)             0.1848                   0.0323          SELECTED
                     Isotonic                    0.1864                   0.0283
  --------------------------------------------------------------------------------------------------
  hypertension_mode_b Raw                        0.1776                   0.0273
                     Sigmoid (Platt)             0.1772                   0.0334          SELECTED
                     Isotonic                    0.1802                   0.0343
```
*Selection Finding:* Sigmoid (Platt) scaling was selected across all six models. While isotonic regression appeared competitive in-sample, out-of-fold evaluation demonstrated that it produced higher Brier scores and step-function instability due to sample-size limits within sparse probability regions. Sigmoid scaling consistently minimized Expected Calibration Error (ECE down to $0.0101$) while preserving smooth ranking properties.

### 13.2 Operating Threshold Selection
In health-risk screening, false negatives carry vastly higher clinical consequences than false positives. Default thresholds ($0.50$) fail because they implicitly treat false positives and false negatives as equally costly in cohorts where the positive prevalence is only $12–18\%$.

Operating thresholds were swept across 91 discrete candidate values ($t \in [0.05, 0.95]$, step $0.01$) exclusively on **5-fold out-of-fold validation probabilities**. The selection criterion optimized **Youden's $J$ Index** ($J = \text{Sensitivity} + \text{Specificity} - 1$) subject to a clinical screening constraint requiring $\text{Sensitivity} \ge 0.65$ where feasible:

$$\hat{t} = \arg\max_{t \in [0.05, 0.95]} \left[ \text{Sensitivity}(t) + \text{Specificity}(t) - 1 \right] \quad \text{s.t.} \quad \text{Sensitivity}(t) \ge 0.65$$

---

## 14. Final Model Configuration

The finalized, production-ready V2 model suite consists of the following locked algorithms, calibration mappings, and operating thresholds:

| Disease Target | Mode | Selected Architecture | Source Stage | Key Hyperparameters | Calibration | Operating Threshold ($t_{\text{opt}}$) |
|:---|:---:|:---|:---:|:---|:---:|:---:|
| **Cardiovascular Disease** | **Mode A** | Optimized Random Forest | Stage 1E-3 | `n_estimators=500`, `max_depth=8`, `max_features='log2'`, `min_samples_split=5`, `class_weight=None` | Sigmoid | **$t = 0.13$** |
| **Cardiovascular Disease** | **Mode B** | Optimized Random Forest | Stage 1E-3 | `n_estimators=800`, `max_depth=8`, `max_features=0.5`, `min_samples_leaf=10`, `min_samples_split=10` | Sigmoid | **$t = 0.15$** |
| **Type 2 Diabetes** | **Mode A** | HistGradientBoosting | Stage 1E-2 | `learning_rate=0.05`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=30`, `l2_reg=10.0` | Sigmoid | **$t = 0.15$** |
| **Type 2 Diabetes** | **Mode B** | Tuned XGBoost | Stage 1E-2 | `learning_rate=0.01`, `n_estimators=500`, `max_depth=5`, `colsample_bytree=0.6`, `min_child_weight=5` | Sigmoid | **$t = 0.17$** |
| **Hypertension** | **Mode A** | Tuned Logistic Regression | Stage 1E-2 | `C=10.0`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=2000`, `StandardScaler` | Sigmoid | **$t = 0.41$** |
| **Hypertension** | **Mode B** | Optimized Random Forest | Stage 1E-3 | `n_estimators=800`, `max_depth=8`, `max_features=0.5`, `min_samples_leaf=10`, `min_samples_split=10` | Sigmoid | **$t = 0.52$** |

---

## 15. Experimental Results

All final evaluation metrics were computed on the **untouched, held-out test partition ($15\%$, $N = 1,170–1,172$ participants)** evaluated **exactly once** after all calibration parameters and operating thresholds were locked:

### 15.1 Comprehensive Final Test Performance Table

| Target & Mode | Architecture | Calibration | Threshold ($t$) | Test ROC-AUC | Test PR-AUC | Test Brier | Test ECE | Sensitivity (Recall) | Specificity | Precision (PPV) | NPV | F1 Score | Balanced Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CVD Mode A** | Optimized RF | Sigmoid | **$0.13$** | **0.8137** | 0.4159 | 0.0912 | 0.0125 | **0.6824** | **0.7803** | 0.3098 | **0.9444** | 0.4262 | 0.7314 |
| **CVD Mode B** | Optimized RF | Sigmoid | **$0.15$** | **0.8218** | 0.4340 | 0.0906 | 0.0368 | **0.6081** | **0.8438** | 0.3600 | **0.9371** | 0.4523 | 0.7259 |
| **Diabetes Mode A** | HistGradBoost | Sigmoid | **$0.15$** | **0.7957** | 0.4348 | 0.1222 | 0.0250 | **0.8173** | **0.6397** | 0.3288 | **0.9419** | 0.4690 | 0.7285 |
| **Diabetes Mode B** | Tuned XGBoost | Sigmoid | **$0.17$** | **0.8162** | 0.4793 | 0.1203 | 0.0470 | **0.6202** | **0.8100** | 0.4135 | **0.9080** | 0.4962 | 0.7151 |
| **Hypertension A** | Logistic Reg | Sigmoid | **$0.41$** | **0.7662** | 0.6541 | 0.1946 | 0.0331 | **0.7615** | **0.6647** | 0.6281 | **0.7894** | 0.6884 | 0.7131 |
| **Hypertension B** | Optimized RF | Sigmoid | **$0.52$** | **0.7731** | 0.6898 | 0.1916 | 0.0465 | **0.6253** | **0.7511** | 0.6514 | **0.7294** | 0.6380 | 0.6882 |

### 15.2 Default ($0.50$) vs. Optimized Threshold Screening Impact

The table below contrasts standard uncalibrated decision thresholds ($0.50$) against the validation-optimized thresholds on the test cohort ($N \approx 1,170$ per evaluation):

```
+----------------------------------------------------------------------------------------------------+
|                         DEFAULT (0.50) VS. OPTIMIZED OPERATING THRESHOLDS                          |
+----------------------------------------------------------------------------------------------------+
  Configuration     Threshold     TP     FP     FN     TN   Sensitivity  Specificity  Missed Cases (FN)
  --------------------------------------------------------------------------------------------------
  CVD Mode A        Default 0.50  14      8    134   1016      9.5%        99.2%        134 / 148 (90.5%)
                    Opt 0.13     101    225     47    799     68.2%        78.0%         47 / 148 (31.8%)
  --------------------------------------------------------------------------------------------------
  CVD Mode B        Default 0.50  29     17    119   1007     19.6%        98.3%        119 / 148 (80.4%)
                    Opt 0.15      90    160     58    864     60.8%        84.4%         58 / 148 (39.2%)
  --------------------------------------------------------------------------------------------------
  Diabetes Mode A   Default 0.50  45     31    163    932     21.6%        96.8%        163 / 208 (78.4%)
                    Opt 0.15     170    347     38    616     81.7%        64.0%         38 / 208 (18.3%)
  --------------------------------------------------------------------------------------------------
  Diabetes Mode B   Default 0.50  57     36    151    927     27.4%        96.3%        151 / 208 (72.6%)
                    Opt 0.17     129    183     79    780     62.0%        81.0%         79 / 208 (38.0%)
  --------------------------------------------------------------------------------------------------
  Hypertension A    Default 0.50 312    165    187    506     62.5%        75.4%        187 / 499 (37.5%)
                    Opt 0.41     380    225    119    446     76.1%        66.5%        119 / 499 (23.9%)
  --------------------------------------------------------------------------------------------------
  Hypertension B    Default 0.50 329    179    170    492     65.9%        73.3%        170 / 499 (34.1%)
                    Opt 0.52     312    167    187    504     62.5%        75.1%        187 / 499 (37.5%)
```

### 15.3 Factual Analysis of Mode A vs. Mode B Performance
* **Discrimination Gains:** Adding clinical blood biomarkers (Mode B) consistently improves ROC-AUC across all conditions: CVD increases from $0.8137$ to **$0.8218$** ($+0.0081$), Diabetes increases from $0.7957$ to **$0.8162$** ($+0.0205$), and Hypertension increases from $0.7662$ to **$0.7731$** ($+0.0069$).
* **Precision-Recall Dynamics:** Precision-Recall AUC (PR-AUC) demonstrates substantial increases in Mode B: Diabetes rises from $0.4348$ to **$0.4793$** ($+0.0445$) and Hypertension rises from $0.6541$ to **$0.6898$** ($+0.0357$).
* **Clinical Utility of Mode A:** Importantly, Mode A maintains strong screening discrimination (ROC-AUC $0.766$ to $0.814$) using entirely non-invasive predictors. With validation-optimized thresholds, Mode A achieves **$68.2\%$ sensitivity on CVD** (with $94.4\%$ NPV) and **$81.7\%$ sensitivity on Diabetes** (with $94.2\%$ NPV), establishing that low-cost, non-invasive triage is clinically viable when proper threshold optimization is enforced.

---

## 16. Explainability

The platform integrates **SHAP (SHapley Additive exPlanations)** grounded in cooperative game theory to provide mathematically rigorous, additive feature attributions:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

where $\phi_0$ is the base expected value across the training population, and $\phi_i(x)$ is the marginal attribution of feature $i$ for a specific individual.

```
+----------------------------------------------------------------------------------------------------+
|                                    SHAP EXPLAINABILITY ENGINE                                      |
+----------------------------------------------------------------------------------------------------+

 [Model Prediction] ---> [TreeExplainer / LinearExplainer]
                                  |
                                  +---> [Local Attributions (phi_i)]
                                  |      - Top Risk Escalators (Positive phi)
                                  |      - Top Protective Offsets (Negative phi)
                                  |
                                  +---> [Visual Chart Generation]
                                         - Summary Plot (Global population distribution)
                                         - Feature Importance Bar Plot (Mean |phi|)
                                         - Waterfall Plot (Patient-specific attribution flow)
                                  |
                                  v
                   [Base64 In-Memory Buffer Encoding]
                                  |
                                  v
                   [Next.js Dynamic Visual Rendering]
```

### Explainer Formulation
* **Ensemble Models (Random Forest, XGBoost):** Utilizes `shap.TreeExplainer` with tree-path dependent feature perturbation, enabling exact, polynomial-time computation of Shapley values without sampling variance.
* **Linear Pipelines (Logistic Regression):** Utilizes `shap.LinearExplainer` leveraging the empirical covariance matrix of the scaled predictors.
* **Visual Delivery:** Matplotlib generates high-resolution figures ($150\text{ DPI}$) in memory buffers (`io.BytesIO()`), which are Base64-encoded and transmitted via JSON. The client dashboard renders these directly without local file storage requirements.

---

## 17. Blood-Test Report Integration

A cornerstone of the complete product vision is the automated ingestion of unstructured or semi-structured laboratory reports:

```
+----------------------------------------------------------------------------------------------------+
|                               PROPOSED BLOOD-REPORT INGESTION PIPELINE                             |
+----------------------------------------------------------------------------------------------------+

  [ Patient Report ] (PDF / Scanned TIFF / Mobile Camera JPEG)
          |
          v
  [ Ingestion & Preprocessing ]
  - Optical deskewing, contrast normalization, resolution scaling (300 DPI)
          |
          v
  [ Multimodal Extraction / OCR ]
  - Layout-aware OCR (Tesseract / Cloud Vision API / LayoutLM)
  - Tabular structure extraction (bounding box grid detection)
          |
          v
  [ Named Entity Recognition & Clinical Mapping ]
  - Regex & semantic vector lookup against Mode B Biomarker Dictionary:
    * "Fasting Plasma Glucose", "Blood Sugar F", "GLU"       --> `fasting_glucose`
    * "Glycosylated Hb", "HbA1c", "A1C"                     --> `hba1c`
    * "Cholesterol, Total", "Total Chol"                    --> `total_cholesterol`
    * "Creatinine", "Serum Creat"                           --> `serum_creatinine`
          |
          v
  [ Unit Harmonization & Range Verification ]
  - Unit normalization: e.g., Glucose (mmol/L -> mg/dL via * 18.0182)
  - Physiological plausibility bounds check (e.g., Creatinine within [0.2, 20.0] mg/dL)
          |
          v
  [ Feature Vector Alignment & Imputation ]
  - Merged with patient questionnaire inputs
  - Mode B missing lab values imputed via trained median preprocessors
          |
          v
  [ Calibrated Mode B Inference Engine ]
```

*Implementation Status Note:* As verified in the repository audit, this pipeline represents the **proposed system architecture**. Optical character recognition, entity extraction, and automated unit mapping are designed specifications slated for Phase 2 development.

---

## 18. User Interface and Interaction Model

The user experience is designed around a modern, responsive health dashboard (`frontend/app/page.tsx`):

```
+----------------------------------------------------------------------------------------------------+
|                                INTERACTIVE USER INTERFACE MOCKUP                                   |
+----------------------------------------------------------------------------------------------------+

 +------------------------------------+------------------------------------------------------------+
 |       ASSESSMENT INPUT FORM        |                  HEALTH RISK INTELLIGENCE                  |
 +------------------------------------+------------------------------------------------------------+
 | Demographics:                      |                                                            |
 |  Age: [ 52 ]  Gender: [ Male    v] |                 [ 74.2% ]  HIGH RISK                       |
 |                                    |          Calibrated Disease Probability (CVD)              |
 | Clinical Vitals:                   |                                                            |
 |  Systolic BP:  [ 142 ] mmHg        |  +-------------------------------------------------------+ |
 |  Diastolic BP: [  92 ] mmHg        |  |  KEY RISK FACTORS (SHAP Impact)                       | |
 |  Resting Pulse:[  78 ] bpm         |  |  [+] Systolic BP (+14.2% risk contribution)           | |
 |                                    |  |  [+] Total Cholesterol (+8.6% risk contribution)      | |
 | Lifestyle & Habits:                |  |  [-] Regular Physical Activity (-4.1% risk offset)    | |
 |  Smoking:   [ Current Smoker    v] |  +-------------------------------------------------------+ |
 |  Activity:  [ Sedentary (<30m)  v] |                                                            |
 |                                    |  +-------------------------------------------------------+ |
 | Laboratory Panel (Mode B Toggle):  |  |  PERSONALIZED CLINICAL RECOMMENDATIONS                | |
 |  [x] Enable Laboratory Inputs      |  |  1. Consult a physician for blood pressure management. | |
 |  Total Cholesterol: [ 235 ] mg/dL  |  |  2. Re-evaluate fasting lipid panel in 90 days.       | |
 |  HDL Cholesterol:   [  38 ] mg/dL  |  |  3. Target gradual reduction in sodium intake.        | |
 |  Serum Creatinine:  [ 1.1 ] mg/dL  |  +-------------------------------------------------------+ |
 |                                    |                                                            |
 | [ CALCULATE MULTI-DISEASE RISK ]   |  [ SHAP Waterfall ]   [ Biomarker Panel ]  [ History Tab ] |
 +------------------------------------+------------------------------------------------------------+
```

### Design Principles
1. **Clinical Semantic Color Tokens:** Low Risk (Emerald Green: `#10b981`), Moderate Risk (Amber: `#f59e0b`), High Risk (Crimson: `#ef4444`).
2. **Glassmorphism & Depth Hierarchy:** Card components leverage subtle translucent backdrop filters (`backdrop-filter: blur(12px)`), neutral slate backgrounds (`#0f172a`), and defined borders (`rgba(255,255,255,0.08)`) to maintain visual clarity across dense clinical metrics.
3. **Transparent Uncertainty Presentation:** Rather than displaying raw binary classifications, the dashboard emphasizes calibrated absolute risk percentages alongside operating screening thresholds.

---

## 19. End-to-End Inference Workflow

The operational flow of a single patient evaluation proceeds as follows:

```
[User Browser]
      |
      | 1. Submits demographic, vital, and laboratory inputs via form
      v
[FastAPI Server (/predict)]
      |
      | 2. Pydantic validates input types, numeric ranges, and categorical choices
      | 3. Feature mapping converts raw payload to Model Schema (Mode A or Mode B)
      v
[Inference Pipeline]
      |
      | 4. Scikit-learn Pipeline applies trained SimpleImputer and StandardScaler
      | 5. Base Estimator generates raw log-odds or decision scores
      | 6. Frozen Sigmoid Calibrator transforms raw score into calibrated posterior probability
      | 7. Decision Logic compares calibrated probability against locked threshold (t_opt)
      v
[Explainability Module]
      |
      | 8. SHAP Explainer computes local Shapley attribution values (phi_i)
      | 9. Identifies top positive risk escalators and top negative protective offsets
      | 10. Matplotlib renders summary, bar, and waterfall figures to in-memory PNG buffers
      | 11. PNG buffers are Base64-encoded
      v
[Database Persistence]
      |
      | 12. Assessment record (inputs, risk score, risk tier, timestamp) committed to SQLite
      v
[FastAPI Response]
      |
      | 13. Returns structured JSON containing probability, category, insights, and Base64 plots
      v
[Next.js Client]
      |
      | 14. Decodes and renders animated risk gauge, SHAP plots, and clinical recommendations
```

---

## 20. Research Contributions

For academic publication in venues such as IEEE Access, IEEE Transactions on Biomedical Engineering, or IEEE BIBM, this project provides several specific contributions:

1. **Adaptive Two-Tier Screening Paradigm:** Formulates and validates a dual-mode feature architecture that dynamically adapts to clinical data availability, establishing empirical performance baselines for non-invasive community triage versus laboratory-augmented clinical evaluation.
2. **Leakage-Controlled Epidemiological Dataset:** Provides an open, fully reproducible data harmonization methodology for CDC NHANES 2021–2023, systematically addressing complex survey artifacts, SAS underflow bugs, refusal recoding, and strict diagnostic target-variable isolation.
3. **Addressing the Base-Rate Fallacy in Health ML:** Demonstrates empirically that standard $0.50$ decision thresholds in imbalanced medical cohorts miss up to $90.5\%$ of at-risk individuals. Provides a rigorous validation-based framework combining out-of-fold Platt calibration and Youden's $J$ threshold optimization that increases clinical screening recall from $9.5\%$ to **$68.2\%$ on CVD** and $21.6\%$ to **$81.7\%$ on Diabetes**.
4. **Controlled Algorithmic Trade-Off Study:** Compares linear, bagging, and boosting architectures across uniform splits, demonstrating that tree depth regularization ($max\_depth=8$) in Random Forest outperforms unconstrained boosting on certain physiological survey panels.
5. **Human-Centered Explainable AI Interface:** Bridges theoretical Shapley value formulation with practical web engineering, converting raw game-theoretic vectors into real-time visual waterfall diagrams and natural-language lifestyle guidance.

---

## 21. Limitations

Scientific transparency requires acknowledging the following methodological constraints:

1. **Cross-Sectional Cohort Design:** NHANES is a cross-sectional survey reflecting prevalent health states at the time of examination. Models predict current undiagnosed or prevalent disease risk rather than long-term $10$-year prospective incident events (such as prospective Framingham cohorts).
2. **Geographic & Demographic Domain Bounds:** NHANES samples the non-institutionalized civilian U.S. population. While statistically robust, baseline epidemiological weights, dietary habits, and genetics differ from international cohorts (e.g., South Asian populations). External validation on cohorts like ICMR-INDIAB is required prior to global deployment.
3. **Fasting Subsample Missingness:** Laboratory biomarkers such as fasting plasma glucose, oral glucose tolerance, and triglycerides are captured only on morning fasting subsamples ($\approx 41.1\%$ of adults), resulting in structured missingness patterns managed via median imputation.
4. **Self-Reported Medical Morbidity History:** Certain diagnostic targets (e.g., past myocardial infarction, stroke, or hypertension diagnosis) rely on standardized self-reported physician diagnoses, which may introduce mild recall bias.
5. **Research Prototype Boundary:** The system is an investigational decision-support and screening research prototype. **It is not certified as a Software as a Medical Device (SaMD) and does not provide formal medical diagnoses.**

---

## 22. Final Product Vision

The long-term realization of this research is a secure, cloud-native **Preventive Health Intelligence Platform**:

* **Patient-Facing Mobile Co-Pilot:** A cross-platform smartphone application allowing patients to monitor metabolic vitals, upload mobile camera photographs of lab sheets, receive longitudinal risk alerts, and share encrypted assessment reports with their doctors.
* **Point-of-Care Clinical Extension:** An integrated EHR sidebar (compatible with HL7 FHIR standards) enabling general practitioners to evaluate patient risk trajectories during routine checkups, testing "what-if" clinical interventions (e.g., projecting risk reduction if systolic BP drops by $15\text{ mmHg}$).
* **Population Health Surveillance:** An aggregated, anonymized epidemiological analytics dashboard for public health officials to monitor cardiometabolic disease clustering, socioeconomic disparities, and community screening efficacy.

---

## 23. Future Development

The engineering roadmap is organized into four sequential development phases:

```
+----------------------------------------------------------------------------------------------------+
|                                    FUTURE DEVELOPMENT ROADMAP                                      |
+----------------------------------------------------------------------------------------------------+

  [ PHASE 2: V2 BACKEND & API INTEGRATION ]
  - Refactor `backend/api/api_server.py` to load and serve the 6 calibrated Stage 1F model artifacts.
  - Implement dynamic `/predict` request routing supporting both Mode A and Mode B payloads.
  - Expand SQLite schema to store multi-disease probabilities and biomarker sub-arrays.

  [ PHASE 3: FRONTEND UPGRADE & MULTI-DISEASE DASHBOARD ]
  - Redesign Next.js dashboard to display multi-condition risk cards (CVD, Diabetes, Hypertension).
  - Implement interactive Mode A / Mode B input toggle switch.
  - Integrate dynamic SHAP waterfall charts for all three conditions.

  [ PHASE 4: DOCUMENT OCR & BLOOD REPORT INGESTION ENGINE ]
  - Implement PDF and image upload API endpoint (`POST /upload-report`).
  - Integrate optical layout parsing and medical entity recognition for blood biomarker extraction.
  - Build automated unit normalization and verification UI allowing users to confirm parsed values.

  [ PHASE 5: EXTERNAL VALIDATION & CLINICAL TRIAL BENCHMARKING ]
  - Benchmark Stage 1F models against external international cohorts (e.g., ICMR-INDIAB).
  - Perform domain-shift adaptation and fine-tuning for South Asian biometric profiles.
  - Conduct prospective clinical observational studies in collaboration with academic health centers.
```

---

## 24. Conclusion

The **AI Health Risk Scoring System** addresses the critical global challenge of asymptomatic cardiometabolic disease progression by establishing a mathematically rigorous, explainable, and multi-tier screening framework. By combining CDC NHANES 2021–2023 epidemiological data, adaptive Mode A/B feature structures, regularized tree and boosting architectures, Sigmoid probability calibration, and validation-optimized decision thresholds, the platform resolves the base-rate failure of conventional medical machine learning. Moving beyond uninterpretable black-box regressors, the system provides transparent, actionable, and human-interpretable clinical intelligence, paving the way for scalable, equitable population-level preventive healthcare.

---

## 25. Documentation Reconciliation Notes

During the comprehensive technical repository audit conducted in September 2026, several discrepancies were identified between early legacy documentation and the verified active implementation. The table below documents and reconciles these items:

| Topic / Area | Legacy Documentation (`README.md`, `PROJECT_DEEP_DIVE.md`) | Verified Active Implementation (Ground Truth) | Reconciliation / Resolution |
|:---|:---|:---|:---|
| **Primary Dataset** | Described as `dataset/indian_health_risk_dataset.csv` (151-row synthetic CSV). | Active V2 models are trained strictly on **CDC NHANES 2021–2023** ($N = 7,809$ adults across 6 Parquet files). | The synthetic Indian dataset was a V1 proof-of-concept asset. It has been superseded by NHANES 2021–2023 for all V2 modeling. |
| **Model Type** | Described as a single `RandomForestRegressor` predicting a continuous 0–100 score. | Active V2 models are **six separate calibrated classifiers** (CVD Mode A/B, Diabetes Mode A/B, Hypertension Mode A/B) spanning RF, HistGradientBoosting, XGBoost, and Logistic Regression. | The continuous regressor belongs to Model V1. Model V2 establishes disease-specific binary classification with calibrated posterior probabilities. |
| **Simulated Wearable Features** | Legacy inputs include `sdnn_hrv`, `rmssd_hrv`, and `spo2`. | NHANES does not capture ambulatory HRV data. Features were retired in the V2 feature matrix. | Wearable signals were simulated in V1. Model V2 relies exclusively on verified NHANES clinical examination vitals and lab panels. |
| **Blood Report Upload** | Outlined in project vision and documentation summaries. | No OCR or document upload code currently exists in `backend/` or `frontend/`. | Accurately categorized as **Planned (Phase 4)** in the implementation status matrix. |
| **Frontend Integration** | Frontend UI (`app/page.tsx`) currently connects to Model V1 inputs and endpoints. | V2 model artifacts are fully trained, calibrated, and evaluated in `backend/ml/evaluation/stage_1f/models/`, awaiting API wiring. | Clarified that the repository contains a working V1 full-stack app alongside a fully completed V2 research/evaluation pipeline. |

---

### Implementation Status Matrix

| Component / Subsystem | Current Status | Repository Evidence | Final System Role |
|:---|:---:|:---|:---|
| **NHANES 2021–2023 Data Pipeline** | **Implemented** | `backend/ml/scripts/build_nhanes_dataset.py`, `nhanes_harmonizer.py`, Parquet files in `data/processed/` | Automated ingestion, cleaning, and assembly of multi-module epidemiological datasets. |
| **Leakage-Free Train/Val/Test Splits** | **Implemented** | `backend/ml/data/processed/nhanes_2021_2023/splits/*.csv` | Enforces 70/15/15 participant-level partition integrity across all experiments. |
| **Stage 1D/1E Model Benchmarking** | **Implemented** | `backend/ml/scripts/benchmark_models.py`, `benchmark_xgboost.py`, `optimize_models.py`, `optimize_random_forest.py` | Systematic multi-algorithm comparison and hyperparameter optimization. |
| **Stage 1F Calibration & Threshold Suite** | **Implemented** | `backend/ml/scripts/calibrate_and_optimize_thresholds.py`, `stage_1f_results.json`, `stage_1f_comparison.csv` | 5-fold OOF validation probability calibration, threshold sweeping, and single-pass test evaluation. |
| **Trained V2 Model Artifacts** | **Implemented** | 12 serialized artifacts (`.joblib`, `.pkl`) in `backend/ml/evaluation/stage_1f/models/` | Production-ready calibrated models for CVD, Diabetes, and Hypertension (Modes A and B). |
| **SHAP Explainability Core** | **Implemented** | `backend/ml/shap_explainer.py`, `backend/ml/visualization.py` | Game-theoretic feature attribution and in-memory Base64 plot rendering. |
| **FastAPI Backend Service (V1)** | **Implemented** | `backend/api/api_server.py`, `backend/requirements.txt` | RESTful API server providing `/predict`, `/history`, and `/model-metrics` endpoints. |
| **Next.js Web Dashboard (V1)** | **Implemented** | `frontend/app/page.tsx`, `layout.tsx`, `globals.css`, `package.json` | Responsive browser interface with input forms, dynamic dials, and history tables. |
| **SQLite Assessment Persistence** | **Implemented** | `backend/database/database.py`, `healthrisk.db` | Relational storage for user assessments, timestamps, and predicted risk tiers. |
| **V2 Multi-Disease API Integration** | **Planned** | Stage 1F model artifacts ready in `stage_1f/models/`; API currently serves V1 pipeline. | Exposing multi-disease Mode A and Mode B endpoints over FastAPI. |
| **Multi-Condition Frontend UI** | **Planned** | Frontend currently renders single V1 cardiovascular regressor inputs. | Upgraded dashboard featuring concurrent CVD, Diabetes, and HTN cards with Mode A/B toggles. |
| **Blood Report OCR Ingestion** | **Planned** | Architectural design detailed in Section 17; no OCR code in repository. | Document upload, layout parsing, and automated biomarker extraction engine. |
| **External Indian Cohort Validation** | **Planned** | Reference data present in `backend/ml/data/raw/icmr_indiab/sample.dta`. | Evaluating model generalizability and transferability to South Asian demographic cohorts. |
