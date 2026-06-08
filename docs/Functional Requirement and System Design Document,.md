# Functional Requirement + System Design Document

## Project Title

**Sensor-Augmented Personalized Health Risk Scoring Using Explainable Machine Learning**

---

# 1. Objective

The objective of this Proof-of-Concept (PoC) is to implement a simplified AI-driven healthcare risk prediction system capable of:

* Predicting personalized health risk scores using clinical and lifestyle data.
* Providing explainable insights using SHAP-based feature contribution analysis.
* Demonstrating the feasibility of explainable machine learning in preventive healthcare.

This implementation focuses on the execution of selected functional requirements from the project synopsis.

---

# 2. Selected Functional Requirements

## FR5 — Health Risk Prediction

> The system shall apply a trained Random Forest or XGBoost model to compute a numeric health risk score in the 0–100 range, mapping to Low, Medium, or High risk levels.

### Expected Outcome

* The system generates:

  * Risk Score (0–100)
  * Risk Category:

    * Low
    * Medium
    * High

---

## FR6 — SHAP-based Explainability

> The system shall employ SHAP-based explainability to identify top contributing features for each prediction.

### Expected Outcome

* The system identifies:

  * Features increasing risk
  * Features reducing risk
* SHAP visualizations provide model interpretability.

---

# 3. Scope of Implementation

The current implementation is limited to a prototype-level Proof-of-Concept focused on validating the ML prediction and explainability pipeline.

## Included in Scope

* Synthetic healthcare dataset generation
* Data preprocessing
* Random Forest model training
* Risk score prediction
* SHAP explainability generation
* Streamlit-based user interface

## Excluded from Scope

* Real-time wearable integration
* Live PPG signal acquisition
* Cloud deployment
* Authentication systems
* Continuous retraining
* Full-scale medical validation

---

# 4. System Workflow

The proposed workflow follows a simplified AI healthcare analytics pipeline.

```text
User Inputs
   ↓
Feature Preprocessing
   ↓
Random Forest ML Model
   ↓
Health Risk Score Prediction
   ↓
SHAP Explainability Layer
   ↓
Risk Dashboard Output
```

---

# 5. System Architecture

## High-Level Architecture Diagram

```text
+----------------------+
|   User Interface     |
|     (Streamlit)      |
+----------+-----------+
           |
           v
+----------------------+
|  User Health Inputs  |
|----------------------|
| Age                  |
| BMI                  |
| Blood Pressure       |
| Cholesterol          |
| Smoking Status       |
| Physical Activity    |
| Family History       |
+----------+-----------+
           |
           v
+----------------------+
| Data Preprocessing   |
|----------------------|
| Encoding             |
| Scaling              |
| Feature Formatting   |
+----------+-----------+
           |
           v
+----------------------+
| Random Forest Model  |
|----------------------|
| Risk Score Prediction|
| Risk Classification  |
+----------+-----------+
           |
           v
+----------------------+
| SHAP Explainability  |
|----------------------|
| Feature Contribution |
| Importance Analysis  |
+----------+-----------+
           |
           v
+----------------------+
| Dashboard Output     |
|----------------------|
| Risk Score           |
| Risk Level           |
| SHAP Graphs          |
| Recommendations      |
+----------------------+
```

---

# 6. Data Pipeline

The system processes healthcare-related features through multiple stages before prediction.

## Step 1 — Data Collection

The system collects:

* Demographic data
* Lifestyle information
* Clinical health indicators

### Input Features

| Feature             | Description            |
| ------------------- | ---------------------- |
| Age                 | Patient age            |
| Gender              | Male/Female            |
| BMI                 | Body Mass Index        |
| Systolic BP         | Upper blood pressure   |
| Diastolic BP        | Lower blood pressure   |
| Cholesterol         | Cholesterol level      |
| Smoking             | Smoking habit          |
| Alcohol Consumption | Alcohol usage          |
| Physical Activity   | Activity level         |
| Family History      | Genetic risk indicator |
| Heart Rate          | BPM                    |
| SDNN                | HRV metric             |
| RMSSD               | HRV metric             |
| SpO₂                | Oxygen saturation      |

---

## Step 2 — Data Preprocessing

Preprocessing operations include:

* Handling categorical values
* Feature encoding
* Data normalization
* Input formatting

---

## Step 3 — Machine Learning Prediction

The Random Forest model:

* Processes the feature vector
* Predicts a risk score
* Assigns a risk category

### Risk Categories

| Score Range | Risk Level |
| ----------- | ---------- |
| 0–34        | Low        |
| 35–69       | Medium     |
| 70–100      | High       |

---

## Step 4 — Explainability Layer

SHAP computes:

* Positive contributors to risk
* Negative contributors to risk
* Feature-wise impact on prediction

This improves:

* transparency,
* interpretability,
* and trust in the ML model.

---

# 7. Technology Stack

| Component            | Technology    |
| -------------------- | ------------- |
| Programming Language | Python 3.10   |
| ML Framework         | scikit-learn  |
| Explainability       | SHAP          |
| Data Processing      | pandas, NumPy |
| Visualization        | Matplotlib    |
| Frontend             | Streamlit     |

---

# 8. Expected Outputs

The final prototype is expected to generate:

## 1. Health Risk Score

Example:

```text
Risk Score: 82
```

---

## 2. Risk Category

Example:

```text
High Risk
```

---

## 3. SHAP Explainability

Example:

```text
Top Risk Contributors:
+ High Cholesterol
+ Smoking Habit
+ High BMI
- Moderate Physical Activity
```

---

## 4. Dashboard Visualization

The dashboard will display:

* User health inputs
* Risk prediction
* Risk category
* SHAP feature importance plots
* Personalized recommendations

---

# 9. Conclusion

This prototype demonstrates the feasibility of combining machine learning and explainable AI for personalized healthcare risk prediction. By integrating health-related inputs with interpretable ML predictions, the system provides a transparent and user-centric approach to preventive digital healthcare.

The PoC validates:

* ML-based risk scoring,
* explainability integration,
* and dashboard-based healthcare analytics,

forming the foundation for future real-time wearable integration and advanced healthcare intelligence systems.
