# UI / Frontend & Demo Flow Document

## Project Title

**Sensor-Augmented Personalized Health Risk Scoring Using Explainable Machine Learning**

---

# 1. Objective

The objective of this document is to describe:

* the frontend design of the prototype,
* Streamlit-based user interface workflow,
* user interaction process,
* input and output visualization,
* and the demonstration flow used during project presentation.

The frontend is designed to provide a simple, interactive, and interpretable healthcare analytics experience for users.

---

# 2. Frontend Technology

The prototype frontend is developed using:

| Component          | Technology    |
| ------------------ | ------------- |
| Frontend Framework | Streamlit     |
| Visualization      | Matplotlib    |
| ML Integration     | scikit-learn  |
| Explainability     | SHAP          |
| Data Processing    | pandas, NumPy |

---

# 3. Frontend Objectives

The frontend is designed to:

* collect healthcare inputs,
* send data to the ML prediction pipeline,
* display personalized health risk scores,
* visualize SHAP explainability,
* and provide an easy-to-understand healthcare dashboard.

---

# 4. Streamlit Application Workflow

```text id="ecp78x"
User Opens Streamlit App
            ↓
Health Input Form
            ↓
Submit / Predict Button
            ↓
ML Model Prediction
            ↓
Risk Score Generation
            ↓
SHAP Explainability Analysis
            ↓
Dashboard Output Display
```

---

# 5. User Interaction Flow

The application follows a simple step-by-step interaction process.

---

## Step 1 — User Opens Application

The user launches the Streamlit application locally.

Example:

```bash id="zud2f4"
streamlit run app.py
```

---

## Step 2 — User Enters Health Information

The user provides:

* demographic details,
* lifestyle information,
* physiological indicators.

---

## Step 3 — User Clicks Predict

The entered information is:

* preprocessed,
* formatted,
* and passed to the trained Random Forest model.

---

## Step 4 — Risk Prediction is Generated

The system displays:

* numerical risk score,
* risk category.

---

## Step 5 — SHAP Explainability Display

SHAP visualizations explain:

* which features increased risk,
* which features reduced risk,
* and overall feature importance.

---

# 6. Input Design

The frontend contains interactive input components for healthcare data collection.

---

# 7. Input Components

| Feature             | UI Component |
| ------------------- | ------------ |
| Age                 | Slider       |
| Gender              | Dropdown     |
| BMI                 | Number Input |
| Blood Pressure      | Slider/Input |
| Cholesterol         | Slider/Input |
| Smoking             | Radio Button |
| Alcohol Consumption | Radio Button |
| Physical Activity   | Dropdown     |
| Family History      | Radio Button |
| Heart Rate          | Slider       |
| SDNN                | Number Input |
| RMSSD               | Number Input |
| SpO₂                | Slider       |

---

# 8. Example Input Screen

```text id="73fdpr"
----------------------------------------
 Personalized Health Risk Dashboard
----------------------------------------

Age:                  [ 55 ]
Gender:               [ Male ▼ ]
BMI:                  [ 31.2 ]
Systolic BP:          [ 145 ]
Cholesterol:          [ 250 ]
Smoking:              (•) Yes
Physical Activity:    [ Low ▼ ]
Family History:       (•) Yes

            [ Predict Risk ]
```

---

# 9. Output Design

The system displays:

* health risk score,
* risk category,
* SHAP explainability,
* and recommendations.

---

# 10. Example Output Screen

```text id="jlwm7m"
----------------------------------------
        Prediction Results
----------------------------------------

Risk Score: 84

Risk Category: HIGH RISK

Top Risk Contributors:
+ High Cholesterol
+ Smoking Habit
+ High BMI
+ Low Physical Activity

Recommendations:
- Increase daily physical activity
- Reduce cholesterol intake
- Monitor BP regularly
- Avoid smoking
```

---

# 11. SHAP Visualization Section

The dashboard includes:

* SHAP feature importance plots,
* summary charts,
* contribution analysis.

---

## SHAP Visualizations Included

| Visualization     | Purpose                           |
| ----------------- | --------------------------------- |
| SHAP Summary Plot | Overall feature importance        |
| Bar Plot          | Top contributors                  |
| Waterfall Plot    | Individual prediction explanation |

---

# 12. Wireframe Layout

## High-Level Dashboard Wireframe

```text id="e5d1ee"
+------------------------------------------------+
|        Personalized Health Dashboard           |
+------------------------------------------------+

+----------------+   +--------------------------+
| User Inputs    |   | Prediction Results       |
|----------------|   |--------------------------|
| Age            |   | Risk Score               |
| BMI            |   | Risk Category            |
| BP             |   | SHAP Insights            |
| Cholesterol    |   | Recommendations          |
| Smoking        |   |                          |
+----------------+   +--------------------------+

+------------------------------------------------+
|           SHAP Explainability Charts           |
+------------------------------------------------+
```

---

# 13. Demo Flow

The following workflow is used during project demonstration.

---

# 14. Demo Walkthrough

## Step 1 — Launch Application

Open the Streamlit dashboard.

Example:

```bash id="n0g08m"
streamlit run app.py
```

---

## Step 2 — Explain Project Objective

Briefly explain:

* personalized healthcare prediction,
* explainable AI,
* preventive healthcare analytics.

---

## Step 3 — Enter Sample User Data

Example:

| Feature     | Value |
| ----------- | ----- |
| Age         | 55    |
| BMI         | 31    |
| Cholesterol | 250   |
| Smoking     | Yes   |
| Activity    | Low   |

---

## Step 4 — Click Predict

The model processes input features and generates:

* risk score,
* risk category.

---

## Step 5 — Explain Results

Discuss:

* why the score is high,
* which features contributed most,
* and healthcare implications.

---

## Step 6 — Display SHAP Graphs

Show:

* feature importance,
* positive contributors,
* negative contributors.

Explain how SHAP improves transparency.

---

# 15. Expected Frontend Outputs

The frontend is expected to provide:

* intuitive healthcare input collection,
* real-time prediction display,
* interpretable ML explanations,
* and user-friendly visualization.

---

# 16. Future Improvements

Possible frontend enhancements include:

* real-time wearable integration,
* longitudinal trend analysis,
* dark/light theme support,
* cloud deployment,
* mobile-responsive dashboard,
* and personalized healthcare recommendations.

---

# 17. Conclusion

The Streamlit-based frontend successfully demonstrates the feasibility of an interactive explainable healthcare analytics dashboard.

The interface integrates:

* machine learning prediction,
* SHAP explainability,
* and healthcare visualization

into a user-friendly Proof-of-Concept application for personalized health risk assessment.
