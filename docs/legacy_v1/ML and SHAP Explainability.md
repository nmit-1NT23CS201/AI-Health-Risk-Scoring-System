# ML Model & SHAP Explainability Document

## Project Title

**Sensor-Augmented Personalized Health Risk Scoring Using Explainable Machine Learning**

---

# 1. Objective

The objective of this document is to describe:

* the machine learning model used for health risk prediction,
* the model training workflow,
* prediction logic,
* SHAP-based explainability integration,
* and evaluation metrics used in the prototype.

This document supports the implementation of:

* FR5 — Health Risk Prediction
* FR6 — SHAP Explainability

---

# 2. Why Random Forest?

The prototype uses the **Random Forest Regressor** algorithm for predicting personalized health risk scores.

---

## Reasons for Selecting Random Forest

### 1. Handles Mixed Healthcare Features Effectively

The dataset contains:

* numerical features,
* categorical features,
* physiological indicators,
* lifestyle attributes.

Random Forest performs well on heterogeneous healthcare data.

---

### 2. Robust Against Overfitting

Random Forest combines predictions from multiple decision trees, reducing:

* variance,
* instability,
* and overfitting.

This makes it suitable for healthcare prediction tasks.

---

### 3. Works Well with Small Datasets

The prototype uses a relatively small synthetic dataset.

Random Forest:

* performs reliably on small-to-medium datasets,
* requires minimal hyperparameter tuning,
* and provides stable results.

---

### 4. Compatible with SHAP Explainability

Tree-based models work efficiently with:

* SHAP TreeExplainer,
* feature contribution analysis,
* and interpretable prediction pipelines.

This makes Random Forest ideal for explainable healthcare AI systems.

---

# 3. Machine Learning Workflow

The ML pipeline follows multiple stages before prediction.

---

# 4. Training Process

## Step 1 — Dataset Loading

The dataset is loaded from a CSV file using pandas.

Example:

```python
df = pd.read_csv("indian_health_risk_dataset.csv")
```

---

## Step 2 — Feature Selection

Selected healthcare features are extracted.

### Input Features

| Category      | Features                   |
| ------------- | -------------------------- |
| Demographic   | Age, Gender                |
| Clinical      | BMI, BP, Cholesterol       |
| Lifestyle     | Smoking, Alcohol, Activity |
| Physiological | HR, SDNN, RMSSD, SpO₂      |
| Genetic       | Family History             |

---

## Step 3 — Encoding

Categorical variables are converted into numerical form.

Examples:

* Yes/No → 1/0
* Male/Female → encoded values

Encoding ensures compatibility with the ML model.

---

## Step 4 — Train-Test Split

The dataset is divided into:

* training set,
* testing set.

Typical split:

```text id="o7b3nf"
80% Training
20% Testing
```

This allows performance evaluation on unseen data.

---

## Step 5 — Model Training

The Random Forest Regressor is trained using the training dataset.

Example:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

---

## Step 6 — Prediction

The trained model predicts:

* numerical risk score,
* associated health risk level.

Example:

```text id="d7gm79"
Predicted Risk Score = 82
```

---

# 5. Prediction Logic

The ML model predicts a personalized health risk score using combined healthcare indicators.

---

## High-Risk Indicators

The following attributes generally increase risk:

* high cholesterol,
* smoking habit,
* high BMI,
* high blood pressure,
* low physical activity,
* low HRV,
* family medical history.

---

## Low-Risk Indicators

The following attributes generally reduce risk:

* healthy BMI,
* high physical activity,
* normal blood pressure,
* healthy HRV values,
* good SpO₂ levels.

---

# 6. Risk Classification Logic

The predicted numerical score is mapped into risk categories.

| Score Range | Risk Level |
| ----------- | ---------- |
| 0–34        | Low        |
| 35–69       | Medium     |
| 70–100      | High       |

---

# 7. SHAP Explainability Overview

## What is SHAP?

SHAP (SHapley Additive exPlanations) is an Explainable AI (XAI) technique used to interpret machine learning predictions.

It explains:

* why the model produced a prediction,
* which features contributed most,
* and how strongly each feature influenced the result.

---

## Why SHAP is Important in Healthcare

Healthcare systems require:

* transparency,
* interpretability,
* and trust.

Black-box predictions are difficult for:

* doctors,
* patients,
* and healthcare analysts to trust.

SHAP helps solve this by providing:

* feature-wise reasoning,
* understandable explanations,
* and transparent decision-making.

---

# 8. SHAP Integration Workflow

```text id="s1jlwm"
Input Features
      ↓
Random Forest Prediction
      ↓
SHAP TreeExplainer
      ↓
Feature Contribution Analysis
      ↓
Visualization & Interpretation
```

---

# 9. SHAP Explainability Process

## Step 1 — Initialize Explainer

```python
explainer = shap.TreeExplainer(model)
```

---

## Step 2 — Generate SHAP Values

```python
shap_values = explainer.shap_values(X_test)
```

---

## Step 3 — Visualize Explanations

SHAP generates:

* feature importance plots,
* waterfall plots,
* summary visualizations.

These explain how each feature impacts prediction.

---

# 10. Sample SHAP Explanations

## Example Prediction

### Predicted Output

```text id="4mjlwm"
Risk Score = 84
Risk Level = High
```

---

## Top Positive Contributors

| Feature               | Contribution   |
| --------------------- | -------------- |
| High Cholesterol      | Increased Risk |
| Smoking Habit         | Increased Risk |
| High BMI              | Increased Risk |
| Low Physical Activity | Increased Risk |

---

## Top Negative Contributors

| Feature            | Contribution |
| ------------------ | ------------ |
| Good SpO₂          | Reduced Risk |
| Healthy Heart Rate | Reduced Risk |

---

# 11. Explainability Benefits

SHAP improves:

* transparency,
* interpretability,
* user trust,
* and healthcare decision support.

It enables users to understand:

* what caused the prediction,
* and which health factors require improvement.

---

# 12. Evaluation Metrics

The prototype evaluates model performance using standard ML metrics.

---

## Regression Metrics

| Metric   | Purpose                  |
| -------- | ------------------------ |
| MAE      | Mean prediction error    |
| MSE      | Squared prediction error |
| RMSE     | Root mean squared error  |
| R² Score | Model fit quality        |

---

## Classification Metrics

Risk levels can additionally be evaluated using:

| Metric    | Purpose                     |
| --------- | --------------------------- |
| Accuracy  | Correct predictions         |
| Precision | Positive prediction quality |
| Recall    | Sensitivity                 |
| F1-Score  | Balanced performance        |

---

# 13. Expected Outputs

The ML + SHAP pipeline produces:

* personalized health risk score,
* risk classification,
* explainable feature contributions,
* and healthcare-oriented insights.

---

# 14. Conclusion

The implemented machine learning pipeline demonstrates the feasibility of AI-driven personalized healthcare risk prediction using explainable machine learning techniques.

The combination of:

* Random Forest prediction,
* healthcare feature analysis,
* and SHAP explainability

creates a transparent and interpretable healthcare analytics system suitable for Proof-of-Concept validation and future expansion into wearable-integrated preventive healthcare applications.
