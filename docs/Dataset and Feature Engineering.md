# Dataset & Feature Engineering Document

## Project Title

**Sensor-Augmented Personalized Health Risk Scoring Using Explainable Machine Learning**

---

# 1. Objective

The purpose of this document is to describe:

* the dataset structure used in the prototype,
* feature selection methodology,
* Indian healthcare assumptions used during synthetic data generation,
* preprocessing pipeline,
* and feature engineering techniques applied before machine learning prediction.

The dataset is designed specifically for demonstrating the feasibility of personalized health risk scoring using explainable machine learning.

---

# 2. Dataset Schema

The dataset consists of demographic, clinical, lifestyle, and physiological health-related attributes.

## Dataset Structure

| Column Name         | Data Type   | Description                 |
| ------------------- | ----------- | --------------------------- |
| patient_id          | String      | Unique patient identifier   |
| name                | String      | Patient name                |
| state               | String      | Indian state/region         |
| age                 | Integer     | Age of patient              |
| gender              | Categorical | Male / Female               |
| bmi                 | Float       | Body Mass Index             |
| systolic_bp         | Integer     | Systolic blood pressure     |
| diastolic_bp        | Integer     | Diastolic blood pressure    |
| cholesterol_mg_dl   | Integer     | Cholesterol level           |
| smoking             | Categorical | Smoking habit (Yes/No)      |
| alcohol_consumption | Categorical | Alcohol consumption status  |
| physical_activity   | Categorical | Activity level              |
| family_history      | Categorical | Family medical history      |
| heart_rate_bpm      | Integer     | Heart rate                  |
| sdnn_hrv            | Float       | HRV metric (SDNN)           |
| rmssd_hrv           | Float       | HRV metric (RMSSD)          |
| spo2                | Float       | Blood oxygen saturation     |
| risk_score          | Float       | Predicted health risk score |
| risk_level          | Categorical | Low / Medium / High         |

---

# 3. Indian Healthcare Assumptions

The synthetic dataset was generated to mimic realistic health trends commonly observed in Indian populations.

## Assumptions Considered

### 1. Higher Cardiovascular Risk with Age

* Blood pressure and cholesterol increase gradually with age.
* Older individuals are more likely to fall into Medium or High risk categories.

---

### 2. Urban Lifestyle Impact

The dataset assumes:

* sedentary lifestyle prevalence,
* lower physical activity levels,
* and rising obesity rates in urban populations.

This affects:

* BMI,
* cholesterol,
* and cardiovascular indicators.

---

### 3. Smoking and Alcohol Trends

The dataset assumes:

* moderate smoking prevalence,
* moderate alcohol consumption,
* stronger risk contribution for smokers.

Smoking significantly increases:

* cardiovascular risk,
* blood pressure,
* and overall health risk score.

---

### 4. Physiological Signal Relationships

The dataset simulates realistic physiological trends:

* lower HRV values in high-risk individuals,
* lower SpO₂ in unhealthy cases,
* elevated heart rate in stressed or unhealthy patients.

---

### 5. Family History Influence

Patients with family history are assumed to have:

* elevated baseline risk,
* higher likelihood of chronic disease indicators.

---

# 4. Synthetic Dataset Generation Logic

The dataset was generated programmatically using:

* Python,
* NumPy,
* pandas,
* and Faker libraries.

---

## Generation Strategy

### Step 1 — Demographic Generation

Random demographic attributes were generated:

* age,
* gender,
* Indian state.

Age range:

```text id="3aq4m6"
18–80 years
```

---

### Step 2 — BMI Distribution

BMI values were generated using a normal distribution centered around:

```text id="plxw4q"
Mean BMI ≈ 25
```

Higher BMI values contribute to:

* hypertension,
* cholesterol increase,
* elevated risk scores.

---

### Step 3 — Blood Pressure Generation

Blood pressure values were correlated with:

* age,
* BMI.

Example logic:

```text id="wl6qot"
Higher age + Higher BMI
→ Higher BP probability
```

---

### Step 4 — Cholesterol Generation

Cholesterol values increase gradually with:

* age,
* obesity,
* unhealthy lifestyle patterns.

---

### Step 5 — Lifestyle Habit Generation

Lifestyle attributes were probabilistically generated:

* smoking,
* alcohol consumption,
* physical activity levels.

Weighted probabilities were used to simulate realistic distributions.

---

### Step 6 — Physiological Metrics

Synthetic wearable-style metrics were generated:

* Heart Rate
* SDNN
* RMSSD
* SpO₂

Low HRV values were correlated with:

* older age,
* higher health risk.

---

### Step 7 — Risk Score Computation

A weighted scoring logic was used.

Features contributing positively to risk:

* age,
* BMI,
* cholesterol,
* smoking,
* low activity,
* family history,
* low HRV,
* low SpO₂.

The final score was normalized to:

```text id="4jv1h5"
0–100
```

---

# 5. Feature Engineering

Feature engineering was performed to prepare data for machine learning prediction.

---

## Selected Features for ML Model

| Feature Category | Features                      |
| ---------------- | ----------------------------- |
| Demographic      | Age, Gender                   |
| Clinical         | BMI, BP, Cholesterol          |
| Lifestyle        | Smoking, Alcohol, Activity    |
| Physiological    | Heart Rate, SDNN, RMSSD, SpO₂ |
| Genetic          | Family History                |

---

## Target Variables

| Target     | Description                      |
| ---------- | -------------------------------- |
| risk_score | Numerical health risk prediction |
| risk_level | Risk classification category     |

---

# 6. Preprocessing Steps

The dataset undergoes preprocessing before training the ML model.

---

## Step 1 — Handling Categorical Variables

Categorical features such as:

* gender,
* smoking,
* activity level

are encoded into numerical format using label encoding or one-hot encoding.

---

## Step 2 — Feature Scaling

Numerical values are normalized/scaled to ensure:

* stable ML training,
* balanced feature contribution.

---

## Step 3 — Data Cleaning

Basic cleaning operations include:

* removal of invalid values,
* range validation,
* formatting consistency.

---

## Step 4 — Feature Formatting

Features are combined into a unified feature vector before model training.

Example:

```text id="e83vv1"
[Age, BMI, BP, Cholesterol, Smoking, Activity, HRV, SpO₂]
```

---

# 7. Risk Classification Logic

The generated risk score is categorized into three classes.

| Risk Score Range | Risk Level |
| ---------------- | ---------- |
| 0–34             | Low        |
| 35–69            | Medium     |
| 70–100           | High       |

---

# 8. Expected Outcome

The final processed dataset supports:

* machine learning training,
* risk prediction,
* SHAP explainability,
* and dashboard visualization.

The dataset acts as a healthcare-oriented Proof-of-Concept foundation for:

* preventive analytics,
* explainable AI,
* and personalized health scoring systems.

---

# 9. Conclusion

The synthetic dataset successfully simulates realistic healthcare trends relevant to Indian populations while maintaining sufficient complexity for machine learning experimentation.

By integrating:

* demographic,
* lifestyle,
* clinical,
* and physiological indicators,

the dataset provides a strong foundation for implementing personalized health risk scoring and explainable AI-based healthcare analytics.