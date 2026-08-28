# AI Health Risk Scoring System

## Overview

The AI Health Risk Scoring System is an intelligent healthcare analytics platform that predicts an individual's cardiovascular health risk using machine learning techniques. The system combines a trained Random Forest model with explainable AI (SHAP) visualizations and an interactive dashboard to provide risk predictions and actionable health insights.

---

## Features

### Machine Learning Risk Prediction

* Random Forest based health risk prediction model
* Predicts cardiovascular risk score from patient health parameters
* Supports real-time risk assessment

### Explainable AI

* SHAP (SHapley Additive Explanations) integration
* Displays feature contributions influencing predictions
* Improves transparency and interpretability

### Interactive Dashboard

* Built using Next.js
* Modern responsive user interface
* Dynamic patient input forms
* Real-time prediction updates

### Health Insights

* Risk categorization:

  * Low Risk
  * Medium Risk
  * High Risk
* Personalized health recommendations
* Key contributing factors analysis

### Data Processing Pipeline

* Data loading and preprocessing
* Feature engineering
* Model training and evaluation
* Prediction generation

---

## Technology Stack

### Frontend

* Next.js
* TypeScript
* React

### Backend

* FastAPI
* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* SHAP

### Data Processing

* Pandas
* NumPy

---

## Project Structure

```text
backend/
├── api/                FastAPI and Streamlit entry points
├── database/           SQLite database and access module
├── ml/                 ML and explainability modules
├── models/             Trained ML model artifacts
├── outputs/            Evaluation and generated outputs
└── requirements.txt    Python dependencies
frontend/               Next.js dashboard and frontend configuration
dataset/                Dataset files (kept at project root)
docs/                   Project documentation
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd AI-Health-Risk-Scoring-System
```

### Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

### Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## Running the Application

### Start Backend

```bash
python -m uvicorn backend.api.api_server:app --reload
```

or

```bash
python -m streamlit run backend/api/app.py
```

### Start Frontend

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

---

## Model Outputs

* Risk Score Prediction
* Risk Category Classification
* SHAP Feature Importance
* Personalized Health Insights
* Actionable Recommendations

---

## Future Enhancements

* User authentication
* Health record integration
* Advanced clinical analytics
* Deployment on cloud infrastructure
* Enhanced visualization modules

---

## Authors

Developed as part of the AI Health Risk Scoring System academic project.
