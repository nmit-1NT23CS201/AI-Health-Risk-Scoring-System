# AI Health Risk Scoring System (Phase 1)

Phase 1 implements a modular machine learning pipeline to load data, preprocess features, train a Random Forest model, evaluate metrics, and export predictions.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional flags:

```bash
python main.py --test-size 0.2 --random-state 42
```

## Outputs

- models/random_forest_model.pkl
- outputs/evaluation_metrics.txt
- outputs/predictions.csv
