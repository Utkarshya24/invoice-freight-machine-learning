# Invoice & Freight Machine Learning

This project contains two ML pipelines built on `data/inventory.db`:

- `Freight Cost Prediction` (regression): predicts freight amount from invoice dollars.
- `Invoice Flagging` (classification): predicts whether an invoice should be flagged for review.

It also includes a Streamlit web app to run both models from a UI.

## Project Structure

```text
.
├── app.py
├── data/
│   └── inventory.db
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modelig_evolution.py
│   └── train.py
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
└── notebooks/
    ├── predicting_freight_cost.ipynb
    └── invoice_flagging.ipynb
```

## Tech Stack

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib
- sqlite3 (stdlib)
- streamlit

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib streamlit
```

Or install from file:

```bash
pip install -r requirements.txt
```

## Data Source

- SQLite DB: `data/inventory.db`
- Main tables used:
  - `vendor_invoice` (freight regression + part of flagging features)
  - `purchases` (aggregated for flagging features)

## Model 1: Freight Cost Prediction

Path: `freight_cost_prediction/`

### Feature/Target

- Feature: `Dollars`
- Target: `Freight`

### Models Trained

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Selection Logic

- Evaluate MAE / RMSE / R2
- Best model chosen by lowest MAE in `freight_cost_prediction/train.py`

### Train Command

Run from repo root:

```bash
python freight_cost_prediction/train.py
```

### Output Artifact

- `models/predict_freight_model.pkl`

## Model 2: Invoice Flagging

Path: `invoice_flagging/`

### Labeling Rule

`flag_invoice = 1` when either condition is true:

- `abs(invoice_dollars - total_item_dollars) > 5`
- `avg_receiving_delay > 10`

Else `flag_invoice = 0`.

### Features Used

- `invoice_quantity`
- `invoice_dollars`
- `Freight`
- `total_item_quantity`
- `days_po_to_invoice`
- `total_item_dollars`

### Preprocessing

- `StandardScaler` fit on train split and saved as:
  - `models/scaler.pkl`

### Model Trained

- RandomForestClassifier with `GridSearchCV`

### Train Command

Run from repo root:

```bash
python invoice_flagging/train.py
```

### Output Artifacts

- `models/predict_flag_invoice.pkl`
- `models/scaler.pkl`

## Inference Scripts

### Freight Inference

```bash
python inference/predict_freight.py
```

This script loads `models/predict_freight_model.pkl` and predicts `Predicted_Freight`.

### Invoice Flag Inference

```bash
python inference/predict_invoice_flag.py
```

Note: the current script is a basic template and may require alignment with the classifier feature set and scaler pipeline for production usage.

## Streamlit Web App

Path: `app.py`

The app includes two tabs:

- `Freight Model`: input invoice dollars -> predicted freight
- `Invoice Flag Model`: input six engineered features -> predicted flag + confidence (if available)

### Run App

```bash
streamlit run app.py
```

Then open the URL printed in terminal (usually `http://localhost:8501`).

### Streamlit Cloud Deployment

- Ensure `requirements.txt` is present at repo root (already added).
- Ensure `runtime.txt` is present at repo root (already added, `python-3.11`).
- In Streamlit Cloud app settings:
  - Repository: this repo
  - Branch: `master`
  - Main file path: `app.py`

## End-to-End Workflow

1. Train freight model:
   - `python freight_cost_prediction/train.py`
2. Train invoice flag model:
   - `python invoice_flagging/train.py`
3. Launch UI:
   - `streamlit run app.py`
4. Test predictions via web forms.

## Common Issues

- `FileNotFoundError` for model/scaler:
  - Train model pipelines first so files are generated in `models/`.
- Import errors when running scripts:
  - Run commands from repository root.
- Streamlit command not found:
  - Install streamlit in active environment: `pip install streamlit`.

## Future Improvements

- Add `requirements.txt` and pinned versions.
- Unify module naming (`modelig_evolution.py` typo can be renamed safely).
- Add unit tests for preprocessing and inference contracts.
- Add one consistent inference API for both models (CLI + batch input).
- Add model/version metadata and experiment tracking.
