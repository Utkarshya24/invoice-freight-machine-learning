from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


FREIGHT_MODEL_PATH = Path("models/predict_freight_model.pkl")
INVOICE_MODEL_PATH = Path("models/predict_flag_invoice.pkl")
SCALER_PATH = Path("models/scaler.pkl")

INVOICE_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "days_po_to_invoice",
    "total_item_dollars",
]


@st.cache_resource
def load_freight_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_resource
def load_invoice_model_and_scaler(
    model_path: Path,
    scaler_path: Path,
) -> Tuple[object, object]:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_freight_cost(model, dollars: float) -> float:
    input_df = pd.DataFrame({"Dollars": [dollars]})
    prediction = model.predict(input_df)
    return float(np.asarray(prediction).reshape(-1)[0])


def predict_invoice_flag(model, scaler, payload: dict) -> Tuple[int, float]:
    input_df = pd.DataFrame([payload])[INVOICE_FEATURES]
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])

    confidence = np.nan
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        confidence = float(np.max(proba))

    return pred, confidence


def render_header():
    st.set_page_config(page_title="Invoice ML Dashboard", page_icon="📦", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(120deg, #f7f4ef 0%, #e4eef5 50%, #d7e6e0 100%);
            }
            .block-container {
                padding-top: 2rem;
            }
            .app-title {
                font-size: 2rem;
                font-weight: 700;
                color: #132a3a;
                letter-spacing: 0.01em;
            }
            .app-subtitle {
                color: #2c4f63;
                margin-bottom: 1rem;
            }
            .card {
                background: rgba(255,255,255,0.76);
                border: 1px solid rgba(19,42,58,0.12);
                border-radius: 14px;
                padding: 1rem 1rem 0.25rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">Invoice Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Freight cost regression + invoice risk classification</div>',
        unsafe_allow_html=True,
    )


def render_freight_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Freight Cost Prediction")
    st.caption("Model input: `Dollars`")

    col1, col2 = st.columns([2, 1])
    with col1:
        dollars = st.number_input(
            "Invoice Dollars",
            min_value=0.0,
            value=18500.0,
            step=100.0,
        )
    with col2:
        st.write("")
        st.write("")
        predict_btn = st.button("Predict Freight Cost", use_container_width=True)

    if predict_btn:
        if not FREIGHT_MODEL_PATH.exists():
            st.error(f"Missing model file: {FREIGHT_MODEL_PATH}")
        else:
            model = load_freight_model(FREIGHT_MODEL_PATH)
            pred = predict_freight_cost(model, dollars)
            st.metric("Predicted Freight", f"${pred:,.2f}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_invoice_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Invoice Flagging")
    st.caption("Model predicts if invoice should be flagged for review.")

    c1, c2, c3 = st.columns(3)
    with c1:
        invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0, value=120.0, step=1.0)
        invoice_dollars = st.number_input("Invoice Dollars", min_value=0.0, value=9000.0, step=100.0)
    with c2:
        freight = st.number_input("Freight", min_value=0.0, value=350.0, step=10.0)
        total_item_quantity = st.number_input("Total Item Quantity", min_value=0.0, value=130.0, step=1.0)
    with c3:
        days_po_to_invoice = st.number_input("Days PO to Invoice", min_value=0.0, value=7.0, step=1.0)
        total_item_dollars = st.number_input("Total Item Dollars", min_value=0.0, value=9200.0, step=100.0)

    payload = {
        "invoice_quantity": invoice_quantity,
        "invoice_dollars": invoice_dollars,
        "Freight": freight,
        "total_item_quantity": total_item_quantity,
        "days_po_to_invoice": days_po_to_invoice,
        "total_item_dollars": total_item_dollars,
    }

    if st.button("Predict Invoice Flag", use_container_width=True):
        missing = [p for p in [INVOICE_MODEL_PATH, SCALER_PATH] if not p.exists()]
        if missing:
            st.error(f"Missing file(s): {', '.join(str(p) for p in missing)}")
        else:
            model, scaler = load_invoice_model_and_scaler(INVOICE_MODEL_PATH, SCALER_PATH)
            pred, confidence = predict_invoice_flag(model, scaler, payload)
            label = "Flagged for Review" if pred == 1 else "Normal Invoice"

            st.metric("Prediction", label)
            if not np.isnan(confidence):
                st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    render_header()
    tab1, tab2 = st.tabs(["Freight Model", "Invoice Flag Model"])
    with tab1:
        render_freight_tab()
    with tab2:
        render_invoice_tab()


if __name__ == "__main__":
    main()
