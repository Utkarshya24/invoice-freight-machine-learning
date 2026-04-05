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


def predict_freight_cost(model, dollars: float | np.ndarray) -> np.ndarray:
    input_df = pd.DataFrame({"Dollars": np.asarray(dollars).reshape(-1)})
    prediction = model.predict(input_df)
    return np.asarray(prediction).reshape(-1)


def predict_invoice_flag(model, scaler, payload: dict) -> Tuple[int, np.ndarray]:
    input_df = pd.DataFrame([payload])[INVOICE_FEATURES]
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])

    probabilities = np.array([np.nan, np.nan], dtype=float)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(scaled)[0], dtype=float)

    return pred, probabilities


def render_glow_metric(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.set_page_config(page_title="Invoice ML Dashboard", page_icon="📦", layout="wide")
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
            .stApp {
                background:
                    radial-gradient(circle at 15% 20%, rgba(255, 123, 84, 0.22), transparent 40%),
                    radial-gradient(circle at 85% 10%, rgba(0, 201, 255, 0.22), transparent 40%),
                    radial-gradient(circle at 50% 90%, rgba(124, 234, 181, 0.28), transparent 35%),
                    linear-gradient(135deg, #f7efe6 0%, #dce9f4 50%, #d9efe4 100%);
            }
            .block-container {
                padding-top: 1.4rem;
                max-width: 1180px;
            }
            .app-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.35rem;
                font-weight: 700;
                color: #0d2b3a;
                letter-spacing: 0.01em;
            }
            .app-subtitle {
                color: #164d67;
                margin-bottom: 1.2rem;
                font-family: 'Space Grotesk', sans-serif;
            }
            .card {
                background: rgba(255,255,255,0.66);
                border: 1px solid rgba(12,45,68,0.16);
                backdrop-filter: blur(8px);
                border-radius: 18px;
                box-shadow: 0 12px 30px rgba(15,38,52,0.12);
                padding: 1.1rem 1.1rem 0.5rem 1.1rem;
                margin-bottom: 0.9rem;
            }
            .metric-card {
                background: linear-gradient(145deg, rgba(15,58,84,0.92), rgba(7,30,44,0.92));
                border: 1px solid rgba(145, 215, 255, 0.3);
                border-radius: 14px;
                padding: 0.85rem 0.95rem;
                margin-bottom: 0.75rem;
                box-shadow: 0 0 28px rgba(45, 155, 255, 0.2);
            }
            .metric-title {
                font-family: 'Space Grotesk', sans-serif;
                color: rgba(216, 237, 255, 0.86);
                font-size: 0.86rem;
            }
            .metric-value {
                font-family: 'IBM Plex Mono', monospace;
                color: #f7ffff;
                font-size: 1.45rem;
                font-weight: 600;
                margin-top: 0.12rem;
            }
            .metric-sub {
                color: rgba(183, 219, 241, 0.95);
                font-size: 0.78rem;
                margin-top: 0.18rem;
            }
            .pill {
                display: inline-block;
                font-size: 0.78rem;
                color: #133548;
                background: rgba(255,255,255,0.58);
                border: 1px solid rgba(19,53,72,0.2);
                border-radius: 999px;
                padding: 0.22rem 0.58rem;
                margin-right: 0.38rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">Invoice Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Predict freight cost, detect risky invoices, and explore model behavior visually.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="pill">Freight Regression</span><span class="pill">Risk Classification</span><span class="pill">Interactive What-if Graphs</span>',
        unsafe_allow_html=True,
    )


def render_freight_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Freight Cost Prediction Engine")
    st.caption("Input: `Dollars` | Output: predicted `Freight`")

    left, right = st.columns([1.15, 1.2], gap="large")
    with left:
        dollars = st.number_input(
            "Invoice Dollars",
            min_value=0.0,
            value=18500.0,
            step=100.0,
        )

        sweep = st.slider(
            "What-if range around selected Dollars (%)",
            min_value=10,
            max_value=120,
            value=45,
            step=5,
        )
        predict_btn = st.button("Run Freight Simulation", use_container_width=True)

    if not FREIGHT_MODEL_PATH.exists():
        st.error(f"Missing model file: {FREIGHT_MODEL_PATH}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    model = load_freight_model(FREIGHT_MODEL_PATH)
    center_pred = float(predict_freight_cost(model, dollars)[0])
    freight_pct = (center_pred / dollars * 100.0) if dollars else 0.0

    with right:
        render_glow_metric("Predicted Freight", f"${center_pred:,.2f}", "single-invoice estimate")
        render_glow_metric("Freight Ratio", f"{freight_pct:.2f}%", "freight ÷ invoice dollars")
        render_glow_metric("Invoice Dollars", f"${dollars:,.2f}", "current selected invoice amount")

    if predict_btn or dollars >= 0:
        low = max(1.0, dollars * (1 - sweep / 100))
        high = max(low + 1.0, dollars * (1 + sweep / 100))
        x_values = np.linspace(low, high, 35)
        y_values = predict_freight_cost(model, x_values)

        curve_df = pd.DataFrame({"Dollars": x_values, "Predicted_Freight": y_values})
        st.line_chart(curve_df.set_index("Dollars"), color="#05668d", use_container_width=True)

        checkpoints = np.array([low, dollars, high], dtype=float)
        cp_pred = predict_freight_cost(model, checkpoints)
        comparison_df = pd.DataFrame(
            {
                "Scenario": ["Low", "Current", "High"],
                "Dollars": checkpoints,
                "Predicted Freight": cp_pred,
            }
        )
        st.dataframe(
            comparison_df.style.format({"Dollars": "${:,.2f}", "Predicted Freight": "${:,.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_invoice_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Invoice Risk Radar")
    st.caption("Classifier outputs flag status with confidence and factor breakdown.")

    c1, c2, c3 = st.columns(3, gap="large")
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

    if st.button("Analyze Invoice Risk", use_container_width=True):
        missing = [p for p in [INVOICE_MODEL_PATH, SCALER_PATH] if not p.exists()]
        if missing:
            st.error(f"Missing file(s): {', '.join(str(p) for p in missing)}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        model, scaler = load_invoice_model_and_scaler(INVOICE_MODEL_PATH, SCALER_PATH)
        pred, probabilities = predict_invoice_flag(model, scaler, payload)

        status = "Flagged for Review" if pred == 1 else "Normal Invoice"
        status_color = "#d7263d" if pred == 1 else "#1b8a5a"
        confidence = float(np.nanmax(probabilities)) if not np.isnan(probabilities).all() else np.nan

        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.markdown(
                f"""
                <div style="border-radius:14px;padding:1rem;background:rgba(255,255,255,0.72);border:1px solid rgba(30,60,80,0.2);">
                    <div style="font-size:0.82rem;color:#2a4f63;">Final Decision</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{status_color};margin-top:0.2rem;">{status}</div>
                    <div style="font-size:0.85rem;color:#355c72;margin-top:0.4rem;">Confidence: {'' if np.isnan(confidence) else f'{confidence * 100:.2f}%'} </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            mismatch = abs(invoice_dollars - total_item_dollars)
            render_glow_metric("Dollar Mismatch", f"${mismatch:,.2f}", "abs(invoice_dollars - total_item_dollars)")
            render_glow_metric("Invoice Lag", f"{days_po_to_invoice:.1f} days", "PO date to invoice date")

        with right:
            if not np.isnan(probabilities).all():
                prob_df = pd.DataFrame(
                    {
                        "Class": ["Normal (0)", "Flagged (1)"],
                        "Probability": probabilities,
                    }
                )
                st.bar_chart(prob_df.set_index("Class"), color="#f26419", use_container_width=True)

            if hasattr(model, "feature_importances_"):
                scaled_input = scaler.transform(pd.DataFrame([payload])[INVOICE_FEATURES])[0]
                impact_score = np.abs(scaled_input) * np.asarray(model.feature_importances_)
                impact_df = pd.DataFrame(
                    {"Feature": INVOICE_FEATURES, "Impact": impact_score}
                ).sort_values("Impact", ascending=False)
                st.caption("Approx. factor influence for this invoice")
                st.bar_chart(impact_df.set_index("Feature"), color="#2a9d8f", use_container_width=True)

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
