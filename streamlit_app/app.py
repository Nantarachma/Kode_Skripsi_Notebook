"""
NIDS Real-Time Dashboard
========================
Streamlit application for simulating real-time Network Intrusion Detection
using Pareto-optimal XGBoost models produced by the training notebook.

Run with:
    cd streamlit_app && streamlit run app.py
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CATALOG_PATH = APP_DIR / "models_catalog.csv"
SIMULATION_PATH = APP_DIR / "simulation_data.csv"
LABEL_ENCODER_PATH = APP_DIR / "label_encoder.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"

LABEL_MAP_DEFAULT: Dict[int, str] = {0: "Normal", 1: "DoS", 2: "Probe", 3: "Malware"}


# ---------------------------------------------------------------------------
# Load preprocessing artifacts (label encoder & scaler)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_label_encoder():
    """Load the LabelEncoder used to encode the target variable."""
    if LABEL_ENCODER_PATH.exists():
        return joblib.load(LABEL_ENCODER_PATH)
    return None


@st.cache_resource
def load_scaler():
    """Load the StandardScaler used to normalize feature columns."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return None


label_encoder = load_label_encoder()
scaler = load_scaler()

# Build LABEL_MAP dynamically from the label encoder when available
if label_encoder is not None:
    LABEL_MAP: Dict[int, str] = {
        i: str(cls) for i, cls in enumerate(label_encoder.classes_)
    }
else:
    LABEL_MAP = LABEL_MAP_DEFAULT

MAX_LOG_ROWS = 10  # visible rows in the log table

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NIDS Real-Time Dashboard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
        .metric-card {
            padding: 15px; border-radius: 10px; text-align: center;
            border: 1px solid #ddd; margin-bottom: 10px;
        }
        .normal {
            background-color: #d4edda; color: #155724;
            border-left: 5px solid #28a745;
        }
        .attack {
            background-color: #f8d7da; color: #721c24;
            border-left: 5px solid #dc3545;
            animation: blink 1s infinite;
        }
        @keyframes blink { 50% { opacity: 0.8; } }
        .scenario-box {
            padding: 10px; background-color: #e2e3e5;
            border-radius: 5px; font-size: 0.9em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """Wraps an XGBoost classifier for single-sample prediction with timing."""

    def __init__(self) -> None:
        self.model: Optional[xgb.XGBClassifier] = None
        self.current_model_file: str = ""

    def load_model(self, filename: str) -> None:
        """Load an XGBoost JSON model from the *models/* directory."""
        if self.current_model_file == filename:
            return
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        self.current_model_file = filename

    def predict(self, data: np.ndarray) -> Tuple[int, str, float, float]:
        """Run single-sample inference and measure latency.

        Returns:
            class_index: Predicted class (0–3).
            label:       Human-readable class name from LABEL_MAP.
            confidence:  Probability of the predicted class.
            latency_ms:  Wall-clock inference time in milliseconds.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Select a model first.")
        t0 = time.perf_counter_ns()
        probs = self.model.predict_proba(data)[0]
        t1 = time.perf_counter_ns()
        idx = int(np.argmax(probs))
        return idx, LABEL_MAP.get(idx, "Unknown"), float(probs[idx]), (t1 - t0) / 1e6


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = InferenceEngine()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "run" not in st.session_state:
    st.session_state.run = False
if "fp_count" not in st.session_state:
    st.session_state.fp_count = 0


# ---------------------------------------------------------------------------
# Sidebar – Control Panel
# ---------------------------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")

# 1. Model selector ----------------------------------------------------------
catalog_loaded = False
try:
    catalog = pd.read_csv(CATALOG_PATH)
    methods = catalog["method"].unique()
    selected_method = st.sidebar.selectbox("1. Metode Optimasi", methods)

    filtered_catalog = catalog[catalog["method"] == selected_method]
    model_choice = st.sidebar.selectbox("2. Model Pareto", filtered_catalog["label"])

    row = filtered_catalog[filtered_catalog["label"] == model_choice].iloc[0]
    st.session_state.engine.load_model(row["filename"])
    st.sidebar.success(
        f"F1: {row['f1_score']:.3f} | Latency: {row['latency_us']:.1f} µs"
    )
    catalog_loaded = True
except FileNotFoundError:
    st.sidebar.error(
        f"Catalog not found at `{CATALOG_PATH}`. "
        "Run the training notebook first to generate it."
    )
except (KeyError, IndexError) as exc:
    st.sidebar.error(f"Error reading catalog: {exc}")

st.sidebar.markdown("---")

# Preprocessing artifacts status ------------------------------------------------
_le_status = "✅ Loaded" if label_encoder is not None else "⚠️ Not found"
_sc_status = "✅ Loaded" if scaler is not None else "⚠️ Not found"
st.sidebar.caption(f"Label Encoder: {_le_status}  \nScaler: {_sc_status}")

st.sidebar.markdown("---")

# 2. Scenario selector -------------------------------------------------------
scenario_mode = st.sidebar.radio(
    "3. Skenario Pengujian",
    ["Skenario 1: Baseline (Normal)", "Skenario 2: Injection (Serangan)"],
)

is_baseline = "Skenario 1" in scenario_mode

if is_baseline:
    st.sidebar.info(
        """
        **📌 Baseline Testing**
        - **Trafik:** 100% Normal (Benign)
        - **Fokus:** Latensi Dasar & Validitas False Positive.
        - **Harapan:** Indikator stabil HIJAU.
        """
    )
else:
    st.sidebar.warning(
        """
        **⚠️ Adversarial Injection**
        - **Trafik:** Campuran (Normal + Serangan)
        - **Fokus:** Responsivitas & Akurasi Log.
        - **Harapan:** Indikator berubah MERAH saat ada serangan.
        """
    )

speed = st.sidebar.slider("Kecepatan Simulasi (ms)", 10, 1000, 200)

# 3. Start / Stop / Reset controls ------------------------------------------
col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶️ START"):
    st.session_state.run = True
if col_stop.button("⏹️ STOP"):
    st.session_state.run = False
if st.sidebar.button("🔄 RESET"):
    st.session_state.logs = []
    st.session_state.idx = 0
    st.session_state.fp_count = 0


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
st.title("🛡️ NIDS Real-Time Dashboard")


@st.cache_data
def load_simulation_data() -> pd.DataFrame:
    """Load the pre-sampled simulation data (generated by the training notebook).

    If a fitted scaler (``scaler.pkl``) is available, the feature columns are
    transformed so that the model receives properly scaled inputs.
    """
    if not SIMULATION_PATH.exists():
        st.error(
            f"Simulation data not found at `{SIMULATION_PATH}`. "
            "Run the training notebook first to generate it."
        )
        st.stop()
    df = pd.read_csv(SIMULATION_PATH)

    # Apply scaler to feature columns when available
    if scaler is not None:
        feature_cols = [c for c in df.columns if c != "Label_True"]
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df


df = load_simulation_data()

# Filter stream based on scenario
if is_baseline:
    stream = df[df["Label_True"] == 0].reset_index(drop=True)
else:
    stream = df

col_stat1, col_stat2, col_stat3 = st.columns(3)
chart_spot = st.empty()
log_spot = st.empty()

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
if st.session_state.run and st.session_state.idx < len(stream) and catalog_loaded:
    row_data = stream.iloc[[st.session_state.idx]]
    input_data = row_data.drop(columns=["Label_True"], errors="ignore")

    pred_idx, pred_label, conf, lat = st.session_state.engine.predict(
        input_data.values
    )

    # Track false positives in baseline scenario
    if is_baseline and pred_idx > 0:
        st.session_state.fp_count += 1

    # Append to rolling log (most-recent first)
    st.session_state.logs.insert(
        0,
        {
            "ID": st.session_state.idx,
            "Waktu": time.strftime("%H:%M:%S"),
            "Prediksi": pred_label,
            "Confidence": f"{conf:.1%}",
            "Latensi": f"{lat:.3f} ms",
        },
    )
    if len(st.session_state.logs) > MAX_LOG_ROWS:
        st.session_state.logs.pop()

    # --- Status card ---
    status_cls = "normal" if pred_idx == 0 else "attack"
    icon = "✅" if pred_idx == 0 else "🚨"
    col_stat1.markdown(
        f'<div class="metric-card {status_cls}"><h2>{icon} {pred_label}</h2></div>',
        unsafe_allow_html=True,
    )

    # --- Latency metric ---
    col_stat2.metric("Inference Latency", f"{lat:.4f} ms")

    # --- Scenario-specific metric ---
    if is_baseline:
        col_stat3.metric(
            "False Positives",
            f"{st.session_state.fp_count}",
            delta="Harus 0",
            delta_color="inverse",
        )
    else:
        col_stat3.metric(
            "Packet Processed", f"{st.session_state.idx}", delta="Live"
        )

    # --- Latency gauge ---
    bar_color = "#2ecc71" if lat < 0.1 else "#e74c3c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=lat,
            title={"text": "Latency Monitor (ms)"},
            gauge={
                "axis": {"range": [0, 0.5]},
                "bar": {"color": bar_color},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.1,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30))
    chart_spot.plotly_chart(fig, use_container_width=True)

    # --- Log table ---
    log_spot.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True)

    st.session_state.idx += 1
    time.sleep(speed / 1000)
    st.rerun()

elif st.session_state.idx >= len(stream) and st.session_state.run:
    st.session_state.run = False
    st.success("✅ Simulasi Selesai.")
