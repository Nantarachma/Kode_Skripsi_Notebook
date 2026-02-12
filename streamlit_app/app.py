"""
NIDS Real-Time Dashboard
========================
Streamlit application for simulating real-time Network Intrusion Detection
using Pareto-optimal XGBoost models produced by the training notebook.

Pipeline:
  1. Pemuatan Artefak   – Load XGBoost model, StandardScaler, & session state.
  2. Akuisisi & Pra-proses – Stream one row, map features, Z-score normalize.
  3. Inferensi & Timing  – predict_proba wrapped by perf_counter_ns.
  4. Dekoding Keputusan   – Map integer class to human-readable label.
  5. Visualisasi & UI     – Update indicator, gauge, and log table.

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

MAX_LOG_ROWS = 50  # rows retained in the detection history
LATENCY_THRESHOLD_MS = 70  # latency threshold for flagging rows (ms)

# Event type display configuration for log table
_EVENT_ICONS: Dict[str, str] = {"tp": "🚨", "tn": "✅", "fp": "⚠️", "fn": "🟣"}
_EVENT_ROW_CLASSES: Dict[str, str] = {"tp": "row-attack", "fp": "row-fp", "fn": "row-fn"}


# ---------------------------------------------------------------------------
# 1. Pemuatan Artefak Model (Artifact Loading)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_label_encoder():
    """Load the LabelEncoder used to encode the target variable."""
    if LABEL_ENCODER_PATH.exists():
        return joblib.load(LABEL_ENCODER_PATH)
    return None


@st.cache_resource
def load_scaler():
    """Load the StandardScaler (mean & std from training data) for Z-score."""
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

# Derive expected feature order from scaler when available
if scaler is not None and hasattr(scaler, "feature_names_in_"):
    EXPECTED_FEATURES: List[str] = list(scaler.feature_names_in_)
else:
    EXPECTED_FEATURES = []

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NIDS Real-Time Dashboard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
        .metric-card {
            padding: 18px 15px; border-radius: 12px; text-align: center;
            margin-bottom: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            transition: box-shadow 0.2s ease;
        }
        .metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .metric-card h2 { margin: 0; }
        .normal {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border-left: 5px solid #28a745;
        }
        .attack {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border-left: 5px solid #dc3545;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.75; }
        }
        .scenario-box {
            padding: 10px; background-color: #e2e3e5;
            border-radius: 5px; font-size: 0.9em;
        }
        .alert-card {
            padding: 18px; border-radius: 12px; text-align: center;
            margin-bottom: 10px;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .alert-card .detail {
            font-size: 0.85em; margin-top: 5px;
        }
        .alert-card-attack {
            border: 2px solid #dc3545;
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            box-shadow: 0 2px 8px rgba(220,53,69,0.18);
        }
        .alert-card-fp {
            border: 2px solid #e67e22;
            background: linear-gradient(135deg, #fef3e2 0%, #fdebd0 100%);
            color: #7d4e00;
            box-shadow: 0 2px 8px rgba(230,126,34,0.18);
        }
        .alert-card-fn {
            border: 2px solid #8e44ad;
            background: linear-gradient(135deg, #f4ecf7 0%, #e8daef 100%);
            color: #4a235a;
            box-shadow: 0 2px 8px rgba(142,68,173,0.18);
        }
        .alert-card-stopped {
            border: 2px solid #dc3545;
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            box-shadow: 0 3px 12px rgba(220,53,69,0.25);
        }
        .summary-box {
            padding: 12px 16px; border-radius: 10px;
            background: linear-gradient(135deg, #eaf2f8 0%, #d6eaf8 100%);
            border: 1px solid #aed6f1; color: #1b4f72;
            font-size: 0.9em; margin-bottom: 10px;
        }
        .summary-box b { color: #154360; }

        /* History log table */
        .log-table {
            width: 100%; border-collapse: separate;
            border-spacing: 0; border-radius: 10px;
            overflow: hidden; font-size: 0.92em;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .log-table thead tr {
            background: linear-gradient(90deg, #4a6fa5 0%, #3b5998 100%);
            color: #ffffff;
        }
        .log-table th {
            padding: 10px 12px; text-align: left;
            font-weight: 600; letter-spacing: 0.02em;
            border: none;
        }
        .log-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #e8e8e8;
        }
        .log-table tbody tr {
            background-color: #ffffff;
            transition: background-color 0.15s ease;
        }
        .log-table tbody tr:nth-child(even) {
            background-color: #f7f9fc;
        }
        .log-table tbody tr:hover {
            background-color: #edf2f7;
        }
        .log-table tbody tr.row-bad {
            background-color: #fff0f0 !important;
        }
        .log-table tbody tr.row-bad td {
            color: #8b1a1a;
            border-bottom-color: #f5c6cb;
        }
        .log-table tbody tr.row-bad:hover {
            background-color: #ffe0e0 !important;
        }
        .log-table tbody tr.row-fp {
            background-color: #fef9e7 !important;
        }
        .log-table tbody tr.row-fp td {
            color: #7d4e00;
            border-bottom-color: #f9e79f;
        }
        .log-table tbody tr.row-fp:hover {
            background-color: #fcf3cf !important;
        }
        .log-table tbody tr.row-fn {
            background-color: #f5eef8 !important;
        }
        .log-table tbody tr.row-fn td {
            color: #4a235a;
            border-bottom-color: #d7bde2;
        }
        .log-table tbody tr.row-fn:hover {
            background-color: #ebdef0 !important;
        }
        .log-table tbody tr.row-attack {
            background-color: #fdedec !important;
        }
        .log-table tbody tr.row-attack td {
            color: #78281f;
            border-bottom-color: #f5b7b1;
        }
        .log-table tbody tr.row-attack:hover {
            background-color: #fadbd8 !important;
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

    # ------------------------------------------------------------------
    # 3. Inferensi & Pengukuran Waktu (Critical Timing & Inference)
    # ------------------------------------------------------------------
    def predict(self, data: np.ndarray) -> Tuple[int, str, float, float]:
        """Run single-sample inference and measure latency.

        The timer strictly wraps only the ``predict_proba`` call so that
        UI rendering time is never included in the latency measurement.

        Returns:
            class_index: Predicted class (0–3).
            label:       Human-readable class name from LABEL_MAP.
            confidence:  Probability of the predicted class.
            latency_ms:  Wall-clock inference time in milliseconds.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Select a model first.")

        # Start Timer (nanosecond precision)
        t0 = time.perf_counter_ns()
        # Model prediction – probability for each class (0, 1, 2, 3)
        probs = self.model.predict_proba(data)[0]
        # Stop Timer
        t1 = time.perf_counter_ns()

        # 4. Dekoding Keputusan (Decision Decoding)
        idx = int(np.argmax(probs))
        label = LABEL_MAP.get(idx, "Unknown")
        confidence = float(probs[idx])
        latency_ms = (t1 - t0) / 1e6

        return idx, label, confidence, latency_ms


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
if "fn_count" not in st.session_state:
    st.session_state.fn_count = 0
if "tp_count" not in st.session_state:
    st.session_state.tp_count = 0
if "tn_count" not in st.session_state:
    st.session_state.tn_count = 0
if "latency_history" not in st.session_state:
    st.session_state.latency_history = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None  # stores (pred_idx, pred_label, conf, lat)
if "last_mismatch" not in st.session_state:
    st.session_state.last_mismatch = None  # stores (true_label, pred_label) or None
if "last_event_type" not in st.session_state:
    st.session_state.last_event_type = None  # "tp", "tn", "fp", "fn", or None
if "stop_reason" not in st.session_state:
    st.session_state.stop_reason = None  # reason the simulation was auto-stopped
if "step_once" not in st.session_state:
    st.session_state.step_once = False  # single-step mode flag


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

speed = st.sidebar.slider("Kecepatan Simulasi (detik)", 1, 10, 2)

auto_stop = st.sidebar.checkbox(
    "🛑 Auto-stop saat serangan terdeteksi",
    value=True,
    help="Simulasi otomatis berhenti jika model mendeteksi serangan, "
    "sehingga Anda bisa menganalisis paket sebelum melanjutkan.",
)

# 3. Start / Stop / Reset controls ------------------------------------------
col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶️ START"):
    st.session_state.run = True
    st.session_state.stop_reason = None
if col_stop.button("⏹️ STOP"):
    st.session_state.run = False
if st.sidebar.button("⏭️ NEXT (1 paket)"):
    st.session_state.run = True
    st.session_state.step_once = True
    st.session_state.stop_reason = None
if st.sidebar.button("🔄 RESET"):
    st.session_state.run = False
    st.session_state.logs = []
    st.session_state.idx = 0
    st.session_state.fp_count = 0
    st.session_state.fn_count = 0
    st.session_state.tp_count = 0
    st.session_state.tn_count = 0
    st.session_state.latency_history = []
    st.session_state.last_pred = None
    st.session_state.last_mismatch = None
    st.session_state.last_event_type = None
    st.session_state.stop_reason = None
    st.session_state.step_once = False


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
st.title("🛡️ NIDS Real-Time Dashboard")


# ---------------------------------------------------------------------------
# Data loading & preprocessing helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_simulation_data() -> pd.DataFrame:
    """Load the pre-sampled simulation data (generated by the training notebook).

    The training notebook exports ``simulation_data.csv`` with features that
    are **already Z-score normalised** using the same ``StandardScaler``
    saved in ``scaler.pkl``.  Therefore, no additional scaling is applied
    here to avoid double-transformation.
    """
    if not SIMULATION_PATH.exists():
        st.error(
            f"Simulation data not found at `{SIMULATION_PATH}`. "
            "Run the training notebook first to generate it."
        )
        st.stop()
    return pd.read_csv(SIMULATION_PATH)


def prepare_features(row_df: pd.DataFrame) -> np.ndarray:
    """Pra-pemrosesan: feature mapping & ordering for a single row.

    1. **Pemetaan Fitur** – Reorder columns to match the feature order
       expected by the scaler / model (``EXPECTED_FEATURES``).
    2. The simulation data is already Z-score scaled by the training
       notebook, so no further transformation is needed.

    Returns a 2-D numpy array of shape ``(1, n_features)``.
    """
    features = row_df.drop(columns=["Label_True"], errors="ignore")

    # Pemetaan Fitur: ensure column order matches model expectations
    if EXPECTED_FEATURES:
        missing = [c for c in EXPECTED_FEATURES if c not in features.columns]
        if missing:
            for col in missing:
                features[col] = 0.0
        features = features[EXPECTED_FEATURES]

    return features.values


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
_should_run = (
    st.session_state.run
    and st.session_state.idx < len(stream)
    and catalog_loaded
)
if _should_run:
    # -- 2. Akuisisi & Pra-pemrosesan ----------------------------------------
    row_data = stream.iloc[[st.session_state.idx]]
    true_label_idx = int(row_data["Label_True"].iloc[0])
    true_label = LABEL_MAP.get(true_label_idx, "Unknown")

    # Feature mapping & preparation
    input_data = prepare_features(row_data)

    # -- 3. Inferensi & Pengukuran Waktu -------------------------------------
    pred_idx, pred_label, conf, lat = st.session_state.engine.predict(input_data)

    # Record latency history
    st.session_state.latency_history.append(lat)

    # -- Classify event type (TP / TN / FP / FN) ----------------------------
    is_true_attack = true_label_idx > 0
    is_pred_attack = pred_idx > 0

    if is_pred_attack and is_true_attack:
        event_type = "tp"
        st.session_state.tp_count += 1
    elif not is_pred_attack and not is_true_attack:
        event_type = "tn"
        st.session_state.tn_count += 1
    elif is_pred_attack and not is_true_attack:
        event_type = "fp"
        st.session_state.fp_count += 1
    elif not is_pred_attack and is_true_attack:
        event_type = "fn"
        st.session_state.fn_count += 1

    st.session_state.last_event_type = event_type

    # Store last prediction for persistent display
    st.session_state.last_pred = (pred_idx, pred_label, conf, lat)

    # Track mismatch for enhanced visualization
    is_mismatch = pred_idx != true_label_idx
    if is_mismatch:
        st.session_state.last_mismatch = (true_label, pred_label)
    else:
        st.session_state.last_mismatch = None

    # -- 5. Visualisasi & Pembaruan UI ---------------------------------------
    # Append to rolling log (most-recent first)
    st.session_state.logs.insert(
        0,
        {
            "ID": st.session_state.idx,
            "Waktu": time.strftime("%H:%M:%S"),
            "Label Asli": true_label,
            "Prediksi": pred_label,
            "Confidence": f"{conf:.1%}",
            "Latensi": f"{lat:.3f} ms",
            "_lat_val": lat,
            "_mismatch": is_mismatch,
            "_event": event_type,
        },
    )
    if len(st.session_state.logs) > MAX_LOG_ROWS:
        st.session_state.logs.pop()

    st.session_state.idx += 1

    # -- Auto-stop on attack detection (if enabled) --------------------------
    if auto_stop and is_pred_attack:
        st.session_state.run = False
        st.session_state.stop_reason = (
            f"🚨 Serangan **{pred_label}** terdeteksi pada paket #{st.session_state.idx - 1} "
            f"(confidence {conf:.1%}). "
            "Simulasi dijeda otomatis. Tekan **▶️ START** untuk melanjutkan "
            "atau **⏭️ NEXT** untuk maju satu paket."
        )

    # If single-step mode, stop after processing one packet
    if st.session_state.step_once:
        st.session_state.run = False
        st.session_state.step_once = False

elif st.session_state.idx >= len(stream) and st.session_state.run:
    st.session_state.run = False
    st.success("✅ Simulasi Selesai.")

# ---------------------------------------------------------------------------
# Dashboard display (always visible)
# ---------------------------------------------------------------------------
if st.session_state.last_pred is not None:
    pred_idx, pred_label, conf, lat = st.session_state.last_pred

    # --- Status card ---
    status_cls = "normal" if pred_idx == 0 else "attack"
    icon = "✅" if pred_idx == 0 else "🚨"
    col_stat1.markdown(
        f'<div class="metric-card {status_cls}"><h2>{icon} {pred_label}</h2></div>',
        unsafe_allow_html=True,
    )

    # --- Latency metric ---
    avg_lat = np.mean(st.session_state.latency_history)
    col_stat2.metric(
        "Inference Latency",
        f"{lat:.4f} ms",
        delta=f"avg {avg_lat:.4f} ms",
        delta_color="off",
    )

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

    # --- Detection summary (TP / TN / FP / FN) ---
    st.markdown(
        '<div class="summary-box">'
        f"📊 <b>Ringkasan Deteksi</b> &nbsp;|&nbsp; "
        f"✅ TP: <b>{st.session_state.tp_count}</b> &nbsp;|&nbsp; "
        f"✅ TN: <b>{st.session_state.tn_count}</b> &nbsp;|&nbsp; "
        f"⚠️ FP: <b>{st.session_state.fp_count}</b> &nbsp;|&nbsp; "
        f"🟣 FN: <b>{st.session_state.fn_count}</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    # --- Latency gauge ---
    bar_color = "#2ecc71" if lat < LATENCY_THRESHOLD_MS else "#e74c3c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=lat,
            title={"text": "Latency Monitor (ms)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=80, b=10, l=30, r=30))
    chart_spot.plotly_chart(fig, use_container_width=True)

    # --- Alert cards for attack / FP / FN ------------------------------------
    last_event = st.session_state.last_event_type
    if last_event == "tp":
        # True Positive: correctly detected attack
        st.markdown(
            f'<div class="alert-card alert-card-attack">'
            f"<h3>🚨 Serangan Terdeteksi: {pred_label}</h3>"
            f'<div class="detail">'
            f"Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp; "
            f"Latensi: <b>{lat:.3f} ms</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif last_event == "fp":
        # False Positive: normal traffic flagged as attack
        m_true, m_pred = st.session_state.last_mismatch
        st.markdown(
            f'<div class="alert-card alert-card-fp">'
            f"<h3>⚠️ False Positive Terdeteksi</h3>"
            f'<div class="detail">'
            f"Trafik <b>{m_true}</b> salah diklasifikasi sebagai "
            f"<b>{m_pred}</b></div>"
            f'<div class="detail">Model memberikan alarm palsu pada paket normal.</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    elif last_event == "fn":
        # False Negative: attack missed by model
        m_true, m_pred = st.session_state.last_mismatch
        st.markdown(
            f'<div class="alert-card alert-card-fn">'
            f"<h3>🟣 False Negative Terdeteksi</h3>"
            f'<div class="detail">'
            f"Serangan <b>{m_true}</b> tidak terdeteksi — "
            f"model memprediksi <b>{m_pred}</b></div>"
            f'<div class="detail">Model gagal mengenali serangan ini.</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    # Default state before simulation starts
    col_stat1.markdown(
        '<div class="metric-card normal"><h2>⏸️ Menunggu Simulasi</h2></div>',
        unsafe_allow_html=True,
    )
    col_stat2.metric("Inference Latency", "— ms", delta="avg — ms", delta_color="off")
    if is_baseline:
        col_stat3.metric("False Positives", "0", delta="Harus 0", delta_color="inverse")
    else:
        col_stat3.metric("Packet Processed", "0", delta="Idle")

    # --- Empty latency gauge ---
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=0,
            title={"text": "Latency Monitor (ms)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71"},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=80, b=10, l=30, r=30))
    chart_spot.plotly_chart(fig, use_container_width=True)


def _style_log_table(log_data: list) -> str:
    """Build an HTML table with class-based styling for different event types."""
    if not log_data:
        return ""
    display_cols = ["ID", "Waktu", "Label Asli", "Prediksi", "Confidence", "Latensi"]
    header = "<th>Status</th>" + "".join(f"<th>{c}</th>" for c in display_cols)
    rows_html = ""
    for entry in log_data:
        event = entry.get("_event", None)
        icon = _EVENT_ICONS.get(event, "—")
        row_cls_name = _EVENT_ROW_CLASSES.get(event, "")
        # Also flag high-latency rows
        if entry.get("_lat_val", 0) > LATENCY_THRESHOLD_MS and not row_cls_name:
            row_cls_name = "row-bad"
        row_cls = f' class="{row_cls_name}"' if row_cls_name else ""
        cells = f"<td>{icon}</td>" + "".join(
            f"<td>{entry.get(c, '')}</td>" for c in display_cols
        )
        rows_html += f"<tr{row_cls}>{cells}</tr>"
    return (
        '<table class="log-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


# --- Log table (always visible) ---
if st.session_state.logs:
    log_spot.markdown(_style_log_table(st.session_state.logs), unsafe_allow_html=True)
else:
    log_spot.info("📋 Log deteksi akan muncul di sini saat simulasi berjalan.")

# --- Status message ---
if st.session_state.stop_reason:
    st.warning(st.session_state.stop_reason)
elif not st.session_state.run and st.session_state.last_pred is not None:
    st.info("⏹️ Simulasi Dihentikan. Tekan ▶️ START untuk melanjutkan atau ⏭️ NEXT untuk maju satu paket.")

# --- Trigger rerun for next step ---
if st.session_state.run and st.session_state.idx < len(stream) and catalog_loaded:
    time.sleep(speed)
    st.rerun()
