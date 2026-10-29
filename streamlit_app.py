"""
streamlit_app.py — Traffic Sign Recognition
Streamlit Cloud entry point.

Prediction pipeline mirrors the training notebook exactly:
  PIL → grayscale → 32×32 → /255 → (1,32,32,1) → model → argmax → signs[pred+1]
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import time
import sys
import logging
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "my_model.h5"
sys.path.insert(0, str(BASE_DIR))

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Signs dictionary (inline from signs.py) ───────────────────────────────
from signs import classes as SIGN_CLASSES   # keys 1–43

# ── Sign categories & meanings ────────────────────────────────────────────
SIGN_INFO = {
    1:  ("Regulatory – Speed Limit", "Maximum speed 20 km/h"),
    2:  ("Regulatory – Speed Limit", "Maximum speed 30 km/h"),
    3:  ("Regulatory – Speed Limit", "Maximum speed 50 km/h"),
    4:  ("Regulatory – Speed Limit", "Maximum speed 60 km/h"),
    5:  ("Regulatory – Speed Limit", "Maximum speed 70 km/h"),
    6:  ("Regulatory – Speed Limit", "Maximum speed 80 km/h"),
    7:  ("Regulatory – Speed Limit", "End of 80 km/h speed limit"),
    8:  ("Regulatory – Speed Limit", "Maximum speed 100 km/h"),
    9:  ("Regulatory – Speed Limit", "Maximum speed 120 km/h"),
    10: ("Regulatory – Prohibition",  "No overtaking for all vehicles"),
    11: ("Regulatory – Prohibition",  "No overtaking for vehicles over 3.5 t"),
    12: ("Regulatory – Priority",     "Right-of-way at the next intersection"),
    13: ("Regulatory – Priority",     "Priority road — you have priority"),
    14: ("Regulatory – Priority",     "Yield to oncoming traffic"),
    15: ("Regulatory – Mandatory",    "Stop and give way"),
    16: ("Regulatory – Prohibition",  "No vehicles allowed"),
    17: ("Regulatory – Prohibition",  "Prohibited for vehicles over 3.5 t"),
    18: ("Regulatory – Prohibition",  "No entry for all vehicles"),
    19: ("Warning",                   "General caution"),
    20: ("Warning",                   "Dangerous curve to the left"),
    21: ("Warning",                   "Dangerous curve to the right"),
    22: ("Warning",                   "Double curve ahead"),
    23: ("Warning",                   "Bumpy road ahead"),
    24: ("Warning",                   "Slippery road — reduce speed"),
    25: ("Warning",                   "Road narrows on the right"),
    26: ("Warning",                   "Road work ahead"),
    27: ("Warning",                   "Traffic signals ahead"),
    28: ("Warning",                   "Pedestrians ahead"),
    29: ("Warning",                   "Children crossing"),
    30: ("Warning",                   "Bicycles crossing"),
    31: ("Warning",                   "Beware of ice or snow"),
    32: ("Warning",                   "Wild animals may cross"),
    33: ("Regulatory – End",          "End of all speed and passing limits"),
    34: ("Regulatory – Direction",    "Turn right ahead"),
    35: ("Regulatory – Direction",    "Turn left ahead"),
    36: ("Regulatory – Direction",    "Go straight ahead"),
    37: ("Regulatory – Direction",    "Go straight or turn right"),
    38: ("Regulatory – Direction",    "Go straight or turn left"),
    39: ("Regulatory – Direction",    "Keep right"),
    40: ("Regulatory – Direction",    "Keep left"),
    41: ("Regulatory – Direction",    "Roundabout — follow direction"),
    42: ("Regulatory – End",          "End of no-overtaking zone"),
    43: ("Regulatory – End",          "End of no-overtaking zone for heavy vehicles"),
}

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B1120;
    color: #F8FAFC;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #2D3748;
}
[data-testid="stSidebar"] * { color: #F8FAFC !important; }

/* ---- cards ---- */
.card {
    background: #1E293B;
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent { border-left: 4px solid #38BDF8; }
.card-success { border-left: 4px solid #22C55E; }
.card-warning { border-left: 4px solid #F59E0B; }
.card-error   { border-left: 4px solid #EF4444; }

/* ---- metric labels ---- */
.metric-value { font-size: 2rem; font-weight: 700; color: #38BDF8; }
.metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 2px; }

/* ---- sign name ---- */
.sign-name {
    font-size: 1.6rem; font-weight: 700;
    color: #38BDF8; margin: 8px 0 4px 0;
}
.sign-category { font-size: 0.9rem; color: #94A3B8; margin-bottom: 8px; }

/* ---- confidence bar container ---- */
.conf-bar-bg {
    background: #2D3748; border-radius: 6px;
    height: 10px; width: 100%; margin: 4px 0 12px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 6px; transition: width 0.3s;
}

/* ---- top-3 row ---- */
.top3-label { font-size: 0.88rem; color: #F8FAFC; }
.top3-pct   { font-size: 0.88rem; color: #94A3B8; text-align: right; }

/* ---- disclaimer ---- */
.disclaimer {
    background: #1a2030; border: 1px solid #2D3748;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.8rem; color: #94A3B8; margin-top: 16px;
}

/* ---- hide streamlit branding ---- */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Model loader  (cached — runs once per session)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CNN model…")
def load_model():
    """Load Keras model once and cache it for the session."""
    if not MODEL_PATH.exists():
        st.error(f"❌ Model file not found: `{MODEL_PATH}`")
        st.stop()
    from keras.models import load_model as keras_load
    model = keras_load(str(MODEL_PATH))
    logger.info("Model loaded. Input: %s  Output: %s",
                model.input_shape, model.output_shape)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Prediction pipeline
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Grayscale → 32×32 → /255 → (1,32,32,1)."""
    img = pil_image.convert("L")
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (32,32,1)
    arr = np.expand_dims(arr, axis=0)    # (1,32,32,1)
    return arr


def predict(model, pil_image: Image.Image) -> dict:
    """Run full prediction and return result dict."""
    t0 = time.perf_counter()
    arr = preprocess(pil_image)
    probs = model.predict(arr, verbose=0)[0]          # shape (43,)
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    label = SIGN_CLASSES.get(pred_class + 1, f"Unknown ({pred_class})")
    sorted_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "class_id":    int(i),
            "label":       SIGN_CLASSES.get(int(i) + 1, f"Unknown ({i})"),
            "probability": float(probs[i]),
        }
        for i in sorted_idx
    ]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "predicted_class":    pred_class,
        "predicted_label":    label,
        "confidence":         confidence,
        "top3":               top3,
        "processing_time_ms": elapsed_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def confidence_color(conf: float) -> str:
    if conf >= 0.70:
        return "#22C55E"   # green
    if conf >= 0.40:
        return "#F59E0B"   # amber
    return "#EF4444"       # red


def conf_bar_html(probability: float, color: str) -> str:
    pct = round(probability * 100, 1)
    return f"""
<div class="conf-bar-bg">
  <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚦 Traffic Sign\nRecognition")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📷 Predict Sign", "📊 Dataset Info",
         "🤖 Model Info", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#94A3B8'>"
        "CNN trained on GTSRB<br>43 traffic sign classes<br>Python 3.11 · TF 2.15"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Load model (always, so status is available on all pages) ──────────────
model = load_model()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🚦 Traffic Sign Recognition")
    st.markdown(
        "<p style='color:#94A3B8;font-size:1.05rem;margin-top:-8px'>"
        "AI-powered traffic sign classification using a Convolutional Neural Network</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="card card-accent">
          <div class="metric-value">43</div>
          <div class="metric-label">Traffic Sign Classes</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card card-accent">
          <div class="metric-value">34,776</div>
          <div class="metric-label">Training Images</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        model_status = "✅ Loaded" if model else "❌ Error"
        color = "#22C55E" if model else "#EF4444"
        st.markdown(f"""<div class="card card-success">
          <div class="metric-value" style="color:{color};font-size:1.4rem">{model_status}</div>
          <div class="metric-label">Model Status</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="card card-accent">
          <div class="metric-value">3.28 MB</div>
          <div class="metric-label">Model Size</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### How It Works")
        st.markdown("""
<div class="card">
<ol style="color:#F8FAFC;line-height:2">
  <li>Upload a traffic sign image (JPG / PNG / BMP)</li>
  <li>The CNN preprocesses it to 32×32 grayscale</li>
  <li>The model outputs probabilities for all 43 classes</li>
  <li>The top prediction and confidence score are shown</li>
</ol>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("### Quick Start")
        st.markdown("""
<div class="card">
  <p style="color:#94A3B8">Select <strong style="color:#38BDF8">📷 Predict Sign</strong>
  in the sidebar, upload any traffic sign photo, and click
  <strong style="color:#38BDF8">Classify</strong>.</p>
  <p style="color:#94A3B8;margin-top:12px">Supported formats: JPG · JPEG · PNG · BMP</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Sample Traffic Signs")
    sample_cols = st.columns(5)
    samples = [
        ("Stop", "🛑"), ("Yield", "🔻"), ("Speed Limit", "🔵"),
        ("No Entry", "⛔"), ("Caution", "⚠️"),
    ]
    for col, (name, icon) in zip(sample_cols, samples):
        with col:
            st.markdown(f"""
<div class="card" style="text-align:center;padding:16px 8px">
  <div style="font-size:2rem">{icon}</div>
  <div style="font-size:0.8rem;color:#94A3B8;margin-top:6px">{name}</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Predict Sign
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📷 Predict Sign":
    st.markdown("# 📷 Predict Traffic Sign")
    st.markdown(
        "<p style='color:#94A3B8'>Upload a traffic sign image to classify it.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Upload Image")
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported: JPG, JPEG, PNG, BMP",
        )

        if uploaded:
            # ── File info ────────────────────────────────────────────────
            file_size_kb = round(len(uploaded.getvalue()) / 1024, 1)
            try:
                pil_img = Image.open(BytesIO(uploaded.getvalue()))
                w, h = pil_img.size
                st.markdown(f"""
<div class="card card-accent">
  <strong style="color:#38BDF8">{uploaded.name}</strong><br>
  <span style="color:#94A3B8;font-size:0.85rem">
    {w} × {h} px &nbsp;·&nbsp; {file_size_kb} KB &nbsp;·&nbsp; {uploaded.type}
  </span>
</div>""", unsafe_allow_html=True)
                st.image(pil_img, caption="Preview", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Cannot open image: {e}")
                uploaded = None

    with col_right:
        st.markdown("### Result")
        if not uploaded:
            st.markdown("""
<div class="card" style="text-align:center;padding:40px 24px">
  <div style="font-size:3rem">🖼️</div>
  <div style="color:#94A3B8;margin-top:12px">Upload an image to see the prediction</div>
</div>""", unsafe_allow_html=True)
        else:
            classify_btn = st.button("🔍 Classify Image", type="primary",
                                     use_container_width=True)
            if classify_btn:
                with st.spinner("Running CNN inference…"):
                    try:
                        result = predict(model, pil_img)
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        logger.exception("Prediction error")
                        st.stop()

                pred_label = result["predicted_label"]
                pred_class = result["predicted_class"]
                conf       = result["confidence"]
                conf_pct   = round(conf * 100, 1)
                elapsed    = round(result["processing_time_ms"], 1)
                color      = confidence_color(conf)

                # ── Result card ──────────────────────────────────────────
                st.markdown(f"""
<div class="card card-success">
  <div class="sign-name">{pred_label}</div>
  <div class="sign-category">Class ID: {pred_class}</div>
  <div style="display:flex;align-items:center;gap:12px;margin-top:8px">
    <span style="font-size:1.8rem;font-weight:700;color:{color}">{conf_pct}%</span>
    <span style="color:#94A3B8;font-size:0.85rem">Confidence</span>
  </div>
  {conf_bar_html(conf, color)}
  <div style="font-size:0.8rem;color:#94A3B8">
    Processed in {elapsed} ms
  </div>
</div>""", unsafe_allow_html=True)

                # Low-confidence warning
                if conf < 0.60:
                    st.warning(
                        "⚠️ **Low confidence** — the model is uncertain. "
                        "Try a clearer, well-lit photo of the sign."
                    )

                # ── Sign information ─────────────────────────────────────
                sign_key = pred_class + 1
                if sign_key in SIGN_INFO:
                    category, meaning = SIGN_INFO[sign_key]
                    st.markdown(f"""
<div class="card card-accent">
  <strong>Sign Information</strong><br>
  <div style="margin-top:10px">
    <span style="color:#94A3B8">Category:</span>
    <span style="color:#F8FAFC;margin-left:8px">{category}</span>
  </div>
  <div style="margin-top:6px">
    <span style="color:#94A3B8">Meaning:</span>
    <span style="color:#F8FAFC;margin-left:8px">{meaning}</span>
  </div>
</div>""", unsafe_allow_html=True)

                # ── Top-3 predictions ────────────────────────────────────
                st.markdown("#### Top 3 Predictions")
                for rank, item in enumerate(result["top3"], 1):
                    pct   = round(item["probability"] * 100, 1)
                    col_  = confidence_color(item["probability"])
                    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:6px 0;border-bottom:1px solid #2D3748">
  <span class="top3-label">
    <strong style="color:{col_}">{rank}.</strong> {item['label']}
    <span style="color:#94A3B8;font-size:0.8rem"> (Class {item['class_id']})</span>
  </span>
  <span class="top3-pct" style="min-width:52px">{pct}%</span>
</div>
{conf_bar_html(item['probability'], col_)}
""", unsafe_allow_html=True)

                # ── Disclaimer ───────────────────────────────────────────
                st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This model is trained on the German Traffic Sign
  Recognition Benchmark (GTSRB). Signs may differ from those used in India or
  other countries. Do not use this tool as a substitute for observing actual
  road signs. Confidence scores reflect model probability, not real-world accuracy.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Info
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Info":
    st.markdown("# 📊 Dataset Information")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="card card-accent">
          <div class="metric-value">34,776</div>
          <div class="metric-label">Total Images</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card card-accent">
          <div class="metric-value">43</div>
          <div class="metric-label">Sign Classes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="card card-accent">
          <div class="metric-value">32×32</div>
          <div class="metric-label">Input Size</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Details")
    st.markdown("""
<div class="card">
  <table style="width:100%;border-collapse:collapse;color:#F8FAFC">
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8;width:200px">Dataset Name</td>
      <td>GTSRB — German Traffic Sign Recognition Benchmark</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Image Format</td>
      <td>PNG (32×32 pixels)</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Preprocessing</td>
      <td>Grayscale conversion + normalization (/255)</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Train / Test Split</td>
      <td>70% training / 30% testing</td>
    </tr>
    <tr>
      <td style="padding:8px 0;color:#94A3B8">Augmentation</td>
      <td>Width/height shift, zoom, shear, rotation</td>
    </tr>
  </table>
</div>""", unsafe_allow_html=True)

    # ── Class distribution chart ──────────────────────────────────────────
    st.markdown("### Class Distribution")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.config import DATASET_CLASS_COUNTS

        class_ids   = list(DATASET_CLASS_COUNTS.keys())
        counts      = list(DATASET_CLASS_COUNTS.values())
        labels_abbr = [f"{i}" for i in class_ids]

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("#1E293B")
        ax.set_facecolor("#1E293B")
        bars = ax.bar(class_ids, counts, color="#38BDF8", alpha=0.85, width=0.7)
        ax.set_xlabel("Class ID", color="#94A3B8", fontsize=10)
        ax.set_ylabel("Number of Images", color="#94A3B8", fontsize=10)
        ax.set_title("Images per Class (GTSRB)", color="#F8FAFC", fontsize=12, pad=10)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D3748")
        ax.grid(axis="y", color="#2D3748", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Chart unavailable: {e}")

    # ── Class browser ─────────────────────────────────────────────────────
    st.markdown("### All 43 Classes")
    import pandas as pd
    rows = []
    for class_id, label in SIGN_CLASSES.items():
        category, meaning = SIGN_INFO.get(class_id, ("—", "—"))
        count = DATASET_CLASS_COUNTS.get(class_id - 1, 0)
        rows.append({
            "Class ID (model)": class_id - 1,
            "Label":            label,
            "Category":         category,
            "Meaning":          meaning,
            "Training Images":  count,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Model Info
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Info":
    st.markdown("# 🤖 Model Information")
    st.markdown("---")

    info_col, _ = st.columns([2, 1])
    with info_col:
        st.markdown("""
<div class="card card-accent">
  <table style="width:100%;border-collapse:collapse;color:#F8FAFC">
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8;width:180px">Framework</td>
      <td>TensorFlow 2.15.1 / Keras 2.x</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Architecture</td>
      <td>Sequential CNN</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Input Shape</td>
      <td>(None, 32, 32, 1) — grayscale 32×32</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Output Shape</td>
      <td>(None, 43) — softmax over 43 classes</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:8px 0;color:#94A3B8">Model File</td>
      <td>my_model.h5 (3.28 MB)</td>
    </tr>
    <tr>
      <td style="padding:8px 0;color:#94A3B8">Status</td>
      <td><span style="color:#22C55E">✅ Loaded</span></td>
    </tr>
  </table>
</div>""", unsafe_allow_html=True)

    st.markdown("### CNN Architecture")
    st.markdown("""
<div class="card">
<table style="width:100%;border-collapse:collapse;color:#F8FAFC;font-size:0.9rem">
  <thead>
    <tr style="border-bottom:2px solid #2D3748">
      <th style="padding:8px 0;color:#38BDF8;text-align:left">Layer Type</th>
      <th style="padding:8px 0;color:#38BDF8;text-align:left">Details</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">Conv2D</td><td style="padding:7px 0">32 filters, 3×3, ReLU</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">Conv2D</td><td style="padding:7px 0">32 filters, 3×3, ReLU</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">MaxPooling2D</td><td style="padding:7px 0">2×2 pool size</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">Dropout</td><td style="padding:7px 0">0.25</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">Flatten</td><td style="padding:7px 0">—</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">Dense</td><td style="padding:7px 0">256 units, ReLU</td>
    </tr>
    <tr style="border-bottom:1px solid #2D3748">
      <td style="padding:7px 0">Dropout</td><td style="padding:7px 0">0.50</td>
    </tr>
    <tr>
      <td style="padding:7px 0">Dense (output)</td><td style="padding:7px 0">43 units, Softmax</td>
    </tr>
  </tbody>
</table>
</div>""", unsafe_allow_html=True)

    st.info("ℹ️ Training metrics (accuracy/loss curves) are not stored in the .h5 file. "
            "See `traffic_model.ipynb` for training details.")

    # ── Live layer info ───────────────────────────────────────────────────
    try:
        st.markdown("### Live Layer Summary")
        import pandas as pd
        layer_rows = []
        for layer in model.layers:
            layer_rows.append({
                "Layer":    layer.name,
                "Type":     layer.__class__.__name__,
                "Output Shape": str(layer.output_shape),
                "Params":   layer.count_params(),
            })
        st.dataframe(pd.DataFrame(layer_rows), use_container_width=True, hide_index=True)
        st.caption(f"Total parameters: {model.count_params():,}")
    except Exception as e:
        st.warning(f"Could not build layer table: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
<div class="card card-accent">
  <h3 style="color:#38BDF8;margin:0 0 8px 0">Traffic Sign Recognition Using CNN</h3>
  <p style="color:#94A3B8;line-height:1.7">
    A deep learning application that classifies traffic signs from images using a
    Convolutional Neural Network trained on the German Traffic Sign Recognition Benchmark (GTSRB).
    The model achieves real-time classification of 43 different traffic sign categories.
  </p>
</div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Technology Stack")
        techs = [
            ("🐍", "Python 3.11"),
            ("🧠", "TensorFlow 2.15 / Keras"),
            ("📸", "OpenCV + Pillow"),
            ("🔢", "NumPy"),
            ("🌐", "Streamlit"),
            ("📊", "Matplotlib / Pandas"),
        ]
        for icon, name in techs:
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;
            padding:8px 0;border-bottom:1px solid #2D3748">
  <span style="font-size:1.2rem">{icon}</span>
  <span style="color:#F8FAFC">{name}</span>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("### Project Details")
        st.markdown("""
<div class="card">
  <div style="margin-bottom:12px">
    <span style="color:#94A3B8">Dataset:</span>
    <span style="color:#F8FAFC;margin-left:8px">GTSRB (German Traffic Sign Recognition Benchmark)</span>
  </div>
  <div style="margin-bottom:12px">
    <span style="color:#94A3B8">Training Images:</span>
    <span style="color:#F8FAFC;margin-left:8px">34,776 images across 43 classes</span>
  </div>
  <div style="margin-bottom:12px">
    <span style="color:#94A3B8">Model Type:</span>
    <span style="color:#F8FAFC;margin-left:8px">Sequential CNN</span>
  </div>
  <div>
    <span style="color:#94A3B8">Input:</span>
    <span style="color:#F8FAFC;margin-left:8px">32×32 grayscale image</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("### ⚠️ Limitations")
    st.markdown("""
<div class="card card-warning">
  <ul style="color:#F8FAFC;line-height:2;margin:0;padding-left:20px">
    <li>Trained on <strong>German traffic signs</strong> — may not match signs in India or other countries.</li>
    <li>Confidence score reflects model probability, <strong>not real-world accuracy</strong>.</li>
    <li>Performance may degrade on blurry, dark, or partially obscured signs.</li>
    <li>This tool is for <strong>educational purposes only</strong> — do not use for driving decisions.</li>
    <li>The model was trained on 32×32 images; very high-res photos are downscaled before prediction.</li>
  </ul>
</div>""", unsafe_allow_html=True)
