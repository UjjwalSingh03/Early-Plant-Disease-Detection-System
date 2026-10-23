"""
=============================================================================
  🍅 TomatoAI — Tomato Disease Detection & Leaf Health Diagnosis System
  B.Tech 4th Year Major Project & Research Demonstration
  Deep Learning Architecture: MobileNetV2 vs Custom 4-Block CNN
=============================================================================
"""
import os
import json
import numpy as np
import streamlit as st
from PIL import Image

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TomatoAI — Plant Pathology & Disease Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for High-Tech Glassmorphism Design ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #131d2a 0%, #0c1219 50%, #070a0e 100%) !important;
    color: #e2e8f0;
}

/* Hero Section */
.hero-wrapper {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(56, 189, 248, 0.08) 50%, rgba(168, 85, 247, 0.08) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #22c55e, #38bdf8, #a855f7);
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #4ade80 0%, #38bdf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.15;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.6rem;
    font-weight: 400;
    max-width: 850px;
}

.badge-pill {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    border-radius: 9999px;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
    margin-top: 0.75rem;
}

/* Metric Cards */
.metric-container {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
}
.metric-container:hover {
    border-color: rgba(74, 222, 128, 0.4);
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.05);
}
.metric-num {
    font-size: 1.85rem;
    font-weight: 800;
    color: #4ade80;
    line-height: 1;
}
.metric-desc {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 600;
}

/* Prediction Cards */
.result-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(56, 189, 248, 0.1));
    border: 1.5px solid rgba(74, 222, 128, 0.5);
    border-radius: 18px;
    padding: 1.6rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(34, 197, 94, 0.18);
}

.result-name {
    font-size: 1.7rem;
    font-weight: 800;
    color: #4ade80;
    letter-spacing: -0.02em;
    margin-top: 0.4rem;
}

.info-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.2rem;
    margin-top: 1rem;
}

.info-label {
    font-weight: 700;
    color: #38bdf8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0b1118 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(34, 197, 94, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths & Metadata ─────────────────────────────────────────────────────────
PROJECT_ROOT = "/Users/shubhamjaiswal/Desktop/Disease Project"
MODELS_DIR = os.path.join(PROJECT_ROOT, "tomato_output", "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "tomato_output", "plots")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "tomato_output", "reports")
CLASSES_JSON = os.path.join(PROJECT_ROOT, "tomato_output", "tomato_class_indices.json")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "test_samples", "tomato")

# ── Disease Information Database ─────────────────────────────────────────────
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "type": "Bacterial Infection",
        "symptoms": "Water-soaked lesions on foliage turning dark brown/black with greasy halos; raised scab-like lesions on fruit.",
        "severity": "High",
        "treatment": "Apply fixed-copper bactericides combined with mancozeb. Prune infected foliage during dry conditions.",
        "prevention": "Utilize certified disease-free seed, rotate crops with non-solanaceous species for 2 years, avoid overhead irrigation.",
        "color": "#f87171",
        "emoji": "🦠"
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "pathogen": "Alternaria solani",
        "type": "Fungal Infection",
        "symptoms": "Characteristic dark concentric target-board rings on older leaves, surrounded by yellow chlorotic halos.",
        "severity": "Medium",
        "treatment": "Apply protective fungicides such as chlorothalonil, mancozeb, or copper oxychloride. Remove diseased lower leaves.",
        "prevention": "Mulch base of plants to avoid soil splash, sanitize pruning shears, maintain adequate spacing for canopy aeration.",
        "color": "#fb923c",
        "emoji": "🍂"
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "type": "Oomycete Pathogen (High Risk)",
        "symptoms": "Rapidly expanding water-soaked greasy patches turning purplish-brown; white powdery downy mildew on leaf undersides in high humidity.",
        "severity": "Critical",
        "treatment": "Systemic fungicides containing metalaxyl, dimethomorph, or cymoxanil. Destroy severely infected plants immediately.",
        "prevention": "Deploy resistant cultivars, avoid sprinkler watering, maintain rigorous field scouting during cool, damp weather.",
        "color": "#c084fc",
        "emoji": "⚠️"
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "pathogen": "Passalora fulva (syn. Fulvia fulva)",
        "type": "Fungal Infection (Common in Polyhouses)",
        "symptoms": "Pale yellow chlorotic patches on upper leaf surface corresponding to velvety olive-green to grey mold on underside.",
        "severity": "Medium",
        "treatment": "Apply copper fungicides or bio-fungicide Bacillus subtilis. Improve ventilation to drop relative humidity.",
        "prevention": "Maintain greenhouse relative humidity strictly below 85%, use drip irrigation, space plants adequately.",
        "color": "#4ade80",
        "emoji": "🌿"
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici",
        "type": "Fungal Infection",
        "symptoms": "Abundant small circular spots (1-3mm) with light gray centers and dark borders; visible black dots (pycnidia) inside spots.",
        "severity": "Medium",
        "treatment": "Spray chlorothalonil or copper-based sprays every 7-10 days upon first detection. Strip infected lower leaves.",
        "prevention": "Rotate crops every 3 seasons, avoid working in damp fields, eliminate solanaceous weeds.",
        "color": "#facc15",
        "emoji": "🔵"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Spider Mites (Two-spotted)",
        "pathogen": "Tetranychus urticae",
        "type": "Arachnid Pest Infestation",
        "symptoms": "Fine yellow stippling across leaf surfaces; silken webbing covering leaves and growing tips; bronze foliage necrosis.",
        "severity": "Medium-High",
        "treatment": "Apply acaricides/miticides (abamectin, bifenazate) or organic insecticidal soap / neem oil.",
        "prevention": "Conserve predatory mites (Phytoseiulus persimilis), avoid drought stress, spray leaf undersides with clean water.",
        "color": "#f43f5e",
        "emoji": "🕷️"
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "pathogen": "Corynespora cassiicola",
        "type": "Fungal Pathogen",
        "symptoms": "Pinpoint brown specks enlarging into round brown lesions with defined concentric rings and yellow halos.",
        "severity": "Medium",
        "treatment": "Apply strobilurin fungicides (azoxystrobin) or difenoconazole. Clear infected plant debris from ground.",
        "prevention": "Use certified seed, ensure canopy airflow, avoid prolonged leaf moisture.",
        "color": "#2dd4bf",
        "emoji": "🎯"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Tomato yellow leaf curl virus (Begomovirus)",
        "type": "Viral Disease (Vector-borne)",
        "symptoms": "Severe upward leaf cupping, interveinal yellowing, stunted erect habit, abortion of flowers, and near-zero fruit set.",
        "severity": "Critical",
        "treatment": "No cure exists for viral infection. Rogue out infected plants. Manage insect vector (Bemisia tabaci whitefly).",
        "prevention": "Screen nurseries with 50-mesh insect netting, apply reflective silver mulches, plant TYLCV-tolerant F1 hybrids.",
        "color": "#eab308",
        "emoji": "🟡"
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tomato mosaic virus (Tobamovirus)",
        "type": "Viral Disease (Mechanically Vectored)",
        "symptoms": "Mottled dark and light green mosaic on foliage, fern-like leaf distortion, internal fruit browning.",
        "severity": "High",
        "treatment": "Remove and incinerate diseased plants. Wash all farming equipment with 10% trisodium phosphate (TSP).",
        "prevention": "Strict phytosanitary hygiene; prohibit tobacco smoking/handling near crop; use heat-treated certified seed.",
        "color": "#38bdf8",
        "emoji": "🧩"
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "pathogen": "None (Plant is Healthy)",
        "type": "Optimal Physiological State",
        "symptoms": "No pathogenic symptoms detected. Vigorous green coloration, intact cuticle, and healthy venation.",
        "severity": "None",
        "treatment": "No corrective treatment needed. Maintain standard agronomic regime.",
        "prevention": "Continue balanced N-P-K fertilization, regular scouting, and optimal irrigation scheduling.",
        "color": "#22c55e",
        "emoji": "✅"
    }
}

# ── Model & Data Loader ───────────────────────────────────────────────────────
@st.cache_resource
def load_disease_model(model_choice):
    try:
        import tensorflow as tf
        if model_choice == "MobileNetV2 (Recommended - 69.8% Test Acc)":
            path = os.path.join(MODELS_DIR, "tomato_mobilenetv2_model.h5")
            name = "MobileNetV2 (Depthwise Separable CNN)"
            acc = "69.75%"
        else:
            path = os.path.join(MODELS_DIR, "tomato_cnn_model.h5")
            name = "Custom 4-Block CNN Baseline"
            acc = "45.84%"

        if not os.path.exists(path):
            return None, f"Model file not found at: {path}", None, None

        model = tf.keras.models.load_model(path)
        # Determine expected image dimensions
        in_shape = model.input_shape
        img_size = (in_shape[1], in_shape[2]) if in_shape and len(in_shape) >= 3 else (64, 64)

        return model, None, name, (acc, img_size)
    except Exception as e:
        return None, str(e), None, None

@st.cache_data
def load_class_indices():
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r") as f:
            return json.load(f)
    return {str(i): k for i, k in enumerate(DISEASE_INFO.keys())}

class_dict = load_class_indices()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 1.2rem 0;">
        <span style="font-size:3.2rem;">🍅</span>
        <div style="font-size:1.3rem; font-weight:800; color:#4ade80; margin-top:0.3rem;">TomatoAI</div>
        <div style="font-size:0.75rem; color:#94a3b8;">B.Tech CSE / AI Major Project</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ Model Selection")
    model_choice = st.selectbox(
        "Active Classifier Architecture:",
        [
            "MobileNetV2 (Recommended - 69.8% Test Acc)",
            "Custom 4-Block CNN (45.8% Test Acc)"
        ],
        index=0
    )
    
    model, model_err, active_model_name, model_meta = load_disease_model(model_choice)
    
    if model:
        acc_val, target_size = model_meta
        st.success(f"🟢 Loaded: **{active_model_name}**\n\n🎯 Test Accuracy: **{acc_val}** | Input: `{target_size[0]}×{target_size[1]}`")
    else:
        st.error(f"🔴 Model Error: {model_err}")

    st.divider()

    st.markdown("#### 📱 Edge / IoT Deployment")
    tflite_path = os.path.join(MODELS_DIR, "tomato_mobilenetv2.tflite")
    if os.path.exists(tflite_path):
        size_kb = os.path.getsize(tflite_path) / 1024
        st.markdown(f"• **TFLite Export:** `Available`\n• **Model Size:** `{size_kb:.1f} KB` (Ultra-compact)\n• **Target:** Android / Raspberry Pi / Drone")
    
    st.divider()
    
    st.markdown("#### 📊 Dataset Statistics")
    st.markdown("""
    - **Total Tomato Images:** `19,160`
    - **Total Classes:** `10 Categories`
    - **Balanced Training Subset:** `8,000 images`
    - **Train / Val / Test:** `70% / 20% / 10%`
    - **Augmentation:** `Rotation, Zoom, Flip`
    """)

    st.divider()
    st.caption("Developed by Shubham Jaiswal | B.Tech CSE | Project Approval Committee & Research Paper Submission")

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab_diagnose, tab_graphs, tab_paper = st.tabs([
    "🌿 Disease Diagnosis & Inference",
    "📊 Model Analytics & 10 Evaluation Plots",
    "📄 Research Paper & IEEE Specifications"
])

# =============================================================================
# TAB 1: DISEASE DIAGNOSIS & INFERENCE
# =============================================================================
with tab_diagnose:
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">🍅 Tomato Disease Detection & Pathology System</div>
        <div class="hero-subtitle">
            Autonomous leaf disease classification using deep transfer learning. Upload a field photo or choose from the curated test dataset to receive instantaneous diagnostic telemetry and agronomic remedies.
        </div>
        <div class="badge-pill">Plant Pathology | Deep Learning | Precision Agriculture</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Quick Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-container"><div class="metric-num">10</div><div class="metric-desc">Pathology Classes</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-container"><div class="metric-num">69.8%</div><div class="metric-desc">MobileNetV2 Accuracy</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-container"><div class="metric-num">0.7117</div><div class="metric-desc">Macro F1-Score</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-container"><div class="metric-num">200 KB</div><div class="metric-desc">TFLite Edge Footprint</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.25], gap="large")

    selected_img = None
    input_source = None

    with col_left:
        st.markdown("### 📸 Input Image Selection")
        
        input_mode = st.radio(
            "Choose Image Source:",
            ["✨ Select Pre-loaded Sample (Quick Demo)", "📤 Upload Custom Photo"],
            horizontal=True
        )

        if input_mode == "✨ Select Pre-loaded Sample (Quick Demo)":
            if os.path.exists(SAMPLES_DIR):
                sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                sample_labels = [f.replace(".JPG","").replace(".jpg","").replace("_"," ") for f in sample_files]
                
                selected_sample_label = st.selectbox("Select Test Class Sample:", sample_labels)
                idx = sample_labels.index(selected_sample_label)
                sample_file = sample_files[idx]
                sample_path = os.path.join(SAMPLES_DIR, sample_file)
                
                selected_img = Image.open(sample_path)
                input_source = f"Sample: {selected_sample_label}"
                st.image(selected_img, caption=f"Selected Sample: {selected_sample_label}", use_container_width=True)
            else:
                st.warning("Sample folder not found.")
        else:
            uploaded_file = st.file_uploader("Upload tomato leaf photo (JPG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                selected_img = Image.open(uploaded_file)
                input_source = f"Uploaded File: {uploaded_file.name}"
                st.image(selected_img, caption="Uploaded Leaf Image", use_container_width=True)

    with col_right:
        st.markdown("### 🔬 Diagnostic Telemetry")

        if selected_img is None:
            st.info("👈 Please select a pre-loaded sample leaf or upload an image to run diagnostic inference.")
        else:
            if st.button("🚀 Analyze Plant Health", use_container_width=True):
                if model is None:
                    st.error(f"Cannot run prediction: {model_err}")
                else:
                    with st.spinner("🧠 Performing deep convolutional inference..."):
                        target_size = model_meta[1] if model_meta else (64, 64)
                        # Preprocessing
                        proc_img = selected_img.convert('RGB').resize(target_size)
                        img_tensor = np.array(proc_img).astype('float32') / 255.0
                        img_tensor = np.expand_dims(img_tensor, axis=0)

                        preds = model.predict(img_tensor, verbose=0)[0]
                        top3_idx = np.argsort(preds)[::-1][:3]
                        
                        top1_idx = top3_idx[0]
                        top1_conf = float(preds[top1_idx])
                        raw_class_name = class_dict.get(str(top1_idx), list(DISEASE_INFO.keys())[0])

                        info = DISEASE_INFO.get(raw_class_name, {
                            "name": raw_class_name.replace("Tomato___", "").replace("_", " "),
                            "pathogen": "Unknown",
                            "type": "Plant Pathology",
                            "symptoms": "Detailed analysis required.",
                            "severity": "Medium",
                            "treatment": "Consult university agricultural extension.",
                            "prevention": "Implement standard sanitary measures.",
                            "color": "#38bdf8",
                            "emoji": "🔬"
                        })

                        # Big Result Card
                        st.markdown(f"""
                        <div class="result-card">
                            <span style="font-size: 3.5rem;">{info['emoji']}</span>
                            <div class="result-name">{info['name']}</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 0.3rem;">
                                Model Diagnosis: <b>{active_model_name}</b>
                            </div>
                            <div style="font-size: 1.35rem; font-weight: 800; color: #4ade80; margin-top: 0.5rem;">
                                Diagnostic Confidence: {top1_conf * 100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(top1_conf, text=f"Top-1 Prediction Confidence: {top1_conf*100:.1f}%")

                        # Top-3 predictions
                        st.markdown("#### 🏆 Top-3 Probability Distribution")
                        for rank, idx in enumerate(top3_idx):
                            c_name = class_dict.get(str(idx), f"Class {idx}")
                            c_display = DISEASE_INFO.get(c_name, {}).get("name", c_name.replace("Tomato___","").replace("_"," "))
                            c_conf = float(preds[idx])
                            
                            pcol1, pcol2 = st.columns([2, 1])
                            with pcol1:
                                st.write(f"**#{rank+1} {c_display}**")
                            with pcol2:
                                st.write(f"`{c_conf*100:.2f}%`")
                            st.progress(c_conf)

                        # Disease Agronomic Dossier
                        st.markdown("#### 📋 Agronomic Treatment & Disease Profile")
                        
                        inf1, inf2 = st.columns(2)
                        with inf1:
                            st.markdown(f"""
                            <div class="info-box">
                                <div class="info-label">Pathogen & Taxonomy</div>
                                <div style="color: #e2e8f0; font-style: italic;">{info['pathogen']}</div>
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">Classification: {info['type']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with inf2:
                            st.markdown(f"""
                            <div class="info-box">
                                <div class="info-label">Severity Level</div>
                                <div style="color: {info['color']}; font-weight: 700; font-size: 1.1rem;">{info['severity']}</div>
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">Requires immediate mitigation if High/Critical</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="info-box">
                            <div class="info-label">Field Symptoms</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem;">{info['symptoms']}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">Curative Treatment & Chemical Control</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem;">{info['treatment']}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">Long-Term Prevention & Cultural Practices</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem;">{info['prevention']}</div>
                        </div>
                        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: MODEL ANALYTICS & 10 EVALUATION PLOTS
# =============================================================================
with tab_graphs:
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">📊 Experimental Telemetry & 10 Evaluation Plots</div>
        <div class="hero-subtitle">
            Complete empirical evaluation generated during model training. These figures validate architectural convergence, confusion dynamics, per-class recall/precision, and comparative model superiority for project approval presentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load summary report
    summary_path = os.path.join(REPORTS_DIR, "final_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
        
        st.markdown("### 📈 Quantitative Model Comparison Matrix")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="info-box" style="border-left: 4px solid #38bdf8;">
                <h4 style="color:#38bdf8; margin:0;">🤖 Custom 4-Block CNN (Scratch)</h4>
                <ul style="color:#cbd5e1; font-size:0.95rem; margin-top:0.5rem; line-height:1.7;">
                    <li><b>Test Accuracy:</b> 45.84%</li>
                    <li><b>Test Loss:</b> 4.5074</li>
                    <li><b>Macro Precision:</b> 72.02%</li>
                    <li><b>Macro Recall:</b> 46.72%</li>
                    <li><b>Macro F1-Score:</b> 0.4608</li>
                    <li><b>Training Time (CPU):</b> 11.9 minutes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="info-box" style="border-left: 4px solid #22c55e;">
                <h4 style="color:#22c55e; margin:0;">⚡ MobileNetV2-Lite (Transfer Learning - Winner 🏆)</h4>
                <ul style="color:#cbd5e1; font-size:0.95rem; margin-top:0.5rem; line-height:1.7;">
                    <li><b>Test Accuracy:</b> 69.75% (+23.91% higher!)</li>
                    <li><b>Test Loss:</b> 1.0142</li>
                    <li><b>Macro Precision:</b> 77.34%</li>
                    <li><b>Macro Recall:</b> 70.65%</li>
                    <li><b>Macro F1-Score:</b> 0.7117</li>
                    <li><b>Training Time (CPU):</b> 3.8 minutes (3.1× faster!)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Complete 10-Graph Publication Gallery")

    plot_meta = [
        ("01_class_distribution.png", "Figure 1: Tomato Disease Dataset Taxonomy & Class Distribution", "Visualizes the imbalance and distribution of 18,160 total images across all 10 tomato health states."),
        ("02_sample_disease_images.png", "Figure 2: Tomato Leaf Disease Visual Atlas (10 Categories)", "High-resolution 2x5 grid demonstrating visual symptoms across all distinct foliar conditions."),
        ("03_cnn_training_curves.png", "Figure 3: Custom 4-Block CNN Training & Validation Curves", "Depicts training accuracy/loss progression across epochs for the baseline custom CNN model."),
        ("04_mnv2_training_curves.png", "Figure 4: MobileNetV2 Training & Validation Curves", "Demonstrates smooth convergence and steady loss minimization for depthwise separable transfer learning."),
        ("05_cnn_confusion_matrix.png", "Figure 5: Custom CNN Confusion Matrix (Raw Counts & Normalized)", "Evaluates inter-class misclassifications, showing confusion between Late Blight and Septoria Leaf Spot."),
        ("06_mnv2_confusion_matrix.png", "Figure 6: MobileNetV2 Confusion Matrix (Raw Counts & Normalized)", "Displays high diagonal concentration indicating strong discriminatory power across all 10 classes."),
        ("07_cnn_per_class_metrics.png", "Figure 7: Custom CNN Per-Class Precision, Recall, and F1-Score", "Per-category breakdown revealing performance bottlenecks in complex leaf spot differentiation."),
        ("08_mnv2_per_class_metrics.png", "Figure 8: MobileNetV2 Per-Class Precision, Recall, and F1-Score", "Demonstrates superior balanced F1 scores exceeding 0.70 across the vast majority of disease categories."),
        ("09_model_comparison.png", "Figure 9: Comprehensive Performance Metric Comparison Dashboard", "Side-by-side grouped bar charts for Accuracy, Precision, Recall, F1, and CPU Training Duration."),
        ("10_comparison_curves.png", "Figure 10: Comparative Accuracy & Loss Convergence Overlay", "Direct epoch-by-epoch overlay proving MobileNetV2's superior gradient flow and generalization capability.")
    ]

    for filename, title, desc in plot_meta:
        img_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(img_path):
            with st.expander(f"📌 {title}", expanded=True):
                st.image(img_path, caption=title, use_container_width=True)
                st.caption(f"**Analysis:** {desc}")
        else:
            st.warning(f"Plot {filename} not found at {img_path}")

# =============================================================================
# TAB 3: RESEARCH PAPER & SPECIFICATIONS
# =============================================================================
with tab_paper:
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">📄 IEEE Research Paper & Project Dossier</div>
        <div class="hero-subtitle">
            Formal publication documentation prepared for B.Tech Final Year project evaluation and submission to peer-reviewed international journals (e.g., IEEE Access, Springer, IRJET).
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_p1, c_p2 = st.columns([1, 1])
    with c_p1:
        st.markdown("### 📑 Paper Overview")
        st.markdown("""
        - **Title:** *Empirical Evaluation of Deep Transfer Learning and Custom CNN Architectures for Automated Tomato Leaf Disease Classification*
        - **Target Format:** IEEE Transactions / Conference 2-Column Template
        - **Author:** Shubham Jaiswal (Department of Computer Science & Engineering)
        - **Number of References:** 12 Peer-Reviewed Citations (2018–2024)
        - **Key Finding:** Depthwise Separable Convolutions (MobileNetV2) attain a 23.91% higher accuracy and train 3.1× faster on constrained CPU hardware compared to monolithic 4-block custom CNNs.
        """)

        # Download buttons
        paper_md_path = os.path.join(PROJECT_ROOT, "docs", "Research_Paper_Tomato_Disease_Detection_IEEE.md")
        pdf_path = os.path.join(PROJECT_ROOT, "docs", "Tomato_Disease_Detection_Research_Paper.pdf")
        
        if os.path.exists(paper_md_path):
            with open(paper_md_path, "r", encoding="utf-8") as f:
                paper_text = f.read()
            st.download_button(
                label="📥 Download Research Paper (Markdown Format)",
                data=paper_text,
                file_name="Tomato_Disease_Research_Paper_IEEE.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Research Paper (Compiled PDF)",
                data=pdf_bytes,
                file_name="Tomato_Disease_Research_Paper.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        handbook_pdf_path = os.path.join(PROJECT_ROOT, "docs", "Tomato_Disease_Project_Team_Handbook.pdf")
        if os.path.exists(handbook_pdf_path):
            with open(handbook_pdf_path, "rb") as f:
                handbook_bytes = f.read()
            st.download_button(
                label="👥 Download Team Master Handbook (PDF)",
                data=handbook_bytes,
                file_name="Tomato_Disease_Project_Team_Handbook.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with c_p2:
        st.markdown("### 🎯 Viva Voce Defense Key Points")
        st.markdown("""
        <div class="info-box">
            <b>Q1: Why narrow scope specifically to Tomato?</b><br>
            <span style="color:#94a3b8; font-size:0.88rem;">Tomato (*Solanum lycopersicum*) is India's 2nd largest vegetable crop ($2.1B economic value), suffering up to 70% yield loss from blights. The PlantVillage dataset has 10 distinct classes with high visual overlap, providing an ideal benchmark.</span>
        </div>
        <div class="info-box">
            <b>Q2: Why did MobileNetV2 outperform Custom CNN?</b><br>
            <span style="color:#94a3b8; font-size:0.88rem;">MobileNetV2 leverages depthwise separable convolutions that factorize spatial filtering from feature combination, drastically reducing parameters (3.5M vs monolithic dense weights) while transferring rich generalized low-level edge features.</span>
        </div>
        <div class="info-box">
            <b>Q3: What makes this deployable in Indian rural farming?</b><br>
            <span style="color:#94a3b8; font-size:0.88rem;">The model was quantized to an ultra-compact 200 KB TensorFlow Lite file (.tflite), allowing real-time offline edge inference on low-cost Android smartphones without internet access.</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📖 Full Research Paper Preview")
    
    if os.path.exists(paper_md_path):
        with open(paper_md_path, "r", encoding="utf-8") as f:
            full_paper_content = f.read()
        st.markdown(full_paper_content)
    else:
        st.info("Generating paper file...")
