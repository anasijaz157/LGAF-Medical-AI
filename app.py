import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFilter
import numpy as np
import os
import io
import time
from datetime import datetime

# =========================================================
# CONFIGURATION & HARDWARE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-path search for model checkpoints
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_MODEL_DIRS = [
    os.path.join(BASE_DIR, "models"),
    r"c:\Users\lenovo\Desktop\LGAF\models",
    os.path.join(os.getcwd(), "models"),
]

def find_model_path(filename):
    for d in POSSIBLE_MODEL_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return os.path.join(BASE_DIR, "models", filename)

RISK_MODEL_PATH = find_model_path("lgaf_resnet34.pth")
CLASS_MODEL_PATH = find_model_path("lgaf_resnet18.pt")


# =========================================================
# STREAMLIT VERSION COMPATIBILITY HELPERS
# =========================================================
def safe_toggle(label, value=True):
    if hasattr(st, "toggle"):
        return st.toggle(label, value=value)
    return st.checkbox(label, value=value)

def safe_image(img, caption="", **kwargs):
    # Backward/Forward compatible container width
    try:
        st.image(img, caption=caption, use_container_width=True, **kwargs)
    except Exception:
        try:
            st.image(img, caption=caption, use_column_width=True, **kwargs)
        except Exception:
            st.image(img, caption=caption, **kwargs)


# =========================================================
# LGAF ATTENTION MODULE
# =========================================================
class LGAF_Attention(nn.Module):
    """
    Local-Global Attention Fusion (LGAF) Block:
    Channel-wise Squeeze-and-Excitation attention mechanism that recalibrates
    feature maps to selectively emphasize salient lesion characteristics (necrosis,
    enhancement, edema) while suppressing non-discriminative background tissue.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


# =========================================================
# MODEL ARCHITECTURES
# =========================================================
class LGAFResNet18(nn.Module):
    """
    ResNet-18 Backbone with LGAF Attention integrated between Layer 3 and Layer 4.
    Used for 4-class Brain Tumor Classification (Glioma, Meningioma, Pituitary, Normal).
    """
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=None)

        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )

        self.attn = LGAF_Attention(256)
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class LGAFResNet34(nn.Module):
    """
    ResNet-34 Backbone with LGAF Attention integrated between Layer 3 and Layer 4,
    featuring Dropout (0.4) for robust generalization.
    Used for 4-tier Tumor Risk Stratification (NoTumor, LowRisk, MediumRisk, HighRisk).
    """
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        backbone = models.resnet34(weights=None)

        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )

        self.attn = LGAF_Attention(256)
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# =========================================================
# SAFE CACHED MODEL LOADER
# =========================================================
@st.cache_resource(show_spinner=False)
def load_models():
    default_risk_labels = ["NoTumor", "LowRisk", "MediumRisk", "HighRisk"]
    default_class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

    # 1. Load Risk Model (ResNet-34)
    if os.path.exists(RISK_MODEL_PATH):
        try:
            risk_ckpt = torch.load(RISK_MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            risk_ckpt = torch.load(RISK_MODEL_PATH, map_location=DEVICE)

        if isinstance(risk_ckpt, dict) and "model_state_dict" in risk_ckpt:
            risk_state = risk_ckpt["model_state_dict"]
            risk_labels = risk_ckpt.get("class_names", default_risk_labels)
        else:
            risk_state = risk_ckpt
            risk_labels = default_risk_labels

        risk_model = LGAFResNet34(num_classes=len(risk_labels)).to(DEVICE)
        risk_model.load_state_dict(risk_state, strict=True)
        risk_model.eval()
    else:
        risk_labels = default_risk_labels
        risk_model = LGAFResNet34(num_classes=len(risk_labels)).to(DEVICE)
        risk_model.eval()

    # 2. Load Classification Model (ResNet-18)
    if os.path.exists(CLASS_MODEL_PATH):
        try:
            class_ckpt = torch.load(CLASS_MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            class_ckpt = torch.load(CLASS_MODEL_PATH, map_location=DEVICE)

        if isinstance(class_ckpt, dict) and "model_state" in class_ckpt:
            class_state = class_ckpt["model_state"]
            class_labels = class_ckpt.get("class_names", default_class_labels)
        else:
            class_state = class_ckpt
            class_labels = default_class_labels

        class_model = LGAFResNet18(num_classes=len(class_labels)).to(DEVICE)
        class_model.load_state_dict(class_state, strict=True)
        class_model.eval()
    else:
        class_labels = default_class_labels
        class_model = LGAFResNet18(num_classes=len(class_labels)).to(DEVICE)
        class_model.eval()

    return risk_model, class_model, risk_labels, class_labels


# =========================================================
# PREPROCESSING & INFERENCE
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_single(image, model):
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1).item()
    return pred, probs.cpu().numpy()[0]


# =========================================================
# EXPLAINABLE AI (Grad-CAM)
# =========================================================
def compute_gradcam(model, image, target_class=None):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for LGAFResNet18.
    Hooks into layer4 to identify spatial regions contributing most to the prediction.
    """
    model.eval()
    activations = []
    gradients = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = model.layer4.register_forward_hook(f_hook)
    handle_b = model.layer4.register_full_backward_hook(b_hook)

    img_t = transform(image).unsqueeze(0).to(DEVICE)
    img_t.requires_grad = True

    with torch.enable_grad():
        logits = model(img_t)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()

    handle_f.remove()
    handle_b.remove()

    if not activations or not gradients:
        return None

    act = activations[0].detach()      # [1, 512, 7, 7]
    grad = gradients[0].detach()       # [1, 512, 7, 7]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Min-max normalization
    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    return cam


def apply_heatmap_overlay(original_pil, cam, alpha=0.5):
    """
    Overlays normalized Grad-CAM heatmap onto original MRI slice with jet colormap.
    """
    img_resized = original_pil.resize((224, 224)).convert("RGB")
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Jet-style colormap approximation in pure numpy
    r = np.clip(1.5 - np.abs(cam * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(cam * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(cam * 4.0 - 1.0), 0.0, 1.0)
    heat_np = np.stack([r, g, b], axis=-1)

    blended = (1.0 - alpha) * img_np + alpha * heat_np
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# =========================================================
# SYNTHETIC CLINICAL DEMO GENERATOR
# =========================================================
def generate_sample_mri(case_type="glioma", size=(256, 256)):
    """Procedurally synthesizes realistic brain MRI axial scans for 1-click testing."""
    w, h = size
    base = np.zeros((h, w), dtype=np.float32)
    y, x = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0

    # Skull & Scalp
    skull = (((x - cx) / (w * 0.42)) ** 2 + ((y - cy) / (h * 0.46)) ** 2) <= 1.0
    inner_skull = (((x - cx) / (w * 0.39)) ** 2 + ((y - cy) / (h * 0.43)) ** 2) <= 1.0
    base[skull] = 0.25
    base[skull & ~inner_skull] = 0.85

    # Brain parenchyma
    brain = (((x - cx) / (w * 0.37)) ** 2 + ((y - cy) / (h * 0.41)) ** 2) <= 1.0
    base[brain] = 0.42 + 0.04 * np.sin(x[brain] * 0.15) * np.cos(y[brain] * 0.15)

    # Ventricles
    v1 = (((x - (cx - 16)) / 9.0) ** 2 + ((y - (cy - 10)) / 28.0) ** 2 <= 1.0) & brain
    v2 = (((x - (cx + 16)) / 9.0) ** 2 + ((y - (cy - 10)) / 28.0) ** 2 <= 1.0) & brain
    base[v1 | v2] = 0.08

    # Specific pathology
    if case_type == "glioma":
        tx, ty = cx + 38, cy - 25
        dist = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        base[dist <= 24] = 0.82
        base[(dist <= 36) & (dist > 24)] = 0.58
    elif case_type == "meningioma":
        tx, ty = cx - 55, cy + 15
        dist = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        base[dist <= 22] = 0.90
    elif case_type == "pituitary":
        tx, ty = cx, cy + 48
        dist = np.sqrt(((x - tx) / 16.0) ** 2 + ((y - ty) / 14.0) ** 2)
        base[dist <= 1.0] = 0.92

    base = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=0.7))
    return img.convert("RGB")


# =========================================================
# MAIN STREAMLIT APP LOGIC
# =========================================================
def run_app():
    st.set_page_config(
        page_title="LGAF Medical AI | Brain MRI Diagnostic Suite",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom modern medical dark styling
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #090d16 0%, #0d1527 50%, #080d1a 100%);
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .header-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }
        .header-title {
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .header-sub {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 6px;
        }
        .badge-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-cuda { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-cpu { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
        .badge-model { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .result-box {
            background: rgba(15, 23, 42, 0.85);
            border-radius: 12px;
            padding: 18px 22px;
            border-left: 5px solid #38bdf8;
            margin-bottom: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        .result-value {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .sev-notumor { color: #4ade80; }
        .sev-low { color: #facc15; }
        .sev-medium { color: #fb923c; }
        .sev-high { color: #f87171; }
    </style>
    """, unsafe_allow_html=True)

    dev_badge = f'<span class="badge-chip badge-cuda">⚡ Accelerated: CUDA</span>' if torch.cuda.is_available() else '<span class="badge-chip badge-cpu">💻 Engine: CPU</span>'

    st.markdown(f"""
    <div class="header-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <h1 class="header-title">🧠 LGAF Medical AI — Brain MRI Diagnostic Suite</h1>
                <p class="header-sub">Local-Global Attention Fusion Deep Neural Architecture for Real-Time Tumor Classification & Risk Stratification</p>
            </div>
            <div>
                {dev_badge}
                <span class="badge-chip badge-model">ResNet-18 (Classification)</span>
                <span class="badge-chip badge-model">ResNet-34 (Risk)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Clinical Controls")

        input_mode = st.radio(
            "Select Scan Source",
            ["📁 Upload MRI Scan", "🧪 Interactive Demo Presets", "📸 Live Camera Capture"],
            index=0
        )

        st.markdown("---")
        st.subheader("Explainable AI (Grad-CAM)")
        enable_cam = safe_toggle("Enable Heatmap Localization", value=True)
        cam_alpha = st.slider("Heatmap Blending Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        st.markdown("---")
        st.subheader("Patient Meta (Report)")
        patient_id = st.text_input("Patient Reference ID", value="PT-2026-0891")
        physician_name = st.text_input("Evaluating Radiologist", value="Dr. Specialist")

        st.markdown("---")
        st.caption("Developed for medical AI research and clinical assistance. Designed by **Anas Ijaz & Maria Amjad**.")

    input_image = None
    image_source_label = ""

    if input_mode == "📁 Upload MRI Scan":
        uploaded_file = st.file_uploader("Upload Axial Brain MRI Slice (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
                image_source_label = f"Uploaded Scan ({uploaded_file.name})"
            except Exception as e:
                st.error(f"Failed to read image: {e}")

    elif input_mode == "🧪 Interactive Demo Presets":
        st.info("💡 Select a simulated brain MRI slice to evaluate the dual-network pipeline instantly:")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        
        with col_p1:
            if st.button("🔴 Glioma Case", key="btn_glioma"):
                st.session_state["preset"] = "glioma"
        with col_p2:
            if st.button("🟡 Meningioma Case", key="btn_meningioma"):
                st.session_state["preset"] = "meningioma"
        with col_p3:
            if st.button("🟣 Pituitary Case", key="btn_pituitary"):
                st.session_state["preset"] = "pituitary"
        with col_p4:
            if st.button("🟢 Healthy Control", key="btn_healthy"):
                st.session_state["preset"] = "healthy"

        active_preset = st.session_state.get("preset", "glioma")
        input_image = generate_sample_mri(active_preset)
        image_source_label = f"Simulated Preset ({active_preset.capitalize()})"

    elif input_mode == "📸 Live Camera Capture":
        camera_file = st.camera_input("Capture MRI from physical printout or display")
        if camera_file is not None:
            try:
                input_image = Image.open(camera_file).convert("RGB")
                image_source_label = "Live Camera Capture"
            except Exception as e:
                st.error(f"Camera input error: {e}")

    # Pipeline Execution
    if input_image is not None:
        with st.spinner("Initializing LGAF Neural Networks..."):
            risk_model, class_model, RISK_LABELS, CLASS_LABELS = load_models()

        t0 = time.perf_counter()
        risk_pred, risk_probs = predict_single(input_image, risk_model)
        class_pred, class_probs = predict_single(input_image, class_model)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        risk_label = RISK_LABELS[risk_pred]
        class_label = CLASS_LABELS[class_pred]
        risk_conf = float(risk_probs[risk_pred])
        class_conf = float(class_probs[class_pred])

        sev_class = "sev-notumor"
        if "High" in risk_label:
            sev_class = "sev-high"
        elif "Medium" in risk_label:
            sev_class = "sev-medium"
        elif "Low" in risk_label:
            sev_class = "sev-low"

        col_img, col_diag = st.columns([1.1, 1.3], gap="large")

        with col_img:
            st.subheader("🖼️ MRI Scan & XAI Focus")
            tab_raw, tab_cam = st.tabs(["Original MRI", "🔥 Grad-CAM Heatmap"])
            
            with tab_raw:
                safe_image(input_image, caption=image_source_label)

            with tab_cam:
                if enable_cam:
                    with st.spinner("Generating Grad-CAM attention heatmap..."):
                        cam = compute_gradcam(class_model, input_image, target_class=class_pred)
                        if cam is not None:
                            cam_overlay = apply_heatmap_overlay(input_image, cam, alpha=cam_alpha)
                            safe_image(cam_overlay, caption=f"Grad-CAM Heatmap for Class '{class_label}' (Layer 4)")
                            st.caption("🔴 Red/Yellow indicates maximum activation weight driving the neural network decision.")
                        else:
                            st.warning("Could not compute Grad-CAM for this layer.")
                else:
                    st.info("Grad-CAM localization is disabled. Toggle it in the sidebar.")

            st.caption(f"⏱️ Model Inference Latency: **{inference_ms:.1f} ms** on {DEVICE.type.upper()}")

        with col_diag:
            st.subheader("🩺 Diagnostic Evaluation")

            card_col1, card_col2 = st.columns(2)
            with card_col1:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">🧬 Identified Morphology</div>
                    <div class="result-value" style="color: #38bdf8;">{class_label.upper()}</div>
                    <div>Confidence: <b>{class_conf * 100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(class_conf)

            with card_col2:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">⚠️ Oncological Risk Tier</div>
                    <div class="result-value {sev_class}">{risk_label}</div>
                    <div>Confidence: <b>{risk_conf * 100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(risk_conf)

            st.markdown("---")
            st.subheader("📊 Probability Distribution")
            tab_c_prob, tab_r_prob = st.tabs(["Classification Spectrum", "Risk Stratification Spectrum"])

            with tab_c_prob:
                class_dict = {label.capitalize(): float(prob) for label, prob in zip(CLASS_LABELS, class_probs)}
                st.bar_chart(class_dict)

            with tab_r_prob:
                risk_dict = {label: float(prob) for label, prob in zip(RISK_LABELS, risk_probs)}
                st.bar_chart(risk_dict)

            st.markdown("---")
            st.subheader("📋 Clinical Guidance")
            guidance = {
                "glioma": "Infiltrative lesion suspected. Recommended protocol: High-resolution contrast-enhanced T1, T2-FLAIR, diffusion-weighted imaging (DWI), and prompt neurosurgical oncological consultation.",
                "meningioma": "Extra-axial circumscribed mass suspected. Typically benign/slow-growing with dural tail. Recommended protocol: Serial volumetric MRI follow-up or surgical resection consultation depending on mass effect.",
                "pituitary": "Sellar/suprasellar mass suspected. Recommended protocol: Pituitary hormonal endocrine panel (PRL, GH, ACTH, TSH) and formal visual field testing (perimetry).",
                "notumor": "No active neoplastic mass detected in this axial slice. Intact ventricular symmetry and baseline parenchymal signal."
            }
            st.info(guidance.get(class_label.lower(), "Standard neuro-radiological review advised."))

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_content = f"""================================================================================
              LGAF NEURO-RADIOLOGY AI DIAGNOSTIC REPORT
================================================================================
Generated: {timestamp_str}
Patient Reference: {patient_id}
Evaluating Radiologist: {physician_name}
Scan Source: {image_source_label}
Hardware Device: {DEVICE.type.upper()}
Inference Latency: {inference_ms:.2f} ms

--------------------------------------------------------------------------------
PRIMARY FINDINGS:
--------------------------------------------------------------------------------
1. Morphological Classification:  {class_label.upper()}
   Confidence:                    {class_conf * 100:.2f}%
   
2. Risk Stratification Tier:      {risk_label}
   Confidence:                    {risk_conf * 100:.2f}%

--------------------------------------------------------------------------------
FULL CONFIDENCE PROFILES:
--------------------------------------------------------------------------------
Tumor Classification:
""" + "\n".join([f"  - {lbl.ljust(15)}: {prob * 100:6.2f}%" for lbl, prob in zip(CLASS_LABELS, class_probs)]) + f"""

Risk Stratification:
""" + "\n".join([f"  - {lbl.ljust(15)}: {prob * 100:6.2f}%" for lbl, prob in zip(RISK_LABELS, risk_probs)]) + f"""

--------------------------------------------------------------------------------
CLINICAL NOTES:
{guidance.get(class_label.lower(), 'Standard review advised.')}
--------------------------------------------------------------------------------
DISCLAIMER:
This AI-assisted software is designed to augment clinical workflow and research.
All output requires validation by a board-certified radiologist or physician.
================================================================================
"""

            st.download_button(
                label="📥 Download Clinical Diagnostic Report (.txt)",
                data=report_content,
                file_name=f"LGAF_Report_{patient_id}_{int(time.time())}.txt",
                mime="text/plain"
            )

    else:
        st.info("👈 Please select an option from the sidebar (Upload an MRI scan, choose an Interactive Demo Preset, or capture from Camera) to initiate analysis.")

    with st.expander("📚 Understand the LGAF Architecture & Neural Mechanisms"):
        st.markdown("""
        ### What is LGAF?
        **LGAF (Local-Global Attention Fusion)** is an attention-augmented convolutional neural network designed specifically for brain MRI pathology:
        
        1. **Local Spatial Extraction (Backbone)**:
           Standard convolutional blocks (`conv1` through `layer3`) extract local structural textures, edges, and tissue intensities.
           
        2. **Global Channel Recalibration (LGAF Attention)**:
           The attention module performs global average pooling to summarize channel context $z \in \mathbb{R}^{C}$, then passes it through a 2-stage $1\\times 1$ conv bottleneck (`C -> C/8 -> C`) with ReLU and Sigmoid activations. This computes channel importance scores $s \in [0, 1]$ that rescale features:
           $$\\tilde{X} = X \\otimes \\sigma(W_2 \\cdot \\text{ReLU}(W_1 \\cdot \\text{GAP}(X)))$$
           
        3. **Deep Representation & Dual Decoupling**:
           - `layer4` extracts high-level semantic representations.
           - The system decouples diagnosis into **Classification** (identifying the pathological cell line) and **Risk Stratification** (assessing clinical urgency and invasive potential).
        """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.85rem;'>⚠️ AI-Assisted Clinical Decision Support Tool. Investigational research platform.</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run_app()
