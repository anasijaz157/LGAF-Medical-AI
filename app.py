import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RISK_MODEL_PATH  = os.path.join(MODEL_DIR, "lgaf_resnet34.pth")
CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "lgaf_resnet18.pt")

# =========================
# LGAF ATTENTION
# =========================
class LGAF_Attention(nn.Module):
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

# =========================
# LGAF RESNET-18 (CLASSIFICATION)
# =========================
class LGAFResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=False)

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

# =========================
# LGAF RESNET-34 (RISK)
# =========================
class LGAFResNet34(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        try:
            backbone = models.resnet34(weights=None)
        except TypeError:
            backbone = models.resnet34(pretrained=False)

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

# =========================
# LOAD MODELS (SAFE + CACHED)
# =========================
@st.cache_resource
def load_models():
    # =====================
    # RISK MODEL (ResNet34)
    # =====================
    try:
        risk_ckpt = torch.load(RISK_MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        risk_ckpt = torch.load(RISK_MODEL_PATH, map_location=DEVICE)

    if isinstance(risk_ckpt, dict) and "model_state_dict" in risk_ckpt:
        risk_state = risk_ckpt["model_state_dict"]
        risk_labels = risk_ckpt.get(
            "class_names",
            ["NoTumor", "LowRisk", "MediumRisk", "HighRisk"]
        )
    else:
        risk_state = risk_ckpt
        risk_labels = ["NoTumor", "LowRisk", "MediumRisk", "HighRisk"]

    risk_model = LGAFResNet34(num_classes=len(risk_labels)).to(DEVICE)
    risk_model.load_state_dict(risk_state, strict=True)
    risk_model.eval()

    # ============================
    # CLASSIFICATION (ResNet18)
    # ============================
    try:
        class_ckpt = torch.load(CLASS_MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        class_ckpt = torch.load(CLASS_MODEL_PATH, map_location=DEVICE)

    if isinstance(class_ckpt, dict) and "model_state" in class_ckpt:
        class_state = class_ckpt["model_state"]
        class_labels = class_ckpt.get(
            "class_names",
            ["glioma", "meningioma", "notumor", "pituitary"]
        )
    else:
        class_state = class_ckpt
        class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

    class_num_classes = len(class_labels)

    class_model = LGAFResNet18(num_classes=class_num_classes).to(DEVICE)
    class_model.load_state_dict(class_state, strict=True)
    class_model.eval()

    return risk_model, class_model, risk_labels, class_labels

# =========================
# IMAGE PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image, model):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1).item()
    return pred, probs.cpu().numpy()[0]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="LGAF Medical AI",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
body { background-color: #0f172a; }
.main {
    background-color: #020617;
    border-radius: 16px;
    padding: 30px;
}
h1, h2, h3 { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 LGAF Medical Diagnosis System")
st.write("**Risk Detection (ResNet-34) + Classification (ResNet-18)**")

uploaded = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    try:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    except TypeError:
        st.image(image, caption="Uploaded MRI", use_column_width=True)

    with st.spinner("Running LGAF models..."):
        risk_model, class_model, RISK_LABELS, CLASS_LABELS = load_models()

        risk_pred, risk_probs = predict(image, risk_model)
        class_pred, class_probs = predict(image, class_model)

    st.subheader("🩺 Risk Assessment")
    st.success(f"**{RISK_LABELS[risk_pred]}**")
    st.progress(float(risk_probs[risk_pred]))

    st.subheader("🧬 Tumor Classification")
    st.info(f"**{CLASS_LABELS[class_pred]}**")
    st.progress(float(class_probs[class_pred]))

    st.subheader("📊 Detailed Confidence Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🩺 Risk Levels")
        for label, prob in zip(RISK_LABELS, risk_probs):
            st.write(f"**{label}**: {prob * 100:.1f}%")
            st.progress(float(prob))
    with col2:
        st.markdown("##### 🧬 Tumor Types")
        for label, prob in zip(CLASS_LABELS, class_probs):
            st.write(f"**{label.capitalize()}**: {prob * 100:.1f}%")
            st.progress(float(prob))

    # =====================================================
    # Footer
    # =====================================================
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b;'>⚠️ AI-assisted research tool. Clinical validation required.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;'>Designed with ❤️ by <b>Anas Ijaz & Maria Amjad</b></p>",
        unsafe_allow_html=True
    )
