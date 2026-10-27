# 🧠 LGAF Medical AI — Brain MRI Diagnostic Suite

> **Local-Global Attention Fusion (LGAF) Deep Neural Network for Real-Time Brain Tumor Classification & Oncological Risk Stratification**  
> *Developed by Anas Ijaz & Maria Amjad*

---

## 📌 Executive Summary

**LGAF Medical AI** is a dual-stream deep learning clinical diagnostic assistant designed to analyze axial brain Magnetic Resonance Imaging (MRI) scans. The system simultaneously performs two complementary medical tasks:
1. **Histological Tumor Classification (ResNet-18 + LGAF Attention)**: Differentiates between **Glioma**, **Meningioma**, **Pituitary Adenoma**, and **Healthy Brain (No Tumor)**.
2. **Oncological Risk Stratification (ResNet-34 + LGAF Attention)**: Evaluates tumor severity and urgency tier (**No Tumor**, **Low Risk**, **Medium Risk**, **High Risk**).
3. **Explainable AI (Grad-CAM)**: Generates real-time spatial heatmaps overlaying the brain MRI, allowing neuro-radiologists to verify the exact focal features influencing the neural network's decision.

---

## 📂 Comprehensive File-by-File Breakdown

```
LGAF/
├── app.py                  # Full-stack Streamlit medical dashboard & inference engine
├── requirements.txt        # Pinned Python package dependencies
├── models/
│   ├── lgaf_resnet18.pt    # Pre-trained weights for 4-class tumor classification (~43 MB)
│   └── lgaf_resnet34.pth   # Pre-trained weights for 4-tier risk stratification (~81 MB)
├── Dockerfile              # Production-grade containerization with PyTorch CPU optimization
├── docker-compose.yml      # Single-command local container orchestrator
├── run.bat                 # Windows 1-click virtual environment builder & runner
├── run.sh                  # Linux/macOS 1-click launcher script
├── .dockerignore           # Excludes local caches from Docker build context
├── .gitignore              # Standard git exclusion rules
└── README.md               # Architecture documentation & deployment guide
```

### 1. `app.py`
The nerve center of the application:
- **`LGAF_Attention(nn.Module)`**: A channel-wise attention block (Squeeze-and-Excitation principle). It uses `AdaptiveAvgPool2d(1)` to extract global context across channels, passes it through a $1 \times 1$ conv bottleneck (`channels // 8`), and applies Sigmoid gating to recalibrate feature maps ($X \cdot \sigma(W_2 \text{ReLU}(W_1 \text{GAP}(X)))$). This amplifies contrast-enhanced tumor margins and suppresses skull/normal tissue noise.
- **`LGAFResNet18`**: Takes ResNet-18 layers 1-3, inserts the `LGAF_Attention(256)` module before `layer4`, followed by global pooling and a 4-class linear head.
- **`LGAFResNet34`**: Deeper 34-layer backbone with LGAF attention and a `Dropout(0.4)` layer before the 4-tier risk classification head.
- **`load_models()`**: Cached using Streamlit's `@st.cache_resource` for zero-latency consecutive inferences. Includes multi-path discovery (`models/`, Desktop, system paths) and handles both PyTorch 2.0-2.6 `weights_only` compatibility.
- **`compute_gradcam()`**: PyTorch backward hook into `layer4` extracting feature activations and gradients to generate a spatial activation map.
- **Clinical Presets & Camera**: Provides 1-click synthetic brain MRI scans (Glioma, Meningioma, Pituitary, Normal) for instant demos when raw scans are unavailable, alongside live camera input.
- **Diagnostic Report Generator**: Dynamically compiles a structured clinical findings document ready for export as `.txt`.

### 2. `models/` Checkpoints
- **`lgaf_resnet18.pt` (42.8 MB)**: Checkpoint dictionary holding `model_state` tensor weights and class mapping `['glioma', 'meningioma', 'notumor', 'pituitary']`.
- **`lgaf_resnet34.pth` (81.4 MB)**: Checkpoint dictionary holding `model_state_dict` tensor weights and risk categories `['NoTumor', 'LowRisk', 'MediumRisk', 'HighRisk']`.

### 3. `requirements.txt`
Specifies minimal, conflict-free versions:
- `streamlit`: Reactive web UI framework.
- `torch` & `torchvision`: Neural network tensors, backprop, and ResNet backbone structures.
- `numpy`: Fast vectorized matrix computations for probabilities and Jet heatmap color maps.
- `pillow`: Medical image I/O, transforms, and filtering.

### 4. `Dockerfile` & `docker-compose.yml`
- Built on `python:3.10-slim`.
- Uses `--index-url https://download.pytorch.org/whl/cpu` for PyTorch to reduce image size from 5+ GB down to ~900 MB.
- Configures Streamlit headless server flags and exposes port `8501`.
- Built-in `HEALTHCHECK` ensuring automated cloud orchestrator resilience.

---

## 🚀 Deployment Instructions

### Method 1: Local Windows Deployment (1-Click)
Double-click `run.bat` in this folder, or execute via Command Prompt / PowerShell:
```cmd
cd LGAF
run.bat
```
*The script will automatically create a virtual environment (`.venv`), install requirements, and open `http://localhost:8501` in your default browser.*

---

### Method 2: Local Linux / macOS Deployment
Open your terminal:
```bash
cd LGAF
chmod +x run.sh
./run.sh
```

---

### Method 3: Docker Deployment (Containerized)
Run anywhere Docker is installed with zero Python setup:
```bash
# Build and run with Docker Compose
docker compose up --build

# Or with raw Docker:
docker build -t lgaf-medical-ai .
docker run -d -p 8501:8501 --name lgaf_ai lgaf-medical-ai
```
Navigate to `http://localhost:8501`.

---

### Method 4: Deploying to Hugging Face Spaces (Free Cloud Hosting)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Streamlit** as the Space SDK and choose **Public** or **Private**.
3. Clone the space repository locally or upload files via the web interface:
   - `app.py`
   - `requirements.txt`
   - `models/lgaf_resnet18.pt`
   - `models/lgaf_resnet34.pth`
4. Commit and push. Hugging Face will automatically install dependencies and launch the live URL within 2 minutes!

---

### Method 5: Deploying to Streamlit Community Cloud (Free)
1. Push your project to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: LGAF Medical AI"
   git remote add origin https://github.com/YOUR_USERNAME/LGAF-Medical-AI.git
   git push -u origin main
   ```
   *(Note: The model files are 42MB and 81MB, both below GitHub's 100MB limit. If you use Git LFS, install `git lfs track "models/*"`).*
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account, select your repository, specify `app.py` as the entrypoint, and click **Deploy**.

---

## 🔬 Scientific & Technical Specifications
- **Input Dimension**: $224 \times 224 \times 3$ (RGB normalized with ImageNet $\mu, \sigma$).
- **Inference Latency**: ~15-25 ms on NVIDIA CUDA; ~45-80 ms on modern multi-core CPU.
- **Attention Type**: Spatial-Channel Squeeze & Excitation ($C=256$, Reduction ratio $r=8$).
- **Validation Guidelines**: Intended for investigational clinical decision support. Always verify findings with certified neuro-radiological review.
