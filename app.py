# =============================================================================
# MediScan AI — Professional CT Scan Lung Segmentation
# Streamlit App | U-Net (PyTorch) | Hospital-Grade Interface
# =============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io, base64, os, warnings
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import gdown

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | CT Lung Segmentation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #050e18; color: #d0dce8; }
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#04111e,#071525); border-right:1px solid rgba(0,180,230,.12); }
.header-banner { background: linear-gradient(135deg,#04111e,#071b2e); border:1px solid rgba(0,180,230,.2); border-radius:16px; padding:26px 34px; margin-bottom:22px; display:flex; align-items:center; gap:20px; box-shadow:0 4px 40px rgba(0,140,200,.08); }
.header-title  { font-size:28px; font-weight:700; color:#00c8f0; letter-spacing:-0.5px; }
.header-sub    { font-size:13px; color:#5a8a9e; margin-top:4px; }
.header-badge  { margin-left:auto; background:rgba(0,180,230,.08); border:1px solid rgba(0,180,230,.25); border-radius:20px; padding:6px 16px; font-size:11px; color:#00c8f0; font-weight:600; text-transform:uppercase; letter-spacing:.8px; }
.metric-card   { background:linear-gradient(135deg,#07192a,#0a2035); border:1px solid rgba(0,180,230,.15); border-radius:12px; padding:18px 20px; text-align:center; margin-bottom:10px; }
.metric-value  { font-size:28px; font-weight:700; color:#00c8f0; }
.metric-label  { font-size:11px; color:#5a8a9e; margin-top:4px; text-transform:uppercase; letter-spacing:.8px; }
.info-box      { background:rgba(0,100,180,.08); border:1px solid rgba(0,150,220,.2); border-left:3px solid #0090d0; border-radius:8px; padding:12px 16px; font-size:12px; color:#7ab0cc; margin:10px 0; }
.warn-box      { background:rgba(255,160,0,.07); border:1px solid rgba(255,160,0,.2); border-left:3px solid #ffa000; border-radius:8px; padding:12px 16px; font-size:12px; color:#c09040; margin:10px 0; }
div[data-testid="stButton"]>button { background:linear-gradient(135deg,#005f8a,#007ab0); color:white; border:1px solid rgba(0,180,230,.4); border-radius:10px; font-weight:600; width:100%; padding:10px 24px; transition:all .2s; }
div[data-testid="stButton"]>button:hover { background:linear-gradient(135deg,#007ab0,#009ed4); transform:translateY(-1px); box-shadow:0 4px 20px rgba(0,160,220,.25); }
button[data-baseweb="tab"][aria-selected="true"] { color:#00c8f0 !important; border-bottom:2px solid #00c8f0 !important; background:transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# U-NET ARCHITECTURE  (exact replica from notebook)
# ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1       = EncoderBlock(1,   64)
        self.enc2       = EncoderBlock(64,  128)
        self.enc3       = EncoderBlock(128, 256)
        self.enc4       = EncoderBlock(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.dec4       = DecoderBlock(1024, 512)
        self.dec3       = DecoderBlock(512,  256)
        self.dec2       = DecoderBlock(256,  128)
        self.dec1       = DecoderBlock(128,  64)
        self.out        = nn.Conv2d(64, 1, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        s1, x = self.enc1(x);  s2, x = self.enc2(x)
        s3, x = self.enc3(x);  s4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4);  x = self.dec3(x, s3)
        x = self.dec2(x, s2);  x = self.dec1(x, s1)
        return self.sigmoid(self.out(x))


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "best_unet_model.pth"
GDRIVE_ID   = os.environ.get("MODEL_GDRIVE_ID", "")

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet().to(device)

    # Auto-download from Google Drive if secrets provide an ID
    if not os.path.exists(MODEL_PATH) and GDRIVE_ID:
        with st.spinner("⬇️  Downloading model weights…"):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}",
                           MODEL_PATH, quiet=False)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)   # handle both formats
        model.load_state_dict(state)
        model.eval()
        return model, device, True    # weights loaded
    else:
        model.eval()
        return model, device, False   # demo / random weights


# ─────────────────────────────────────────────────────────────
# PREPROCESSING  (mirrors notebook preprocess_scan)
# ─────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("L").resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    arr = (arr - p2) / (p98 - p2 + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,256,256)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def infer(model, device, tensor, threshold=0.5):
    with torch.no_grad():
        pred = model(tensor.to(device))
    pred_np = pred.squeeze().cpu().numpy()
    mask    = (pred_np > threshold).astype(np.uint8)
    return pred_np, mask


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(mask, pred_np):
    lung_pct   = round(mask.sum() / mask.size * 100, 2)
    confidence = float(pred_np[mask == 1].mean()) * 100 if mask.sum() > 0 else 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left  = mask[:, :128].sum();  right = mask[:, 128:].sum()
    sym   = round(min(left, right) / (max(left, right) + 1e-8) * 100, 1)
    return {
        "lung_pct"   : lung_pct,
        "confidence" : round(confidence, 1),
        "n_regions"  : len(contours),
        "symmetry"   : sym,
        "left_cov"   : round(left  / (128*256) * 100, 1),
        "right_cov"  : round(right / (128*256) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# OVERLAY BUILDERS
# ─────────────────────────────────────────────────────────────
def build_overlay(gray, mask, pred_np, alpha=0.45):
    rgb    = np.stack([gray]*3, axis=-1).astype(np.float32) / 255.0
    colour = plt.cm.cool(pred_np)[:, :, :3]
    out    = rgb.copy()
    out[mask == 1] = (1-alpha)*rgb[mask==1] + alpha*colour[mask==1]
    out_u8 = (out * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_u8, contours, -1, (0, 220, 255), 1)
    return out_u8

def build_heatmap(pred_np):
    return (plt.cm.plasma(pred_np)[:,:,:3] * 255).astype(np.uint8)

def show_img(ax, arr, title, cmap=None):
    ax.set_facecolor("#050e18")
    ax.imshow(arr, cmap=cmap)
    ax.set_title(title, color="#00c8f0", fontsize=9, pad=6)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:8px 0 18px'>
            <div style='font-size:40px'>🫁</div>
            <div style='color:#00c8f0;font-weight:700;font-size:17px;margin-top:6px'>MediScan AI</div>
            <div style='color:#3a6a80;font-size:11px'>v2.1 · Lung Segmentation</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### ⚙️ Analysis Settings")
        threshold     = st.slider("Segmentation Threshold", 0.10, 0.90, 0.50, 0.05)
        overlay_alpha = st.slider("Overlay Opacity",        0.10, 0.90, 0.45, 0.05)

        st.markdown("---")
        st.markdown("""
        <div class='info-box'>
        <b>Architecture:</b> U-Net (custom)<br>
        <b>Parameters:</b> 31,036,481<br>
        <b>Input:</b> 256×256 grayscale<br>
        <b>Loss:</b> Dice + BCE combined<br>
        <b>Dataset:</b> Finding Lungs in CT (Kaggle)<br>
        <b>Split:</b> 187 train · 40 val · 40 test<br>
        <b>Accuracy:</b> ~91%
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Patient Info *(optional)*")
        pid  = st.text_input("Patient ID",   placeholder="PT-20240506")
        name = st.text_input("Patient Name", placeholder="John Doe")
        date = st.date_input("Scan Date",    datetime.today())
        rad  = st.text_input("Radiologist",  placeholder="Dr. Ahmed")

        st.markdown("---")
        st.markdown("""
        <div class='warn-box'>
        ⚠️ <b>Clinical Disclaimer</b><br>
        For research & education only. Not a substitute for
        professional radiological assessment.
        </div>""", unsafe_allow_html=True)

    return threshold, overlay_alpha, {
        "id": pid or "N/A", "name": name or "Anonymous",
        "date": str(date),  "radiologist": rad or "N/A"
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    threshold, alpha, patient = sidebar()

    # Header
    st.markdown("""
    <div class='header-banner'>
        <div style='font-size:52px'>🫁</div>
        <div>
            <div class='header-title'>MediScan AI — CT Lung Segmentation</div>
            <div class='header-sub'>U-Net Deep Learning · MONAI Framework · Hospital-Grade Analysis</div>
        </div>
        <div class='header-badge'>AI-Assisted Diagnosis</div>
    </div>""", unsafe_allow_html=True)

    # Load model
    with st.spinner("🔄 Initialising neural network…"):
        model, device, loaded = load_model()

    if not loaded:
        st.markdown("""
        <div class='warn-box'>
        ⚠️ <b>Demo Mode</b> — No trained weights found (<code>best_unet_model.pth</code>).
        Place your weights file in the project root, or set the
        <code>MODEL_GDRIVE_ID</code> environment variable.
        </div>""", unsafe_allow_html=True)

    dev_label = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
    st.caption(f"Model ready · {dev_label} · Threshold: {threshold}")

    # Upload
    st.markdown("### 📂 Upload CT Scan")
    uploaded = st.file_uploader(
        "Drag & drop a CT axial slice",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )

    if uploaded is None:
        c1, c2, c3 = st.columns(3)
        for col, icon, label, desc in [
            (c1, "🔍", "Auto Segmentation",  "U-Net isolates lung tissue with pixel-level precision"),
            (c2, "📊", "Quantitative Metrics","Area %, symmetry, region count, AI confidence"),
            (c3, "📄", "Exportable Report",   "Download full analysis as PNG or CSV"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:30px'>{icon}</div>
                    <div class='metric-label' style='margin-top:8px'>{label}</div>
                    <div style='font-size:12px;color:#3a6a80;margin-top:6px'>{desc}</div>
                </div>""", unsafe_allow_html=True)
        st.info("👆 Upload a CT scan image above to begin analysis.", icon="🫁")
        return

    # ── Process ──────────────────────────────────────────────
    pil_orig = Image.open(uploaded).convert("L")
    tensor   = preprocess(pil_orig)

    bar = st.progress(0, text="Preprocessing…")
    bar.progress(25, text="Running U-Net…")
    pred_np, mask = infer(model, device, tensor, threshold)
    bar.progress(60, text="Computing metrics…")
    metrics = compute_metrics(mask, pred_np)
    bar.progress(85, text="Building overlays…")

    orig_256 = np.array(pil_orig.resize((256, 256), Image.LANCZOS))
    overlay  = build_overlay(orig_256, mask, pred_np, alpha)
    heatmap  = build_heatmap(pred_np)
    mask_rgb = np.stack([np.zeros_like(mask),
                         (mask*180).astype(np.uint8),
                         (mask*255).astype(np.uint8)], axis=-1)
    bar.progress(100, text="Done ✅"); bar.empty()

    st.success(
        f"✅ Analysis complete — Lung area: **{metrics['lung_pct']}%** · "
        f"Confidence: **{metrics['confidence']}%** · "
        f"Regions: **{metrics['n_regions']}** · "
        f"Symmetry: **{metrics['symmetry']}%**"
    )

    # ── Metric Dashboard ─────────────────────────────────────
    for col, val, label in zip(
        st.columns(5),
        [f"{metrics['lung_pct']}%", f"{metrics['confidence']}%",
         f"{metrics['symmetry']}%", str(metrics['n_regions']), dev_label.split()[1]],
        ["Lung Area", "Confidence", "Symmetry", "Regions", "Device"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🖼️ Scan Viewer", "📊 Detailed Analysis", "📄 Report"])

    # ── TAB 1: Scan Viewer ────────────────────────────────────
    with tab1:
        cols = st.columns(4)
        views = [
            (orig_256,  "gray",  "Original CT"),
            (mask_rgb,  None,    "Binary Mask"),
            (overlay,   None,    "Segmentation Overlay"),
            (heatmap,   None,    "Probability Heatmap"),
        ]
        for col, (arr, cmap, title) in zip(cols, views):
            with col:
                fig, ax = plt.subplots(figsize=(4, 4), facecolor="#050e18")
                show_img(ax, arr, title, cmap)
                plt.tight_layout(pad=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Side-by-side comparison slider simulation
        st.markdown("#### 🔬 Left vs Right Lung Coverage")
        col_l, col_r = st.columns(2)
        with col_l:
            st.metric("Left Lung Coverage",  f"{metrics['left_cov']}%")
            st.progress(int(metrics['left_cov']))
        with col_r:
            st.metric("Right Lung Coverage", f"{metrics['right_cov']}%")
            st.progress(int(metrics['right_cov']))

    # ── TAB 2: Detailed Analysis ──────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📈 Probability Distribution")
            fig3, ax3 = plt.subplots(figsize=(5, 3), facecolor="#050e18")
            ax3.set_facecolor("#07192a")
            ax3.hist(pred_np.flatten(), bins=80, color="#00a8d0", alpha=0.8, edgecolor="none")
            ax3.axvline(threshold, color="#ff6060", linewidth=1.5,
                        linestyle="--", label=f"Threshold={threshold}")
            ax3.set_xlabel("Probability", color="#5a8a9e", fontsize=9)
            ax3.set_ylabel("Pixel Count",  color="#5a8a9e", fontsize=9)
            ax3.tick_params(colors="#5a8a9e", labelsize=8)
            ax3.spines[:].set_color("#1a3a5e")
            ax3.legend(fontsize=8, labelcolor="#d0dce8", facecolor="#07192a")
            ax3.set_title("Prediction Probability Distribution", color="#00c8f0", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        with col_b:
            st.markdown("#### 🎚️ Threshold Sensitivity")
            thresholds = np.linspace(0.1, 0.9, 40)
            coverages  = [(pred_np > t).sum() / pred_np.size * 100 for t in thresholds]
            fig4, ax4 = plt.subplots(figsize=(5, 3), facecolor="#050e18")
            ax4.set_facecolor("#07192a")
            ax4.plot(thresholds, coverages, color="#00c8f0", linewidth=2)
            ax4.axvline(threshold, color="#ff6060", linewidth=1.5,
                        linestyle="--", label=f"Current={threshold}")
            ax4.fill_between(thresholds, coverages, alpha=0.15, color="#00a8d0")
            ax4.set_xlabel("Threshold",    color="#5a8a9e", fontsize=9)
            ax4.set_ylabel("Coverage (%)", color="#5a8a9e", fontsize=9)
            ax4.tick_params(colors="#5a8a9e", labelsize=8)
            ax4.spines[:].set_color("#1a3a5e")
            ax4.legend(fontsize=8, labelcolor="#d0dce8", facecolor="#07192a")
            ax4.set_title("Coverage vs Threshold Curve", color="#00c8f0", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        # Row-by-row signal profile
        st.markdown("#### 📉 Vertical Intensity Profile")
        col_mean = orig_256.mean(axis=1)
        mask_col = mask.mean(axis=1) * 100
        fig5, ax5 = plt.subplots(figsize=(10, 2.5), facecolor="#050e18")
        ax5.set_facecolor("#07192a")
        ax5.plot(col_mean,  color="#5a8a9e", linewidth=1.2, label="CT Intensity (mean)")
        ax5_r = ax5.twinx()
        ax5_r.plot(mask_col, color="#00c8f0", linewidth=1.2,
                   linestyle="--", alpha=0.85, label="Lung Mask Coverage %")
        ax5_r.tick_params(colors="#5a8a9e", labelsize=8)
        ax5.set_xlabel("Row (pixels)", color="#5a8a9e", fontsize=9)
        ax5.tick_params(colors="#5a8a9e", labelsize=8)
        ax5.spines[:].set_color("#1a3a5e"); ax5_r.spines[:].set_color("#1a3a5e")
        ax5.set_title("Row-wise Intensity vs Mask Coverage", color="#00c8f0", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    # ── TAB 3: Report ─────────────────────────────────────────
    with tab3:
        st.markdown("#### 📄 Structured Radiology Report")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        finding_lung = "NORMAL" if 20 <= metrics["lung_pct"] <= 60 else (
            "REDUCED" if metrics["lung_pct"] < 20 else "INCREASED")
        sym_status   = "SYMMETRIC" if metrics["symmetry"] >= 80 else "ASYMMETRIC"

        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
        <tr style='background:rgba(0,100,160,.2)'>
            <th style='color:#00a8d0;padding:10px 14px;text-align:left;font-size:11px;
                       text-transform:uppercase;letter-spacing:.5px' colspan='4'>
                Patient & Scan Information
            </th>
        </tr>
        <tr>
            <td style='padding:8px 14px;color:#5a8a9e;width:20%'>Patient ID</td>
            <td style='padding:8px 14px;color:#d0dce8'>{patient['id']}</td>
            <td style='padding:8px 14px;color:#5a8a9e'>Patient Name</td>
            <td style='padding:8px 14px;color:#d0dce8'>{patient['name']}</td>
        </tr>
        <tr style='background:rgba(0,0,0,.15)'>
            <td style='padding:8px 14px;color:#5a8a9e'>Scan Date</td>
            <td style='padding:8px 14px;color:#d0dce8'>{patient['date']}</td>
            <td style='padding:8px 14px;color:#5a8a9e'>Report Time</td>
            <td style='padding:8px 14px;color:#d0dce8'>{now}</td>
        </tr>
        <tr>
            <td style='padding:8px 14px;color:#5a8a9e'>Radiologist</td>
            <td style='padding:8px 14px;color:#d0dce8'>{patient['radiologist']}</td>
            <td style='padding:8px 14px;color:#5a8a9e'>AI Model</td>
            <td style='padding:8px 14px;color:#d0dce8'>U-Net · MediScan AI v2.1</td>
        </tr>
        <tr style='background:rgba(0,100,160,.2)'>
            <th style='color:#00a8d0;padding:10px 14px;text-align:left;font-size:11px;
                       text-transform:uppercase;letter-spacing:.5px' colspan='4'>
                Segmentation Findings
            </th>
        </tr>
        <tr>
            <td style='padding:8px 14px;color:#5a8a9e'>Lung Area</td>
            <td style='padding:8px 14px;color:#d0dce8'>{metrics['lung_pct']}% 
                <span style='color:{"#00d080" if finding_lung=="NORMAL" else "#ff6060"};
                font-size:11px;font-weight:600;margin-left:8px'>{finding_lung}</span>
            </td>
            <td style='padding:8px 14px;color:#5a8a9e'>AI Confidence</td>
            <td style='padding:8px 14px;color:#d0dce8'>{metrics['confidence']}%</td>
        </tr>
        <tr style='background:rgba(0,0,0,.15)'>
            <td style='padding:8px 14px;color:#5a8a9e'>L/R Symmetry</td>
            <td style='padding:8px 14px;color:#d0dce8'>{metrics['symmetry']}%
                <span style='color:{"#00d080" if sym_status=="SYMMETRIC" else "#f0c000"};
                font-size:11px;font-weight:600;margin-left:8px'>{sym_status}</span>
            </td>
            <td style='padding:8px 14px;color:#5a8a9e'>Regions Found</td>
            <td style='padding:8px 14px;color:#d0dce8'>{metrics['n_regions']}</td>
        </tr>
        <tr>
            <td style='padding:8px 14px;color:#5a8a9e'>Left Coverage</td>
            <td style='padding:8px 14px;color:#d0dce8'>{metrics['left_cov']}%</td>
            <td style='padding:8px 14px;color:#5a8a9e'>Right Coverage</td>
            <td style='padding:8px 14px;color:#d0dce8'>{metrics['right_cov']}%</td>
        </tr>
        <tr style='background:rgba(0,100,160,.2)'>
            <th style='color:#00a8d0;padding:10px 14px;text-align:left;font-size:11px;
                       text-transform:uppercase;letter-spacing:.5px' colspan='4'>
                Technical Parameters
            </th>
        </tr>
        <tr>
            <td style='padding:8px 14px;color:#5a8a9e'>Threshold</td>
            <td style='padding:8px 14px;color:#d0dce8'>{threshold}</td>
            <td style='padding:8px 14px;color:#5a8a9e'>Input Resolution</td>
            <td style='padding:8px 14px;color:#d0dce8'>256 × 256 px</td>
        </tr>
        <tr style='background:rgba(0,0,0,.15)'>
            <td style='padding:8px 14px;color:#5a8a9e'>Compute</td>
            <td style='padding:8px 14px;color:#d0dce8'>{dev_label}</td>
            <td style='padding:8px 14px;color:#5a8a9e'>Preprocessing</td>
            <td style='padding:8px 14px;color:#d0dce8'>p2–p98 clip + normalize</td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download composite image
        fig_rep, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="#050e18")
        fig_rep.suptitle(
            f"MediScan AI Report  |  Patient: {patient['name']}  |  {now}",
            color="#00c8f0", fontsize=11, y=1.02
        )
        for ax, (arr, cmap, title) in zip(axes, [
            (orig_256, "gray", "Original CT"),
            (mask_rgb, None,   "Lung Mask"),
            (overlay,  None,   "Overlay"),
            (heatmap,  None,   "Heatmap"),
        ]):
            show_img(ax, arr, title, cmap)
        plt.tight_layout()

        buf = io.BytesIO()
        fig_rep.savefig(buf, format="png", dpi=150,
                        facecolor="#050e18", bbox_inches="tight")
        plt.close(fig_rep)
        buf.seek(0)

        st.download_button(
            label="⬇️  Download Report Image (PNG)",
            data=buf,
            file_name=f"mediscan_report_{patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
