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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# U-NET ARCHITECTURE
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

    if not os.path.exists(MODEL_PATH) and GDRIVE_ID:
        with st.spinner("⬇️  Downloading model weights…"):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_ID}",
                MODEL_PATH,
                quiet=False
            )

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        return model, device, True
    else:
        model.eval()
        return model, device, False

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("L").resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    arr = (arr - p2) / (p98 - p2 + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

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
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🫁 MediScan AI — CT Lung Segmentation")

    with st.spinner("🔄 Initialising neural network…"):
        model, device, loaded = load_model()

    if not loaded:
        st.warning("⚠️ Demo Mode — No trained weights found.")

    uploaded = st.file_uploader("Upload a CT axial slice", type=["png","jpg","jpeg","tif","bmp"])
    if uploaded is None:
        st.info("👆 Upload a CT scan image above to begin analysis.")
        return

    pil_orig = Image.open(uploaded).convert("L")
    tensor   = preprocess(pil_orig)

    pred_np, mask = infer(model, device, tensor, threshold=0.5)

    st.success("✅ Analysis complete")
    st.image(mask*255, caption="Segmentation Mask", use_column_width=True)

if __name__ == "__main__":
    main()
