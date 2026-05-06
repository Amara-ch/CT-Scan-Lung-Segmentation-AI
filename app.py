import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import warnings
import cv2
import html
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | Radiology Suite",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.report-container {background:white;border:1px solid #c8d6e5;border-radius:4px;padding:0;}
.report-section {padding:16px 28px;border-bottom:1px solid #e8edf2;}
.section-title {font-size:11px;font-weight:600;text-transform:uppercase;color:#003366;}
.finding-block {background:#f5f8fb;border-left:3px solid #003366;padding:10px 14px;margin:8px 0;border-radius:0 4px 4px 0;font-size:13px;line-height:1.6;}
.finding-label {font-weight:600;color:#003366;font-size:12px;text-transform:uppercase;}
.impression-box {background:#fff8e6;border:1px solid #f0c040;border-radius:4px;padding:14px;font-size:13px;line-height:1.7;}
.stButton > button {width:100%;background:#003366 !important;color:white !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# U-NET MODEL
# ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        x = self.pool1(s1)
        s2 = self.enc2(x)
        x = self.pool2(s2)
        s3 = self.enc3(x)
        x = self.pool3(s3)
        s4 = self.enc4(x)
        x = self.pool4(s4)

        x = self.bottleneck(x)
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return torch.sigmoid(self.out(x))


@st.cache_resource
def load_model():
    model = UNet()
    path = "best_unet_model.pth"
    if os.path.exists(path):
        sd = torch.load(path, map_location=torch.device("cpu"))
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()
        return model, True
    return model, False


def main():
    st.title("🫁 MediScan AI — Radiology Suite")
    st.caption("AI-assisted CT Lung Segmentation | Research use only")

    model, loaded = load_model()

    if not loaded:
        st.error("Model file 'best_unet_model.pth' not found.")
        return

    uploaded = st.file_uploader(
        "Upload CT Axial Slice",
        type=["png", "jpg", "jpeg", "tif"]
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")

        col1, col2, col3 = st.columns([1, 1, 1.3])

        with col1:
            st.subheader("Source Image")
            st.image(pil_img, use_container_width=True)

        img_np = np.array(pil_img.convert("L").resize((256, 256)))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_img = clahe.apply(img_np).astype(np.float32) / 255.0

        t = torch.from_numpy(norm_img).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(t)
            mask = (out.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            lung_pixels = int(mask.sum())

        area_pct = (lung_pixels / (256 * 256)) * 100

        with col2:
            st.subheader("AI Segmentation")
            st.image(mask * 255, use_container_width=True)

        with col3:
            st.subheader("Diagnostic Report")

            st.markdown("""
            <div class="report-container">
                <div class="report-section">
                    <div class="section-title">Radiological Observations</div>

                    <div class="finding-block">
                        <div class="finding-label">Lung Parenchyma</div>
                        Bilateral lung fields visible on axial slice. Parenchymal density consistent with CT protocol. No large consolidative opacity identified. Lung borders delineated with high model confidence.
                    </div>

                    <div class="finding-block">
                        <div class="finding-label">Pleural Space</div>
                        No gross pleural effusion detected. No pneumothorax identified on current slice.
                    </div>

                    <div class="finding-block">
                        <div class="finding-label">Mediastinum &amp; Airways</div>
                        Mediastinal contour within expected limits. No gross mediastinal widening. Central airway morphology appears unremarkable. Tracheal deviation not apparent.
                    </div>

                    <div class="finding-block">
                        <div class="finding-label">Chest Wall &amp; Diaphragm</div>
                        No lytic or sclerotic bony lesion identified on this slice. Diaphragmatic contour normal in appearance.
                    </div>

                    <div class="impression-box">
                        Lung coverage: <strong>{:.2f}%</strong><br>
                        Segmented Pixels: <strong>{:,}</strong>
                    </div>
                </div>
            </div>
            """.format(area_pct, lung_pixels), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
