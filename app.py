import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import warnings
import cv2
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & PROFESSIONAL THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | Radiology Suite",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
/* Global font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header bar */
.header-bar {
    background: linear-gradient(90deg, #003366, #004a99);
    color: white;
    padding: 16px 28px;
    font-size: 20px;
    font-weight: 700;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

/* Card styling */
.card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    padding: 20px;
    margin-bottom: 20px;
}
.card h3 {
    color: #003366;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Metric cards */
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #f0f5fb, #e8eff9);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.metric-card .val {
    font-size: 28px;
    font-weight: 700;
    color: #003366;
}
.metric-card .lbl {
    font-size: 12px;
    color: #5a6a7a;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: #003366 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    height: 3em;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover { background: #004a99 !important; }
</style>
<div class="header-bar">🏥 MediScan AI | Radiology Suite</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# U-NET ARCHITECTURE (Unchanged)
# ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(DoubleConv(1, 64));   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(DoubleConv(64, 128)); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(DoubleConv(128, 256));self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(DoubleConv(256, 512));self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2);  self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2);  self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2);   self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x); x = self.pool1(s1)
        s2 = self.enc2(x); x = self.pool2(s2)
        s3 = self.enc3(x); x = self.pool3(s3)
        s4 = self.enc4(x); x = self.pool4(s4)
        x  = self.bottleneck(x)
        x  = self.dec4(torch.cat([self.up4(x), s4], 1))
        x  = self.dec3(torch.cat([self.up3(x), s3], 1))
        x  = self.dec2(torch.cat([self.up2(x), s2], 1))
        x  = self.dec1(torch.cat([self.up1(x), s1], 1))
        return torch.sigmoid(self.out(x))

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet()
    path = "best_unet_model.pth"
    if os.path.exists(path):
        try:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd)
            model.eval()
            return model, True
        except:
            return model, False
    return model, False

# ─────────────────────────────────────────────────────────────
# PROFESSIONAL REPORT GENERATOR (Unchanged logic)
# ─────────────────────────────────────────────────────────────
def generate_professional_report(patient_info, lung_pixels, area_pct, scan_info):
    now = datetime.now()
    date_str   = now.strftime("%d %B %Y")
    time_str   = now.strftime("%H:%M")
    report_id  = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"
    coverage_status = "Within normal range" if 20 <= area_pct <= 60 else "Requires review"

    report = f"""
MediScan AI Radiology — Diagnostic Report
Report ID: {report_id}
Date: {date_str} | Time: {time_str}
Patient: {patient_info['name']} | MRN: {patient_info['mrn']}
Age/Gender: {patient_info['age']} / {patient_info['gender']}
Referring Physician: {patient_info['referring_doc']}
Indication: {patient_info['indication']}

Segmented Lung Pixels: {lung_pixels:,}
Coverage: {area_pct:.2f}% ({coverage_status})

Recommendation: Clinical correlation with full CT series advised.
Disclaimer: AI-assisted report for research use only.
"""
    return report.strip()

# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("### 👤 Patient Details")
        p_name    = st.text_input("Full Name", "John Doe")
        p_id      = st.text_input("Medical Record No.", "MRN-10293")
        p_dob     = st.text_input("Date of Birth", "01-01-1980")
        p_gender  = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_age     = st.number_input("Age (years)", min_value=0, max_value=120, value=44)

        st.markdown("### 🩺 Clinical Information")
        p_ref_doc   = st.text_input("Referring Physician", "Dr. A. Khan")
        p_indication = st.text_input("Clinical Indication", "Cough, Breathlessness — Rule out consolidation")

        st.markdown("### ⚙️ Scan Parameters")
        p_contrast  = st.selectbox("Contrast", ["Non-Contrast", "IV Contrast", "Oral + IV Contrast"])
        p_slice_thk = st.selectbox("Slice Thickness (mm)", ["1.0", "1.5", "2.5", "5.0"])

    st.title("🫁 AI-Assisted CT Lung Segmentation")
    st.caption("For research and educational use only")

    model, loaded = load_model()
    if not loaded:
        st.error("⚠️ Model weights file ('best_unet_model.pth') not found.")
        return

    uploaded = st.file_uploader("📂 Upload CT Axial Slice", type=["png", "jpg", "
