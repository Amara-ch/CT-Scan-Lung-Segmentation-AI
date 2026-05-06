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
# MODERN PROFESSIONAL CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | Radiology Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 2rem; }
    
    h1 { font-family: 'Playfair Display', serif; color: #0f172a; }
    .stButton > button {
        background: #003366 !important;
        color: white !important;
        border-radius: 8px !important;
        height: 52px !important;
        font-weight: 600 !important;
    }
    
    .report-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    .report-header {
        background: linear-gradient(135deg, #003366, #1e40af);
        color: white;
        padding: 25px 30px;
    }
    .section-title {
        color: #003366;
        border-bottom: 3px solid #003366;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ================= U-NET (Unchanged) =================
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
        x = self.bottleneck(x)
        x = self.dec4(torch.cat([self.up4(x), s4], 1))
        x = self.dec3(torch.cat([self.up3(x), s3], 1))
        x = self.dec2(torch.cat([self.up2(x), s2], 1))
        x = self.dec1(torch.cat([self.up1(x), s1], 1))
        return torch.sigmoid(self.out(x))

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

# ================= DYNAMIC REPORT =================
def generate_professional_report(patient_info, lung_pixels, area_pct, scan_info):
    now = datetime.now()
    report_id = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"
    coverage_status = "Within normal range" if 20 <= area_pct <= 60 else "Requires review"
    
    # Dynamic text based on indication
    ind = patient_info['indication'].lower()
    if 'pneumonia' in ind or 'consolidation' in ind:
        impression = "Findings are suggestive of possible consolidation. Clinical correlation with symptoms and labs is advised."
    elif 'mass' in ind or 'cancer' in ind or 'tumor' in ind:
        impression = "No obvious mass lesion seen on this slice. Further dedicated imaging recommended."
    elif 'effusion' in ind:
        impression = "No significant pleural effusion detected on current slice."
    else:
        impression = "No acute gross abnormality detected on the submitted axial slice."

    report = f"""MEDISCAN AI RADIOLOGY — OFFICIAL DIAGNOSTIC REPORT
================================================================================
Report ID       : {report_id}
Report Date     : {now.strftime("%d %B %Y")}
Report Time     : {now.strftime("%H:%M")} (PKT)

PATIENT INFORMATION
Patient Name    : {patient_info['name']}
MRN             : {patient_info['mrn']}
Age/Gender      : {patient_info['age']} years / {patient_info['gender']}
Indication      : {patient_info['indication']}

QUANTITATIVE FINDINGS
Lung Pixels     : {lung_pixels:,}
Coverage        : {area_pct:.2f}% ({coverage_status})

IMPRESSION
{impression}
Multi-slice review and clinical correlation strongly recommended.

Disclaimer: This is AI generated report for research/educational purpose only.
"""
    return report

# ===================== MAIN =====================
def main():
    with st.sidebar:
        st.title("🏥 MediScan AI")
        st.markdown("**Advanced Lung Segmentation**")
        st.divider()
        
        st.subheader("Patient Details")
        p_name = st.text_input("Full Name", "John Doe")
        p_id = st.text_input("MRN", "MRN-10293")
        p_age = st.number_input("Age", 0, 120, 45)
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.divider()
        st.subheader("Clinical Information")
        p_indication = st.text_input("Clinical Indication", 
                                   "Cough, Breathlessness — Rule out consolidation",
                                   help="Write detailed indication for better report")
        
        st.divider()
        st.subheader("Scan Info")
        p_contrast = st.selectbox("Contrast", ["Non-Contrast", "IV Contrast"])
        p_slice = st.selectbox("Slice Thickness", ["1.0 mm", "1.5 mm", "5.0 mm"])

    st.title("🫁 MediScan AI Radiology Suite")
    st.caption("Professional AI Lung Segmentation System")

    model, loaded = load_model()
    if not loaded:
        st.error("Model file not found!")
        return

    uploaded = st.file_uploader("Upload Axial CT Slice", type=["png", "jpg", "jpeg", "tif"])

    if uploaded:
        col1, col2, col3 = st.columns([1.1, 1.1, 1.3])
        
        pil_img = Image.open(uploaded)
        
        with col1:
            st.subheader("Original Image")
            st.image(pil_img, use_column_width=True)

        # Processing
        img_np = np.array(pil_img.convert("L").resize((256, 256)))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm_img = clahe.apply(img_np).astype(np.float32)/255.0
        t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(t)
            mask = (out.squeeze().numpy() > 0.5).astype(np.uint8)
            lung_pixels = int(np.sum(mask))

        area_pct = (lung_pixels / 65536) * 100

        with col2:
            st.subheader("AI Segmentation")
            st.image(mask*255, use_column_width=True)
            st.success(f"Lung Coverage: **{area_pct:.1f}%**")

        with col3:
            st.subheader("📋 AI Diagnostic Report")
            
            patient_info = {
                "name": p_name, "mrn": p_id, "age": p_age, 
                "gender": p_gender, "indication": p_indication
            }
            scan_info = {"contrast": p_contrast, "slice_thickness": p_slice}
            
            report_text = generate_professional_report(patient_info, lung_pixels, area_pct, scan_info)
            
            st.text_area("", report_text, height=500)
            
            st.download_button(
                "📥 Download Full Report",
                data=report_text,
                file_name=f"MediScan_{p_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
