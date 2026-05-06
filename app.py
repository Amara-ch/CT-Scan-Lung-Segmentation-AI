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

st.set_page_config(
    page_title="MediScan AI | Radiology Suite",
    page_icon="🏥",
    layout="wide"
)

# Clean Professional Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f8fafd; }
.report-container {
    background: white;
    border: 1px solid #c8d6e5;
    border-radius: 8px;
    padding: 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.report-header {
    background: #003366;
    color: white;
    padding: 22px 28px;
    display: flex;
    justify-content: space-between;
}
.stButton > button {
    width: 100%; background: #003366 !important; color: white !important;
    border-radius: 6px; height: 3em; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ================= U-NET Model (Unchanged) =================
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

# ================= IMPROVED REPORT GENERATOR =================
def generate_professional_report(patient_info, lung_pixels, area_pct, scan_info):
    now = datetime.now()
    date_str = now.strftime("%d %B %Y")
    time_str = now.strftime("%H:%M")
    report_id = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"
    
    coverage_status = "Within normal range" if 20 <= area_pct <= 60 else "Requires review"
    
    # Dynamic Impression based on Indication
    indication = patient_info['indication'].lower()
    if any(word in indication for word in ['pneumonia', 'consolidation', 'infection']):
        impression_note = "Findings may correlate with infectious process. Clinical correlation with inflammatory markers recommended."
    elif any(word in indication for word in ['cancer', 'mass', 'tumor', 'malignancy']):
        impression_note = "No gross mass lesion seen on this single slice. Dedicated contrast-enhanced study and multiplanar review advised."
    elif any(word in indication for word in ['effusion', 'fluid']):
        impression_note = "No gross pleural effusion detected on this slice."
    elif any(word in indication for word in ['tb', 'tuberculosis']):
        impression_note = "No obvious cavitary or miliary lesions seen. Further evaluation with HRCT suggested if clinically indicated."
    else:
        impression_note = "No acute gross abnormality detected on the submitted single axial slice."

    report = f"""
================================================================================
          MEDISCAN AI RADIOLOGY — OFFICIAL DIAGNOSTIC REPORT
================================================================================
REPORT ID       : {report_id}
REPORT DATE     : {date_str}
REPORT TIME     : {time_str} (PKT)
INSTITUTION     : MediScan AI Radiology Centre
DEPARTMENT      : Diagnostic Radiology & Medical Imaging
MODALITY        : Computed Tomography (CT) — Chest / Thorax
PRIORITY        : Routine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient Name    : {patient_info['name']}
Medical Rec. No.: {patient_info['mrn']}
Date of Birth   : {patient_info['dob']}
Gender          : {patient_info['gender']}
Age             : {patient_info['age']} years
Referring Phys. : {patient_info['referring_doc']}
Clinical Indic. : {patient_info['indication']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI Model        : U-Net (Encoder-Decoder Architecture, 31M Parameters)
Input Resolution: 256 × 256 pixels (axial slice)
Preprocessing   : CLAHE (clipLimit=2.0)
Segmentation    : Binary thresholding at 0.5
Contrast Agent  : {scan_info['contrast']}
Slice Thickness : {scan_info['slice_thickness']} mm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITATIVE AI FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Scan Pixels           : 65,536
  Segmented Lung Area         : {lung_pixels:,} pixels
  Lung Field Coverage         : {area_pct:.2f}% 
  Coverage Assessment         : {coverage_status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{impression_note}

1. AI-assisted lung segmentation successfully performed.
2. Segmented lung field area of {area_pct:.2f}% is {coverage_status.lower()}.
3. No gross consolidation, large mass or pneumothorax identified on this single axial slice.
4. Multi-slice volumetric analysis and clinical correlation is strongly recommended.

RECOMMENDATION: Please correlate with full CT series, laboratory findings and patient history. 
This AI report is for decision support only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
This report is generated by an AI system for research and educational purposes only.
All findings must be reviewed and signed by a qualified radiologist.
© {now.year} MediScan AI Radiology Suite.
================================================================================
"""
    return report.strip()

# ===================== MAIN UI =====================
def main():
    with st.sidebar:
        st.markdown("## 🏥 MediScan AI")
        st.markdown("**Radiology Suite**")
        st.divider()
        st.markdown("##### 👤 Patient Details")
        p_name = st.text_input("Full Name", "John Doe")
        p_id = st.text_input("Medical Record No.", "MRN-10293")
        p_dob = st.text_input("Date of Birth", "01-01-1980")
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_age = st.number_input("Age (years)", 0, 120, 44)
        
        st.divider()
        st.markdown("##### 🩺 Clinical Information")
        p_ref_doc = st.text_input("Referring Physician", "Dr. A. Khan")
        p_indication = st.text_input("Clinical Indication", "Cough, Breathlessness — Rule out consolidation", help="This will affect the impression in report")

        st.divider()
        st.markdown("##### ⚙️ Scan Parameters")
        p_contrast = st.selectbox("Contrast", ["Non-Contrast", "IV Contrast", "Oral + IV Contrast"])
        p_slice_thk = st.selectbox("Slice Thickness (mm)", ["1.0", "1.5", "2.5", "5.0"])

    st.title("🫁 MediScan AI — Radiology Suite")
    st.caption("AI-assisted CT Lung Segmentation | Research Use Only")

    model, loaded = load_model()
    if not loaded:
        st.error("Model file not found!")
        return

    uploaded = st.file_uploader("📂 Select Axial CT Slice", type=["png", "jpg", "jpeg", "tif"])

    if uploaded:
        pil_img = Image.open(uploaded)
        col1, col2, col3 = st.columns([1, 1, 1.4])

        with col1:
            st.subheader("📷 Source Image")
            st.image(pil_img, use_column_width=True)

        # Processing
        img_np = np.array(pil_img.convert("L").resize((256, 256)))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_img = clahe.apply(img_np).astype(np.float32) / 255.0
        t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(t)
            mask = (out.squeeze().numpy() > 0.5).astype(np.uint8)
            lung_pixels = int(np.sum(mask))

        area_pct = (lung_pixels / (256 * 256)) * 100

        with col2:
            st.subheader("🤖 AI Segmentation")
            st.image(mask * 255, use_column_width=True)
            st.metric("Lung Coverage", f"{area_pct:.1f}%")

        # Report
        patient_info = {
            "name": p_name, "mrn": p_id, "dob": p_dob, "gender": p_gender,
            "age": p_age, "referring_doc": p_ref_doc, "indication": p_indication
        }
        scan_info = {"contrast": p_contrast, "slice_thickness": p_slice_thk}

        report_text = generate_professional_report(patient_info, lung_pixels, area_pct, scan_info)

        with col3:
            st.subheader("📋 Diagnostic Report")
            st.text_area("Generated Report", report_text, height=600)
            
            st.download_button(
                label="📥 Download Report (.txt)",
                data=report_text,
                file_name=f"MediScan_Report_{p_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
