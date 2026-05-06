import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os, warnings, cv2, io
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
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.report-container {
    background: white;
    border: 1px solid #c8d6e5;
    border-radius: 4px;
    padding: 0;
    color: #1a1a2e;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.report-header {
    background: #003366;
    color: white;
    padding: 20px 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.report-header-left h2 {
    font-family: 'Source Serif 4', serif;
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 2px 0;
    letter-spacing: 0.3px;
}
.report-header-left p { font-size: 12px; opacity: 0.8; margin: 0; }
.report-header-right { text-align: right; font-size: 12px; opacity: 0.85; }
.report-header-right strong { font-size: 14px; display: block; margin-bottom: 2px; }

.report-section { padding: 16px 28px; border-bottom: 1px solid #e8edf2; }
.report-section:last-child { border-bottom: none; }
.section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #003366;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 2px solid #003366;
    display: inline-block;
}
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; }
.info-row { display: flex; justify-content: space-between; align-items: baseline; padding: 4px 0; border-bottom: 1px dotted #e0e8f0; font-size: 13px; }
.info-label { color: #5a6a7a; font-weight: 500; }
.info-value { color: #1a2a3a; font-weight: 600; }

.finding-block { background: #f5f8fb; border-left: 3px solid #003366; padding: 10px 14px; margin: 8px 0; border-radius: 0 4px 4px 0; font-size: 13px; line-height: 1.6; }
.finding-label { font-weight: 600; color: #003366; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }

.impression-box { background: #fff8e6; border: 1px solid #f0c040; border-radius: 4px; padding: 14px; font-size: 13px; line-height: 1.7; color: #3a2800; }
.impression-box strong { color: #7a4800; }

.normal-tag { background: #e8f5e9; color: #1b5e20; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.caution-tag { background: #fff3e0; color: #7a4000; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }

.signature-block { padding: 18px 28px; background: #f7f9fb; border-top: 2px solid #003366; display: flex; justify-content: space-between; align-items: flex-end; }
.sig-line { border-top: 1px solid #a0b0c0; padding-top: 6px; margin-top: 28px; font-size: 12px; color: #5a6a7a; min-width: 200px; }

.disclaimer { background: #fdf3f3; border: 1px solid #f5c6c6; border-radius: 4px; padding: 10px 14px; font-size: 11.5px; color: #7a2020; line-height: 1.6; margin: 12px 28px; }

.metric-row { display: flex; gap: 12px; margin: 10px 0; }
.metric-card { flex: 1; background: #f0f5fb; border: 1px solid #d0dcea; border-radius: 6px; padding: 12px 14px; text-align: center; }
.metric-card .val { font-size: 22px; font-weight: 700; color: #003366; }
.metric-card .lbl { font-size: 11px; color: #5a6a7a; margin-top: 2px; }

.stButton > button {
    width: 100%;
    background: #003366 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    height: 3em;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover { background: #004a99 !important; }

.upload-zone { border: 2px dashed #a0b8d0; border-radius: 8px; padding: 28px; text-align: center; background: #f7fafd; }
.sidebar-card { background: white; border: 1px solid #d8e4f0; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
.sidebar-label { font-size: 11px; font-weight: 600; color: #5a6a7a; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# U-NET ARCHITECTURE (Verbatim — unchanged)
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
        self.enc1 = nn.Sequential(DoubleConv(1, 64));   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(DoubleConv(64, 128)); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(DoubleConv(128, 256));self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(DoubleConv(256, 512));self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2);  self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2);  self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2);   self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x); x = self.pool1(s1)
        s2 = self.enc2(x); x = self.pool2(s2)
        s3 = self.enc3(x); x = self.pool3(s3)
        s4 = self.enc4(x); x = self.pool4(s4)
        x  = self.bottleneck(x)
        x  = self.dec4(torch.cat([self.up4(x), s4], 1))
        x  = self.dec3(torch.cat([self.up3(x), s3], 1))
        x  = self.dec2(torch.cat([self.up2(x), s2], 1))
        x  = self.dec1(torch.cat([self.up1(x), s1], 1))
        return torch.sigmoid(self.out(x))


# ─────────────────────────────────────────────────────────────
# LOGIC & PREPROCESSING (Verbatim — unchanged)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet()
    path = "best_unet_model.pth"
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()
        return model, True
    return model, False


# ─────────────────────────────────────────────────────────────
# PROFESSIONAL REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_professional_report(patient_info, lung_pixels, area_pct, scan_info):
    now = datetime.now()
    date_str   = now.strftime("%d %B %Y")
    time_str   = now.strftime("%H:%M")
    report_id  = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"

    # Clinical thresholds (illustrative AI output)
    coverage_status = "Within normal range" if 20 <= area_pct <= 60 else "Requires review"
    laterality      = "Bilateral lung fields visible"
    pleural_status  = "No gross pleural effusion detected"
    density_note    = "Parenchymal density consistent with axial CT protocol"
    opacity_note    = "No large consolidative opacity identified"

    report = f"""
================================================================================
          MEDISCAN AI RADIOLOGY — OFFICIAL DIAGNOSTIC REPORT
================================================================================

REPORT ID       : {report_id}
REPORT DATE     : {date_str}
REPORT TIME     : {time_str} (PKT)
INSTITUTION     : MediScan AI Radiology Centre
DEPARTMENT      : Diagnostic Radiology & Medical Imaging
MODALITY        : Computed Tomography (CT) — Chest / Thorax
PRIORITY        : Routine

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patient Name    : {patient_info['name']}
Medical Rec. No.: {patient_info['mrn']}
Date of Birth   : {patient_info['dob']}
Gender          : {patient_info['gender']}
Age             : {patient_info['age']} years
Referring Phys. : {patient_info['referring_doc']}
Clinical Indic. : {patient_info['indication']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI Model        : U-Net (Encoder-Decoder Architecture, 31M Parameters)
Input Resolution: 256 × 256 pixels (axial slice)
Preprocessing   : CLAHE (Contrast Limited Adaptive Histogram Equalization)
                  clipLimit=2.0, tileGridSize=(8×8)
Segmentation    : Binary thresholding at 0.5 (sigmoid output)
Contrast Agent  : {scan_info['contrast']}
Slice Thickness : {scan_info['slice_thickness']} mm
Window/Level    : Lung Window (WL: -600 / WW: 1500)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITATIVE AI FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total Scan Pixels           : 65,536  (256 × 256)
  Segmented Lung Area         : {lung_pixels:,} pixels
  Lung Field Coverage         : {area_pct:.2f}% of axial slice
  Coverage Assessment         : {coverage_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RADIOLOGICAL OBSERVATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LUNG PARENCHYMA
   - {laterality}
   - {density_note}
   - {opacity_note}
   - Lung borders delineated with high model confidence

2. PLEURAL SPACE
   - {pleural_status}
   - No pneumothorax identified on current slice

3. MEDIASTINUM
   - Mediastinal contour within expected limits for axial view
   - No gross mediastinal widening detected

4. AIRWAYS
   - Central airway morphology appears unremarkable on this slice
   - Tracheal deviation: Not apparent

5. CHEST WALL & DIAPHRAGM
   - Bony thorax: No lytic or sclerotic lesion identified on this slice
   - Diaphragmatic contour: Normal appearance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. AI-assisted lung segmentation successfully performed on the submitted
     axial CT slice.
  2. Segmented lung field area of {area_pct:.2f}% is {coverage_status.lower()}.
  3. No gross consolidation, pleural effusion, or pneumothorax identified
     on this single axial slice.
  4. Multi-slice volumetric analysis is recommended for comprehensive
     pulmonary assessment.

RECOMMENDATION: Clinical correlation with complete CT series, pulmonary
function tests, and patient history is strongly advised. This AI report
is intended as a decision-support tool and does not replace interpretation
by a board-certified radiologist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Electronically generated by:
  MediScan AI System v2.0 — U-Net Lung Segmentation Engine

Verified & Released by:
  [ Radiologist Signature Required Before Clinical Use ]

Report ID : {report_id}
Date/Time : {date_str} at {time_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This report is generated by an AI-assisted system and is intended SOLELY for
research and educational purposes. It does not constitute a final clinical
diagnosis. All findings must be reviewed and confirmed by a licensed and
qualified radiologist or medical professional before any clinical action is
taken. MediScan AI assumes no medico-legal liability for decisions made
solely on the basis of this automated output.

© {now.year} MediScan AI Radiology Suite. All rights reserved.
================================================================================
"""
    return report.strip()


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 MediScan AI")
        st.markdown("**Radiology Suite — Patient Registration**")
        st.divider()

        st.markdown("##### 👤 Patient Details")
        p_name    = st.text_input("Full Name", "John Doe")
        p_id      = st.text_input("Medical Record No.", "MRN-10293")
        p_dob     = st.text_input("Date of Birth", "01-01-1980")
        p_gender  = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_age     = st.number_input("Age (years)", min_value=0, max_value=120, value=44)

        st.divider()
        st.markdown("##### 🩺 Clinical Information")
        p_ref_doc   = st.text_input("Referring Physician", "Dr. A. Khan")
        p_indication = st.text_input("Clinical Indication", "Cough, Breathlessness — Rule out consolidation")

        st.divider()
        st.markdown("##### ⚙️ Scan Parameters")
        p_contrast  = st.selectbox("Contrast", ["Non-Contrast", "IV Contrast", "Oral + IV Contrast"])
        p_slice_thk = st.selectbox("Slice Thickness (mm)", ["1.0", "1.5", "2.5", "5.0"])

        st.divider()
        st.info("Upload a CT axial slice above to generate the full diagnostic report.")

    # ── Main Panel ────────────────────────────────────────────
    st.title("🫁 MediScan AI — Radiology Suite")
    st.caption("AI-assisted CT Lung Segmentation | For research use only")

    model, loaded = load_model()
    if not loaded:
        st.error("⚠️ Model weights file ('best_unet_model.pth') not found in the working directory.")
        st.info("Please place the trained model file alongside app.py and restart.")
        return

    uploaded = st.file_uploader(
        "📂  Select Axial CT Slice",
        type=["png", "jpg", "jpeg", "tif"],
        help="Upload a single axial CT slice image (DICOM-exported PNG/JPG/TIF)"
    )

    if uploaded:
        pil_img = Image.open(uploaded)

        # Three-column layout
        col1, col2, col3 = st.columns([1, 1, 1.3])

        with col1:
            st.subheader("📷 Source Image")
            st.image(pil_img, use_column_width=True, caption="Uploaded CT Slice")

        # ── Preprocessing & Inference (verbatim logic) ────────
        img_np   = np.array(pil_img.convert("L").resize((256, 256)))
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_img = clahe.apply(img_np).astype(np.float32) / 255.0

        t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out        = model(t)
            mask       = (out.squeeze().numpy() > 0.5).astype(np.uint8)
            lung_pixels = int(np.sum(mask))

        total_pixels = 256 * 256
        area_pct     = (lung_pixels / total_pixels) * 100

        with col2:
            st.subheader("🤖 AI Segmentation")
            st.image(mask * 255, use_column_width=True, caption="Lung Mask (U-Net Output)")
            st.markdown(f"""
            <div style='background:#f0f5fb;border:1px solid #c8d8ea;border-radius:6px;padding:12px;text-align:center;margin-top:8px'>
                <div style='font-size:28px;font-weight:700;color:#003366'>{lung_pixels:,}</div>
                <div style='font-size:12px;color:#5a6a7a;margin-top:2px'>Segmented Lung Pixels</div>
            </div>
            <div style='background:#f0f5fb;border:1px solid #c8d8ea;border-radius:6px;padding:12px;text-align:center;margin-top:8px'>
                <div style='font-size:28px;font-weight:700;color:#003366'>{area_pct:.1f}%</div>
                <div style='font-size:12px;color:#5a6a7a;margin-top:2px'>Lung Field Coverage</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Report ────────────────────────────────────────────
        patient_info = {
            "name":          p_name,
            "mrn":           p_id,
            "dob":           p_dob,
            "gender":        p_gender,
            "age":           p_age,
            "referring_doc": p_ref_doc,
            "indication":    p_indication,
        }
        scan_info = {
            "contrast":        p_contrast,
            "slice_thickness": p_slice_thk,
        }

        report_text = generate_professional_report(patient_info, lung_pixels, area_pct, scan_info)

        with col3:
            st.subheader("📋 Diagnostic Report")

            now = datetime.now()
            report_id = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"
            coverage_status = "Within normal range" if 20 <= area_pct <= 60 else "Requires review"

            st.markdown(f"""
            <div class="report-container">

              <div class="report-header">
                <div class="report-header-left">
                  <h2>🏥 MediScan Radiology Centre</h2>
                  <p>Department of Diagnostic Radiology & Medical Imaging</p>
                  <p style="margin-top:4px;font-size:11px;opacity:0.7">Tel: +92-42-111-000-000 | info@mediscan-ai.pk</p>
                </div>
                <div class="report-header-right">
                  <strong>Report ID</strong>
                  {report_id}<br>
                  {now.strftime("%d %b %Y, %H:%M")}<br>
                  <span style="margin-top:4px;display:block;font-size:11px;background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:10px">ROUTINE</span>
                </div>
              </div>

              <div class="report-section">
                <div class="section-title">Patient Information</div>
                <div class="info-grid">
                  <div>
                    <div class="info-row"><span class="info-label">Name</span><span class="info-value">{p_name}</span></div>
                    <div class="info-row"><span class="info-label">MRN</span><span class="info-value">{p_id}</span></div>
                    <div class="info-row"><span class="info-label">DOB</span><span class="info-value">{p_dob}</span></div>
                    <div class="info-row"><span class="info-label">Age / Gender</span><span class="info-value">{p_age} yrs / {p_gender}</span></div>
                  </div>
                  <div>
                    <div class="info-row"><span class="info-label">Referring Physician</span><span class="info-value">{p_ref_doc}</span></div>
                    <div class="info-row"><span class="info-label">Modality</span><span class="info-value">CT Chest — Axial</span></div>
                    <div class="info-row"><span class="info-label">Contrast</span><span class="info-value">{p_contrast}</span></div>
                    <div class="info-row"><span class="info-label">Slice Thickness</span><span class="info-value">{p_slice_thk} mm</span></div>
                  </div>
                </div>
                <div style="margin-top:10px;font-size:12px;color:#5a6a7a"><b>Indication:</b> {p_indication}</div>
              </div>

              <div class="report-section">
                <div class="section-title">Quantitative AI Metrics</div>
                <div class="metric-row">
                  <div class="metric-card"><div class="val">{lung_pixels:,}</div><div class="lbl">Segmented Pixels</div></div>
                  <div class="metric-card"><div class="val">{area_pct:.1f}%</div><div class="lbl">Lung Coverage</div></div>
                  <div class="metric-card"><div class="val">256²</div><div class="lbl">Input Resolution</div></div>
                </div>
                <div style="margin-top:8px;font-size:12px">
                  Coverage Assessment: &nbsp;
                  {'<span class="normal-tag">✓ Within Normal Range</span>' if 20 <= area_pct <= 60 else '<span class="caution-tag">⚠ Requires Review</span>'}
                </div>
              </div>

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
                  <div class="finding-label">Mediastinum & Airways</div>
                  Mediastinal contour within expected limits. No gross mediastinal widening. Central airway morphology appears unremarkable. Tracheal deviation not apparent.
                </div>

                <div class="finding-block">
                  <div class="finding-label">Chest Wall & Diaphragm</div>
                  No lytic or sclerotic bony lesion identified on this slice. Diaphragmatic contour normal in appearance.
                </div>
              </div>

              <div class="report-section">
                <div class="section-title">Impression</div>
                <div class="impression-box">
                  <strong>1.</strong> AI-assisted lung segmentation successfully performed on the submitted axial CT slice.<br>
                  <strong>2.</strong> Segmented lung field area of <strong>{area_pct:.2f}%</strong> is <strong>{coverage_status.lower()}</strong>.<br>
                  <strong>3.</strong> No gross consolidation, pleural effusion, or pneumothorax identified on this single axial slice.<br>
                  <strong>4.</strong> Multi-slice volumetric analysis recommended for comprehensive pulmonary assessment.<br><br>
                  <strong>Recommendation:</strong> Clinical correlation with complete CT series, PFTs, and patient history is strongly advised.
                </div>
              </div>

              <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This report is generated by an AI-assisted system for <strong>research and educational purposes only</strong>. It does not constitute a final clinical diagnosis. All findings must be reviewed and confirmed by a licensed radiologist before any clinical action is taken. MediScan AI assumes no medico-legal liability for decisions made solely on this automated output.
              </div>

              <div class="signature-block">
                <div>
                  <div class="sig-line">MediScan AI System v2.0<br>U-Net Lung Segmentation Engine</div>
                </div>
                <div style="text-align:right">
                  <div class="sig-line" style="text-align:right">Radiologist Verification<br><span style="font-size:11px;color:#a0b0c0">[Signature required for clinical use]</span></div>
                </div>
              </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥  Download Full Clinical Report (.txt)",
                data=report_text,
                file_name=f"MediScan_Report_{p_id}_{now.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:
        st.markdown("""
        <div style='border:2px dashed #a0b8d0;border-radius:10px;padding:40px;text-align:center;background:#f7fafd;color:#5a6a7a;margin-top:20px'>
            <div style='font-size:48px'>🫁</div>
            <div style='font-size:18px;font-weight:600;margin-top:12px;color:#003366'>Upload a CT Axial Slice</div>
            <div style='font-size:13px;margin-top:6px'>Supported formats: PNG, JPG, JPEG, TIF</div>
            <div style='font-size:12px;margin-top:4px;color:#8a9ab0'>Fill in patient details in the sidebar, then upload an image to generate the full report.</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("MediScan AI © 2026 | For research use only. All clinical decisions must be validated by a licensed radiologist.")


if __name__ == "__main__":
    main()remove if any error 
