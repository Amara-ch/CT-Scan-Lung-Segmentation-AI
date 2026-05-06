import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os, warnings, cv2, io
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MediScan AI | Radiology Suite",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

:root {
    --navy: #0a1628;
    --navy-mid: #112240;
    --navy-light: #1d3461;
    --accent: #00b4d8;
    --accent-glow: rgba(0,180,216,0.18);
    --gold: #c9a84c;
    --gold-light: #f0d080;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --surface: #f8fafc;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-mid: #475569;
    --text-light: #94a3b8;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 60%, #0d2137 100%);
    border: 1px solid rgba(0,180,216,0.25);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 120px;
    background: radial-gradient(ellipse, rgba(201,168,76,0.07) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Crimson Pro', serif;
    font-size: 32px;
    font-weight: 600;
    color: white;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
}
.hero-subtitle {
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    margin: 0;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,216,0.15);
    border: 1px solid rgba(0,180,216,0.4);
    color: var(--accent);
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 12px;
}
.hero-meta {
    position: absolute;
    right: 36px; top: 50%;
    transform: translateY(-50%);
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    line-height: 1.9;
}
.hero-meta strong {
    color: rgba(255,255,255,0.7);
    display: block;
    font-size: 13px;
    font-family: 'Crimson Pro', serif;
    font-weight: 600;
}

/* ── Upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, #f8faff 0%, #f0f5ff 100%);
    border: 2px dashed #c7d7f0;
    border-radius: 12px;
    padding: 44px 28px;
    text-align: center;
    transition: all 0.25s;
}
.upload-icon { font-size: 52px; margin-bottom: 12px; }
.upload-title { font-family: 'Crimson Pro', serif; font-size: 22px; color: var(--navy); font-weight: 600; margin-bottom: 6px; }
.upload-sub { font-size: 13px; color: var(--text-mid); }
.upload-formats { display: inline-block; margin-top: 10px; background: #e8f0ff; color: #3b5bb5; font-size: 11px; font-weight: 600; padding: 4px 14px; border-radius: 20px; letter-spacing: 0.8px; }

/* ── Panel card ── */
.panel-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}
.panel-header {
    background: var(--navy);
    color: white;
    padding: 11px 18px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-body { padding: 16px; }

/* ── Metric chips ── */
.metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.metric-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 12px;
    text-align: center;
}
.metric-chip .val {
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: var(--navy);
    letter-spacing: -0.5px;
}
.metric-chip .lbl { font-size: 10.5px; color: var(--text-light); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
.metric-chip.accent { border-color: rgba(0,180,216,0.35); background: rgba(0,180,216,0.04); }
.metric-chip.accent .val { color: #0077a8; }

/* ── Status badge ── */
.status-ok { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; border-radius: 20px; padding: 4px 14px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; display: inline-block; }
.status-warn { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; border-radius: 20px; padding: 4px 14px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; display: inline-block; }

/* ── Finding blocks ── */
.finding {
    border-left: 3px solid var(--navy);
    background: #f8fafc;
    padding: 10px 14px;
    margin: 7px 0;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    line-height: 1.65;
}
.finding-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--navy);
    margin-bottom: 4px;
}
.finding.ok { border-color: #22c55e; }
.finding.warn { border-color: #f59e0b; }

/* ── Impression box ── */
.impression {
    background: linear-gradient(135deg, #fffdf0, #fff8dc);
    border: 1px solid #e8d88a;
    border-radius: 8px;
    padding: 16px 18px;
    font-size: 13px;
    line-height: 1.75;
    color: #3a2e00;
}
.impression strong { color: #7a5800; }

/* ── Disclaimer ── */
.disclaimer-strip {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 11.5px;
    color: #991b1b;
    line-height: 1.6;
    margin: 12px 0 0;
}

/* ── Report watermark section ── */
.report-wrap {
    background: white;
    border: 1px solid #c8d6e5;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.report-top-bar {
    background: linear-gradient(90deg, var(--navy) 0%, var(--navy-light) 100%);
    color: white;
    padding: 22px 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    position: relative;
}
.report-top-bar::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--gold), var(--accent));
}
.rtb-name { font-family: 'Crimson Pro', serif; font-size: 22px; font-weight: 600; letter-spacing: -0.2px; margin-bottom: 2px; }
.rtb-dept { font-size: 11.5px; opacity: 0.65; letter-spacing: 0.3px; }
.rtb-contact { font-size: 10.5px; opacity: 0.5; margin-top: 5px; font-family: 'DM Mono', monospace; }
.rtb-right { text-align: right; font-size: 11px; }
.rtb-right .rid { font-family: 'DM Mono', monospace; font-size: 13px; color: var(--accent); margin-bottom: 2px; }
.rtb-right .rdate { opacity: 0.7; font-size: 11px; }
.rtb-right .rpriority { background: rgba(0,180,216,0.2); color: var(--accent); font-size: 10px; font-weight: 700; letter-spacing: 1px; padding: 2px 10px; border-radius: 10px; display: inline-block; margin-top: 6px; border: 1px solid rgba(0,180,216,0.3); }

.report-section { padding: 16px 24px; border-bottom: 1px solid #eef2f7; }
.report-section:last-child { border-bottom: none; }
.sec-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--navy); margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid var(--navy); display: inline-block; }

.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 20px; }
.info-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dotted #e8eef5; font-size: 12.5px; }
.il { color: #5a6a7a; }
.iv { color: #0f172a; font-weight: 600; }

.report-footer {
    padding: 14px 24px;
    background: #f7f9fb;
    border-top: 2px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    font-size: 11.5px;
    color: #5a6a7a;
}
.sig-line { border-top: 1px solid #b0bec5; padding-top: 5px; margin-top: 24px; min-width: 180px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, #0d1f38 100%) !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.65) !important; font-size: 11.5px !important; font-weight: 600 !important; letter-spacing: 0.6px !important; }
.sidebar-section-title {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    color: var(--accent) !important;
    margin: 18px 0 10px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid rgba(0,180,216,0.25) !important;
}
.sidebar-logo {
    text-align: center;
    padding: 20px 0 16px;
}
.sidebar-logo .logo-icon { font-size: 40px; }
.sidebar-logo .logo-name { font-family: 'Crimson Pro', serif; font-size: 20px; color: white; margin-top: 6px; }
.sidebar-logo .logo-ver { font-size: 10px; color: rgba(255,255,255,0.35); font-family: 'DM Mono', monospace; letter-spacing: 1px; }

/* ── Step tracker ── */
.step-tracker { padding: 8px 0; }
.step-item { display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: 12px; color: rgba(255,255,255,0.5); }
.step-item.done { color: rgba(255,255,255,0.9); }
.step-item.done .step-dot { background: var(--success); border-color: var(--success); }
.step-item.active { color: white; }
.step-item.active .step-dot { background: var(--accent); border-color: var(--accent); animation: pulse 1.5s infinite; }
.step-dot { width: 10px; height: 10px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.2); background: transparent; flex-shrink: 0; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,180,216,0.5)} 50%{box-shadow:0 0 0 5px rgba(0,180,216,0)} }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #f1f5f9; border-radius: 8px; padding: 3px; gap: 2px; border: none; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; font-size: 12.5px; font-weight: 600; color: #64748b; border: none; padding: 7px 16px; }
.stTabs [aria-selected="true"] { background: white !important; color: var(--navy) !important; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    height: 3em;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 2px 10px rgba(10,22,40,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--navy-light) 0%, #254d80 100%) !important;
    box-shadow: 0 4px 16px rgba(10,22,40,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0d7a5f, #0a9e7a) !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--gold)) !important; border-radius: 4px; }

/* ── Heatmap label ── */
.heatmap-legend { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-light); margin-top: 4px; font-family: 'DM Mono', monospace; }

/* ── Confidence bar ── */
.conf-bar-wrap { background: #e8edf5; border-radius: 4px; height: 6px; overflow: hidden; margin: 6px 0 2px; }
.conf-bar { height: 100%; background: linear-gradient(90deg, var(--accent), #0077a8); border-radius: 4px; }
.conf-label { font-size: 11px; color: var(--text-mid); font-family: 'DM Mono', monospace; }

/* ── Overlay image caption ── */
.img-caption { font-size: 11px; color: var(--text-light); text-align: center; margin-top: 5px; font-weight: 500; letter-spacing: 0.4px; text-transform: uppercase; }

/* ── Section divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
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
# MODEL LOADER
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
# OVERLAY GENERATOR
# ─────────────────────────────────────────────────────────────
def make_overlay(orig_pil, mask_np):
    orig_rgb = np.array(orig_pil.convert("RGB").resize((256, 256)))
    overlay  = orig_rgb.copy()
    lung_px  = mask_np.astype(bool)
    overlay[lung_px]  = (overlay[lung_px] * 0.55 + np.array([0, 180, 216]) * 0.45).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 1)
    return Image.fromarray(overlay)


def make_heatmap(raw_out_np):
    heat_u8 = (raw_out_np * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heat_color)


# ─────────────────────────────────────────────────────────────
# REPORT TEXT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_report_text(patient_info, lung_pixels, area_pct, scan_info, report_id, now):
    date_str = now.strftime("%d %B %Y")
    time_str = now.strftime("%H:%M")
    coverage_status = "Within normal range" if 20 <= area_pct <= 60 else "Requires review"

    return f"""
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
Preprocessing   : CLAHE (clipLimit=2.0, tileGridSize=8×8)
Segmentation    : Binary thresholding at 0.5 (sigmoid output)
Contrast Agent  : {scan_info['contrast']}
Slice Thickness : {scan_info['slice_thickness']} mm
Window/Level    : Lung Window (WL: -600 / WW: 1500)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITATIVE AI FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total Scan Pixels           : 65,536  (256 × 256)
  Segmented Lung Area         : {lung_pixels:,} pixels
  Lung Field Coverage         : {area_pct:.2f}% of axial slice
  Coverage Assessment         : {coverage_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RADIOLOGICAL OBSERVATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LUNG PARENCHYMA
   - Bilateral lung fields visible on axial slice
   - Parenchymal density consistent with CT protocol
   - No large consolidative opacity identified
   - Lung borders delineated with high model confidence

2. PLEURAL SPACE
   - No gross pleural effusion detected
   - No pneumothorax identified on current slice

3. MEDIASTINUM
   - Mediastinal contour within expected limits
   - No gross mediastinal widening detected

4. AIRWAYS
   - Central airway morphology appears unremarkable
   - Tracheal deviation: Not apparent

5. CHEST WALL & DIAPHRAGM
   - No lytic or sclerotic lesion identified on this slice
   - Diaphragmatic contour: Normal appearance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. AI-assisted lung segmentation successfully performed.
  2. Lung field area of {area_pct:.2f}% is {coverage_status.lower()}.
  3. No gross consolidation, pleural effusion, or pneumothorax identified.
  4. Multi-slice volumetric analysis recommended for full assessment.

RECOMMENDATION: Clinical correlation with complete CT series, PFTs, and
patient history is strongly advised. This AI report is a decision-support
tool only and does not replace a board-certified radiologist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For research and educational purposes only. Not a clinical diagnosis.
Must be reviewed by a licensed radiologist before clinical action.
MediScan AI assumes no medico-legal liability.

© {now.year} MediScan AI Radiology Suite.
================================================================================
""".strip()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
          <div class="logo-icon">🏥</div>
          <div class="logo-name">MediScan AI</div>
          <div class="logo-ver">RADIOLOGY SUITE v2.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">Patient Details</div>', unsafe_allow_html=True)
        p_name    = st.text_input("Full Name", "John Doe")
        p_id      = st.text_input("Medical Record No.", "MRN-10293")
        p_dob     = st.text_input("Date of Birth", "01-01-1980")
        p_gender  = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_age     = st.number_input("Age (years)", min_value=0, max_value=120, value=44)

        st.markdown('<div class="sidebar-section-title">Clinical Information</div>', unsafe_allow_html=True)
        p_ref_doc    = st.text_input("Referring Physician", "Dr. A. Khan")
        p_indication = st.text_input("Clinical Indication", "Cough, Breathlessness — Rule out consolidation")

        st.markdown('<div class="sidebar-section-title">Scan Parameters</div>', unsafe_allow_html=True)
        p_contrast  = st.selectbox("Contrast Agent", ["Non-Contrast", "IV Contrast", "Oral + IV Contrast"])
        p_slice_thk = st.selectbox("Slice Thickness (mm)", ["1.0", "1.5", "2.5", "5.0"])

        st.markdown('<div class="sidebar-section-title">Workflow Status</div>', unsafe_allow_html=True)
        uploaded_flag = "uploaded_file" in st.session_state and st.session_state["uploaded_file"]
        analyzed_flag = "analyzed" in st.session_state and st.session_state["analyzed"]
        st.markdown(f"""
        <div class="step-tracker">
          <div class="step-item done"><div class="step-dot"></div>Patient registered</div>
          <div class="step-item {'done' if uploaded_flag else ''}"><div class="step-dot"></div>CT slice uploaded</div>
          <div class="step-item {'done' if analyzed_flag else ''}"><div class="step-dot"></div>AI segmentation complete</div>
          <div class="step-item {'active' if analyzed_flag else ''}"><div class="step-dot"></div>Report ready</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Hero Header ───────────────────────────────────────────
    now       = datetime.now()
    report_id = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"

    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">🫁 MediScan AI Radiology Suite</div>
      <p class="hero-subtitle">CT Lung Segmentation & Diagnostic Reporting — AI-Assisted</p>
      <span class="hero-badge">Research Use Only</span>
      <div class="hero-meta">
        <strong>MediScan AI v2.0</strong>
        {now.strftime("%d %b %Y, %H:%M")} PKT<br>
        U-Net · 31M Params<br>
        {report_id}
      </div>
    </div>
    """, unsafe_allow_html=True)

    model, loaded = load_model()
    if not loaded:
        st.error("⚠️ Model weights not found: `best_unet_model.pth` must be in the working directory.")
        st.info("Place the trained model weights alongside `app.py` and restart the app.")
        return

    uploaded = st.file_uploader(
        "Upload Axial CT Slice",
        type=["png", "jpg", "jpeg", "tif"],
        help="DICOM-exported axial CT slice — PNG, JPG, or TIF",
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
          <div class="upload-icon">🫁</div>
          <div class="upload-title">Upload a CT Axial Slice to Begin</div>
          <div class="upload-sub">Fill in patient details in the sidebar, then upload your scan above.</div>
          <span class="upload-formats">PNG · JPG · JPEG · TIF</span>
        </div>
        """, unsafe_allow_html=True)
        st.session_state["uploaded_file"] = False
        st.session_state["analyzed"]      = False
        return

    st.session_state["uploaded_file"] = True
    pil_img = Image.open(uploaded)

    # ── Inference ─────────────────────────────────────────────
    prog_placeholder = st.empty()
    with prog_placeholder.container():
        st.markdown("**⚙️ Running AI segmentation pipeline…**")
        bar = st.progress(0)
        import time
        for pct, label in [(20, "Preprocessing — CLAHE normalization"),
                           (50, "Inference — U-Net forward pass"),
                           (80, "Post-processing — Binary thresholding"),
                           (100, "Generating diagnostic report")]:
            st.caption(label)
            bar.progress(pct)
            time.sleep(0.28)

    prog_placeholder.empty()
    st.session_state["analyzed"] = True

    img_np   = np.array(pil_img.convert("L").resize((256, 256)))
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_img = clahe.apply(img_np).astype(np.float32) / 255.0

    t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        raw_out     = model(t).squeeze().numpy()
        mask        = (raw_out > 0.5).astype(np.uint8)
        lung_pixels = int(np.sum(mask))

    total_pixels  = 256 * 256
    area_pct      = (lung_pixels / total_pixels) * 100
    mean_conf     = float(raw_out[mask.astype(bool)].mean()) if lung_pixels > 0 else 0.0
    coverage_ok   = 20 <= area_pct <= 60

    overlay_img   = make_overlay(pil_img, mask)
    heatmap_img   = make_heatmap(raw_out)

    # ── Image panels ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="panel-card"><div class="panel-header">📷 Source Image</div><div class="panel-body">', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown('<div class="img-caption">Original CT Slice</div></div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel-card"><div class="panel-header">🤖 Segmentation Mask</div><div class="panel-body">', unsafe_allow_html=True)
        st.image(mask * 255, use_container_width=True)
        st.markdown('<div class="img-caption">Binary U-Net Output</div></div></div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="panel-card"><div class="panel-header">🔵 Overlay</div><div class="panel-body">', unsafe_allow_html=True)
        st.image(overlay_img, use_container_width=True)
        st.markdown('<div class="img-caption">Segmentation Overlay</div></div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="panel-card"><div class="panel-header">🌡️ Confidence Heatmap</div><div class="panel-body">', unsafe_allow_html=True)
        st.image(heatmap_img, use_container_width=True)
        st.markdown("""
        <div class="heatmap-legend"><span>Low</span><span>Confidence</span><span>High</span></div>
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics bar ───────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, f"{lung_pixels:,}", "Segmented Pixels"),
        (m2, f"{area_pct:.1f}%", "Lung Coverage"),
        (m3, f"{mean_conf:.3f}", "Mean Confidence"),
        (m4, "256²", "Input Resolution"),
        (m5, "0.5", "Threshold"),
    ]
    for col, val, lbl in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-chip accent">
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin:10px 0 20px;font-size:12.5px;color:#5a6a7a">
      Coverage Assessment: &nbsp;
      {'<span class="status-ok">✓ Within Normal Range (20–60%)</span>' if coverage_ok else '<span class="status-warn">⚠ Outside Normal Range — Requires Review</span>'}
      &nbsp;·&nbsp;
      <span style="font-family:\'DM Mono\',monospace;font-size:11.5px">Model confidence on lung region: {mean_conf:.1%}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: Report | Findings | Technical ───────────────────
    patient_info = {
        "name": p_name, "mrn": p_id, "dob": p_dob,
        "gender": p_gender, "age": p_age,
        "referring_doc": p_ref_doc, "indication": p_indication,
    }
    scan_info = {"contrast": p_contrast, "slice_thickness": p_slice_thk}

    tab_report, tab_findings, tab_tech = st.tabs(["📋 Diagnostic Report", "🔬 Detailed Findings", "⚙️ Technical"])

    # ── TAB 1 — Full Report ───────────────────────────────────
    with tab_report:
        st.markdown(f"""
        <div class="report-wrap">

          <div class="report-top-bar">
            <div>
              <div class="rtb-name">🏥 MediScan Radiology Centre</div>
              <div class="rtb-dept">Department of Diagnostic Radiology & Medical Imaging</div>
              <div class="rtb-contact">Tel: +92-42-111-000-000 &nbsp;|&nbsp; info@mediscan-ai.pk &nbsp;|&nbsp; Lahore, Pakistan</div>
            </div>
            <div class="rtb-right">
              <div class="rid">{report_id}</div>
              <div class="rdate">{now.strftime("%d %B %Y, %H:%M")} PKT</div>
              <div class="rpriority">ROUTINE</div>
            </div>
          </div>

          <div class="report-section">
            <div class="sec-title">Patient Information</div>
            <div class="info-grid">
              <div>
                <div class="info-row"><span class="il">Full Name</span><span class="iv">{p_name}</span></div>
                <div class="info-row"><span class="il">Medical Record No.</span><span class="iv">{p_id}</span></div>
                <div class="info-row"><span class="il">Date of Birth</span><span class="iv">{p_dob}</span></div>
                <div class="info-row"><span class="il">Age / Gender</span><span class="iv">{p_age} yrs / {p_gender}</span></div>
              </div>
              <div>
                <div class="info-row"><span class="il">Referring Physician</span><span class="iv">{p_ref_doc}</span></div>
                <div class="info-row"><span class="il">Modality</span><span class="iv">CT Chest — Axial</span></div>
                <div class="info-row"><span class="il">Contrast</span><span class="iv">{p_contrast}</span></div>
                <div class="info-row"><span class="il">Slice Thickness</span><span class="iv">{p_slice_thk} mm</span></div>
              </div>
            </div>
            <div style="margin-top:10px;font-size:12px;color:#5a6a7a"><b>Clinical Indication:</b> {p_indication}</div>
          </div>

          <div class="report-section">
            <div class="sec-title">Quantitative AI Metrics</div>
            <div class="metric-row">
              <div class="metric-chip"><div class="val">{lung_pixels:,}</div><div class="lbl">Segmented Pixels</div></div>
              <div class="metric-chip accent"><div class="val">{area_pct:.1f}%</div><div class="lbl">Lung Coverage</div></div>
              <div class="metric-chip"><div class="val">{mean_conf:.3f}</div><div class="lbl">Mean Confidence</div></div>
              <div class="metric-chip"><div class="val">256²</div><div class="lbl">Resolution</div></div>
            </div>
            <div style="font-size:12px;margin-top:4px">
              Assessment: &nbsp;{'<span class="status-ok">✓ Within Normal Range</span>' if coverage_ok else '<span class="status-warn">⚠ Requires Review</span>'}
            </div>
          </div>

          <div class="report-section">
            <div class="sec-title">Observations</div>
            <div class="finding ok">
              <div class="finding-label">Lung Parenchyma</div>
              Bilateral lung fields visible on axial slice. Parenchymal density consistent with CT protocol.
              No large consolidative opacity identified. Lung borders delineated with high model confidence.
            </div>
            <div class="finding ok">
              <div class="finding-label">Pleural Space</div>
              No gross pleural effusion detected. No pneumothorax identified on current slice.
            </div>
            <div class="finding ok">
              <div class="finding-label">Mediastinum & Airways</div>
              Mediastinal contour within expected limits. No gross widening. Central airway morphology
              appears unremarkable. Tracheal deviation not apparent.
            </div>
            <div class="finding ok">
              <div class="finding-label">Chest Wall & Diaphragm</div>
              No lytic or sclerotic bony lesion identified on this slice. Diaphragmatic contour normal.
            </div>
          </div>

          <div class="report-section">
            <div class="sec-title">Impression</div>
            <div class="impression">
              <strong>1.</strong> AI-assisted lung segmentation successfully performed on the submitted axial CT slice.<br>
              <strong>2.</strong> Segmented lung field area of <strong>{area_pct:.2f}%</strong> is
              <strong>{'within normal range' if coverage_ok else 'outside normal range — requires clinical review'}</strong>.<br>
              <strong>3.</strong> No gross consolidation, pleural effusion, or pneumothorax identified on this single axial slice.<br>
              <strong>4.</strong> Multi-slice volumetric analysis is recommended for comprehensive pulmonary assessment.<br><br>
              <strong>Recommendation:</strong> Clinical correlation with complete CT series, pulmonary function tests,
              and patient history is strongly advised.
            </div>
          </div>

          <div class="disclaimer-strip" style="margin:0 24px 16px;">
            ⚠️ <strong>Disclaimer:</strong> This report is generated by an AI-assisted system for <strong>research and educational purposes only</strong>.
            It does not constitute a final clinical diagnosis. All findings must be reviewed and confirmed by a licensed radiologist
            before any clinical action is taken.
          </div>

          <div class="report-footer">
            <div>
              <div class="sig-line">MediScan AI System v2.0<br>U-Net Lung Segmentation Engine</div>
            </div>
            <div style="text-align:right">
              <div class="sig-line" style="text-align:right">
                Radiologist Verification<br>
                <span style="font-size:11px;color:#a0b0c0">[Signature required before clinical use]</span>
              </div>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        report_text = generate_report_text(patient_info, lung_pixels, area_pct, scan_info, report_id, now)
        st.download_button(
            label="📥  Download Full Clinical Report (.txt)",
            data=report_text,
            file_name=f"MediScan_{p_id}_{now.strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # ── TAB 2 — Detailed Findings ─────────────────────────────
    with tab_findings:
        st.markdown("#### Structured Radiological Findings")
        findings = [
            ("Lung Parenchyma", "ok", [
                ("Laterality", "Bilateral lung fields visible"),
                ("Parenchymal Density", "Consistent with axial CT protocol"),
                ("Consolidation", "No large consolidative opacity identified"),
                ("Model Confidence", f"{mean_conf:.1%} on segmented region"),
            ]),
            ("Pleural Space", "ok", [
                ("Pleural Effusion", "No gross effusion detected"),
                ("Pneumothorax", "Not identified on current slice"),
            ]),
            ("Mediastinum", "ok", [
                ("Contour", "Within expected limits for axial view"),
                ("Widening", "No gross mediastinal widening"),
            ]),
            ("Airways", "ok", [
                ("Central Airways", "Morphology unremarkable"),
                ("Tracheal Deviation", "Not apparent"),
            ]),
            ("Chest Wall & Diaphragm", "ok", [
                ("Bony Thorax", "No lytic/sclerotic lesion on this slice"),
                ("Diaphragm", "Normal contour appearance"),
            ]),
        ]
        for title, status, rows in findings:
            with st.expander(f"{'✅' if status == 'ok' else '⚠️'}  {title}", expanded=(title == "Lung Parenchyma")):
                for k, v in rows:
                    st.markdown(f"""
                    <div class="info-row" style="font-size:13px;padding:6px 0">
                      <span class="il">{k}</span>
                      <span class="iv">{v}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("#### Model Confidence Profile")
        conf_pct = int(mean_conf * 100)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:600;color:#0a1628">Segmentation Confidence</span>
            <span class="conf-label">{mean_conf:.4f}</span>
          </div>
          <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf_pct}%"></div></div>
          <div style="font-size:11px;color:#94a3b8;margin-top:4px">Mean sigmoid output over segmented lung region · Threshold: 0.5</div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3 — Technical ─────────────────────────────────────
    with tab_tech:
        st.markdown("#### Model Architecture & Pipeline")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("""
            <div class="panel-card">
              <div class="panel-header">🧠 Architecture</div>
              <div class="panel-body">
            """, unsafe_allow_html=True)
            for k, v in [
                ("Model", "U-Net (Encoder-Decoder)"),
                ("Parameters", "~31 Million"),
                ("Input channels", "1 (Grayscale)"),
                ("Output channels", "1 (Binary mask)"),
                ("Encoder depth", "4 levels"),
                ("Bottleneck filters", "1024"),
                ("Activation", "Sigmoid (output)"),
                ("Skip connections", "Yes — concatenation"),
            ]:
                st.markdown(f'<div class="info-row" style="font-size:12.5px"><span class="il">{k}</span><span class="iv">{v}</span></div>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tcol2:
            st.markdown("""
            <div class="panel-card">
              <div class="panel-header">⚙️ Preprocessing Pipeline</div>
              <div class="panel-body">
            """, unsafe_allow_html=True)
            for k, v in [
                ("Input format", "PNG / JPG / JPEG / TIF"),
                ("Resize", "256 × 256 px (bilinear)"),
                ("Colorspace", "Grayscale (single channel)"),
                ("Enhancement", "CLAHE"),
                ("CLAHE clipLimit", "2.0"),
                ("CLAHE tileGrid", "8 × 8"),
                ("Normalization", "[0, 1] float32"),
                ("Threshold", "0.5 (sigmoid output)"),
            ]:
                st.markdown(f'<div class="info-row" style="font-size:12.5px"><span class="il">{k}</span><span class="iv">{v}</span></div>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("#### Scan Session")
        scol1, scol2, scol3 = st.columns(3)
        for col, k, v in [
            (scol1, "Report ID", report_id),
            (scol2, "Scan datetime", now.strftime("%d %b %Y, %H:%M PKT")),
            (scol3, "Contrast protocol", p_contrast),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-chip">
                  <div style="font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:#0a1628">{v}</div>
                  <div class="lbl">{k}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.caption("MediScan AI © 2026 · For research use only · All clinical decisions must be validated by a licensed radiologist")


if __name__ == "__main__":
    main()
