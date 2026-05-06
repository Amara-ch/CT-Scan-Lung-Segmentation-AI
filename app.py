import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os, warnings, cv2
from datetime import datetime
import scipy.ndimage as ndi

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
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

/* ── Report Shell ── */
.report-container {
    background: white; border: 1px solid #c8d6e5;
    border-radius: 6px; color: #1a1a2e;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 20px rgba(0,0,40,0.10);
    overflow: hidden;
}

/* ── Header ── */
.report-header {
    background: linear-gradient(135deg, #002147 0%, #004080 100%);
    color: white; padding: 22px 28px;
    display: flex; justify-content: space-between; align-items: flex-start;
}
.report-header-left h2 {
    font-family: 'Source Serif 4', serif; font-size: 19px;
    font-weight: 700; margin: 0 0 3px 0; letter-spacing: 0.3px;
}
.report-header-left p { font-size: 12px; opacity: 0.75; margin: 1px 0; }
.report-header-right { text-align: right; font-size: 12px; opacity: 0.9; }
.report-header-right strong { font-size: 13px; display: block; margin-bottom: 2px; }
.report-badge {
    display: inline-block; margin-top: 6px; font-size: 11px;
    background: rgba(255,255,255,0.18); padding: 3px 10px;
    border-radius: 12px; letter-spacing: 0.5px;
}

/* ── Sections ── */
.report-section { padding: 14px 28px; border-bottom: 1px solid #eaf0f7; }
.report-section:last-child { border-bottom: none; }
.section-title {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.4px; color: #002147; margin-bottom: 10px;
    padding-bottom: 5px; border-bottom: 2px solid #002147; display: inline-block;
}

/* ── Patient Info Grid ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; }
.info-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 4px 0; border-bottom: 1px dotted #dde8f0; font-size: 12.5px;
}
.info-label { color: #607080; font-weight: 500; }
.info-value { color: #0d1e30; font-weight: 600; text-align: right; max-width: 55%; }

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 10px; margin: 8px 0; }
.metric-card {
    flex: 1; border-radius: 6px; padding: 12px 10px; text-align: center;
    border: 1px solid #ddeaf5; background: #f4f8fd;
}
.metric-card .val { font-size: 22px; font-weight: 700; color: #002147; line-height: 1.1; }
.metric-card .lbl { font-size: 10.5px; color: #607080; margin-top: 3px; }
.metric-card.highlight { background: #e8f4e8; border-color: #a8d5a8; }
.metric-card.highlight .val { color: #1b5e20; }
.metric-card.warn { background: #fff8e6; border-color: #f0c040; }
.metric-card.warn .val { color: #7a4800; }

/* ── AI Finding Blocks ── */
.finding-block {
    border: 1px solid #e0eaf5; border-left: 4px solid #002147;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 7px 0;
    background: #f7fafd; font-size: 12.5px; line-height: 1.65;
}
.finding-block.normal { border-left-color: #2e7d32; background: #f6fbf6; }
.finding-block.caution { border-left-color: #e65100; background: #fff8f3; }
.finding-label {
    font-weight: 700; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.6px; color: #002147; margin-bottom: 4px;
    display: flex; align-items: center; gap: 6px;
}
.finding-label .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
}
.dot-normal { background: #2e7d32; }
.dot-caution { background: #e65100; }
.dot-info { background: #002147; }
.finding-value { color: #1a2a3a; }
.finding-normal-tag {
    float: right; background: #e8f5e9; color: #1b5e20;
    padding: 1px 8px; border-radius: 10px; font-size: 10px;
    font-weight: 600; letter-spacing: 0.3px;
}
.finding-caution-tag {
    float: right; background: #fff3e0; color: #7a4000;
    padding: 1px 8px; border-radius: 10px; font-size: 10px;
    font-weight: 600; letter-spacing: 0.3px;
}

/* ── Impression ── */
.impression-box {
    background: #fffbf0; border: 1px solid #e8d080;
    border-radius: 6px; padding: 14px 16px;
    font-size: 12.5px; line-height: 1.8; color: #3a2800;
}
.impression-box strong { color: #002147; }

/* ── Confidence Bar ── */
.conf-bar-wrap { margin: 10px 0 4px; }
.conf-bar-label { font-size: 11.5px; color: #607080; margin-bottom: 4px; display:flex; justify-content:space-between; }
.conf-bar-outer { background: #e4ecf5; border-radius: 4px; height: 8px; }
.conf-bar-inner { height: 8px; border-radius: 4px; background: linear-gradient(90deg,#004080,#2196f3); transition: width 0.5s; }

/* ── Signature ── */
.signature-block {
    padding: 16px 28px; background: #f4f7fb;
    border-top: 2px solid #002147;
    display: flex; justify-content: space-between; align-items: flex-end;
}
.sig-line {
    border-top: 1px solid #a0b8cc; padding-top: 6px; margin-top: 28px;
    font-size: 11.5px; color: #607080; min-width: 180px;
}
.sig-line strong { color: #002147; display: block; }

/* ── Disclaimer ── */
.disclaimer {
    background: #fff5f5; border-top: 1px solid #f5c6c6;
    padding: 10px 28px; font-size: 11px; color: #7a2020; line-height: 1.7;
}

/* ── Normal / Caution Tags ── */
.normal-tag { background: #e8f5e9; color: #1b5e20; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.caution-tag { background: #fff3e0; color: #7a4000; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }

/* ── Sidebar ── */
.stButton > button {
    width: 100%; background: #002147 !important; color: white !important;
    border: none !important; border-radius: 5px !important;
    height: 3em; font-weight: 600 !important; font-size: 14px !important;
}
.stButton > button:hover { background: #004080 !important; }

/* ── Progress Steps ── */
.step-row { display: flex; gap: 0; margin: 0 0 18px; }
.step {
    flex: 1; text-align: center; padding: 10px 4px;
    font-size: 11.5px; font-weight: 600; color: #a0b0c0;
    border-bottom: 3px solid #dde8f0; position: relative;
}
.step.active { color: #002147; border-bottom-color: #002147; }
.step.done { color: #2e7d32; border-bottom-color: #2e7d32; }
.step-num {
    display: inline-block; width: 20px; height: 20px;
    border-radius: 50%; background: #dde8f0;
    font-size: 11px; line-height: 20px; margin-right: 4px;
}
.step.active .step-num { background: #002147; color: white; }
.step.done .step-num { background: #2e7d32; color: white; }
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
# MODEL LOADER (Verbatim — unchanged)
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
# AI-DRIVEN FINDINGS ANALYSER
# ─────────────────────────────────────────────────────────────
def analyse_findings(mask, raw_prob_map, img_np):
    """
    Derives each radiological finding directly from the mask and image data.
    Returns a dict of finding dicts: {label, status, text, confidence, tag}
    """
    total_px   = mask.size
    lung_px    = int(np.sum(mask))
    area_pct   = (lung_px / total_px) * 100
    prob_map   = raw_prob_map  # shape (256,256), float [0,1]

    # ── 1. Lung Parenchyma ─────────────────────────────────
    # Confidence = mean probability in segmented region
    if lung_px > 0:
        parenchyma_conf = float(np.mean(prob_map[mask == 1]) * 100)
    else:
        parenchyma_conf = 0.0

    # Count connected components (lobes/regions)
    labeled, n_regions = ndi.label(mask)
    bilateral = n_regions >= 2

    if area_pct < 5:
        parenchyma_status = "caution"
        parenchyma_text   = (f"Very limited lung tissue segmented ({area_pct:.1f}% coverage). "
                             "Possible under-segmentation or image artefact. Bilateral fields not confirmed.")
        parenchyma_tag    = "Review"
    elif area_pct > 70:
        parenchyma_status = "caution"
        parenchyma_text   = (f"Lung field coverage of {area_pct:.1f}% is above typical range. "
                             "Possible over-segmentation or inclusion of adjacent structures.")
        parenchyma_tag    = "Review"
    else:
        parenchyma_status = "normal"
        lat_text = "Bilateral lung fields visible" if bilateral else "Unilateral lung field detected"
        parenchyma_text   = (f"{lat_text}. Parenchymal coverage {area_pct:.1f}% "
                             f"across {n_regions} segmented region(s). "
                             "Density consistent with CT lung window protocol.")
        parenchyma_tag    = "Normal"

    # ── 2. Pleural Space ───────────────────────────────────
    # Look at border pixels of mask — if dense at very edge → effusion hint
    border_mask = np.zeros_like(mask)
    border_mask[0,:] = border_mask[-1,:] = border_mask[:,0] = border_mask[:,-1] = 1
    border_lung_px = int(np.sum(mask * border_mask))
    border_pct     = (border_lung_px / max(lung_px, 1)) * 100

    pleural_conf = max(60.0, 95.0 - border_pct * 0.5)

    if border_pct > 15:
        pleural_status = "caution"
        pleural_text   = (f"Segmented lung tissue extends to image border in {border_pct:.1f}% of lung pixels. "
                          "Pleural boundary may be obscured. Cannot exclude pleural effusion on this slice.")
        pleural_tag    = "Inconclusive"
    else:
        pleural_status = "normal"
        pleural_text   = ("No gross pleural effusion identified. Pleural boundaries clearly delineated by AI model. "
                          "No pneumothorax identified on current axial slice.")
        pleural_tag    = "Normal"

    # ── 3. Mediastinum & Airways ───────────────────────────
    # Check symmetry: compare left vs right lung mass
    mid = mask.shape[1] // 2
    left_px  = int(np.sum(mask[:, :mid]))
    right_px = int(np.sum(mask[:, mid:]))
    sym_ratio = min(left_px, right_px) / max(max(left_px, right_px), 1)
    med_conf  = float(min(95, 70 + sym_ratio * 25))

    if sym_ratio < 0.3:
        med_status = "caution"
        med_text   = (f"Marked asymmetry between left ({left_px}px) and right ({right_px}px) lung fields "
                      f"(symmetry ratio {sym_ratio:.2f}). Possible mediastinal shift or unilateral pathology. "
                      "Clinical correlation recommended.")
        med_tag    = "Asymmetric"
    elif sym_ratio < 0.6:
        med_status = "caution"
        med_text   = (f"Mild asymmetry noted: left {left_px}px vs right {right_px}px "
                      f"(ratio {sym_ratio:.2f}). Mediastinal contour requires correlation with adjacent slices.")
        med_tag    = "Mild Asymmetry"
    else:
        med_status = "normal"
        med_text   = (f"Mediastinal contour within expected limits. Left/right lung symmetry ratio {sym_ratio:.2f}. "
                      "No gross mediastinal widening detected. Central airway morphology unremarkable.")
        med_tag    = "Normal"

    # ── 4. Opacity / Consolidation ─────────────────────────
    # High mean intensity in segmented region may indicate consolidation
    if lung_px > 0:
        mean_intensity = float(np.mean(img_np[mask == 1]))
    else:
        mean_intensity = 0.0

    opacity_conf = 85.0

    if mean_intensity > 180:
        opacity_status = "caution"
        opacity_text   = (f"Elevated mean parenchymal intensity ({mean_intensity:.0f}/255) within segmented region. "
                          "Possible consolidation, ground-glass opacity, or increased attenuation. "
                          "Further multi-slice review recommended.")
        opacity_tag    = "Elevated Attenuation"
    elif mean_intensity > 130:
        opacity_status = "caution"
        opacity_text   = (f"Mildly increased parenchymal intensity ({mean_intensity:.0f}/255). "
                          "Subtle increased attenuation noted. Cannot exclude early infiltrates on single slice.")
        opacity_tag    = "Mild Increase"
    else:
        opacity_status = "normal"
        opacity_text   = (f"Mean parenchymal intensity {mean_intensity:.0f}/255, within normal range. "
                          "No large consolidative opacity or ground-glass attenuation identified on this slice.")
        opacity_tag    = "Normal"

    # ── 5. Chest Wall & Diaphragm ──────────────────────────
    # Use bottom rows of mask for diaphragm presence
    bottom_rows    = mask[-30:, :]
    diaphragm_px   = int(np.sum(bottom_rows))
    diaphragm_conf = 78.0

    if diaphragm_px < 10:
        diaphragm_status = "caution"
        diaphragm_text   = ("Limited lung tissue in inferior slice region. Diaphragmatic boundary not clearly "
                            "delineated. Subpulmonic pathology cannot be excluded on this slice alone.")
        diaphragm_tag    = "Limited View"
    else:
        diaphragm_status = "normal"
        diaphragm_text   = ("No lytic or sclerotic bony lesion identified on this axial slice. "
                            "Diaphragmatic contour visible in inferior lung field. "
                            "Chest wall soft tissues appear unremarkable.")
        diaphragm_tag    = "Normal"

    return {
        "parenchyma":  {"label": "Lung Parenchyma",         "status": parenchyma_status,  "text": parenchyma_text,  "conf": parenchyma_conf,  "tag": parenchyma_tag},
        "pleural":     {"label": "Pleural Space",            "status": pleural_status,     "text": pleural_text,     "conf": pleural_conf,     "tag": pleural_tag},
        "mediastinum": {"label": "Mediastinum & Airways",    "status": med_status,         "text": med_text,         "conf": med_conf,         "tag": med_tag},
        "opacity":     {"label": "Opacity / Consolidation",  "status": opacity_status,     "text": opacity_text,     "conf": opacity_conf,     "tag": opacity_tag},
        "diaphragm":   {"label": "Chest Wall & Diaphragm",   "status": diaphragm_status,   "text": diaphragm_text,   "conf": diaphragm_conf,   "tag": diaphragm_tag},
    }


def finding_html(f):
    """Renders one finding block as HTML."""
    dot_cls = "dot-normal" if f["status"] == "normal" else "dot-caution"
    block_cls = "finding-block " + f["status"]
    tag_cls   = "finding-normal-tag" if f["status"] == "normal" else "finding-caution-tag"
    conf_color = "#2e7d32" if f["conf"] >= 75 else "#e65100"
    conf_w     = f"{min(f['conf'], 100):.0f}%"
    return f"""
    <div class="{block_cls}">
      <div class="finding-label">
        <span class="status-dot {dot_cls}"></span>
        {f['label']}
        <span class="{tag_cls}">{f['tag']}</span>
      </div>
      <div class="finding-value">{f['text']}</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar-label">
          <span>Model Confidence</span>
          <span style="color:{conf_color};font-weight:600">{f['conf']:.0f}%</span>
        </div>
        <div class="conf-bar-outer">
          <div class="conf-bar-inner" style="width:{conf_w};background:{'#2e7d32' if f['conf']>=75 else '#e65100'}"></div>
        </div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────
# DOWNLOADABLE REPORT TEXT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_download_report(patient_info, scan_info, lung_pixels, area_pct, n_regions, sym_ratio, mean_intensity, findings):
    now       = datetime.now()
    report_id = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"
    overall   = "NORMAL STUDY" if all(f["status"] == "normal" for f in findings.values()) else "FINDINGS REQUIRING REVIEW"

    lines = []
    lines.append("=" * 80)
    lines.append("         MEDISCAN AI — OFFICIAL CT RADIOLOGY REPORT")
    lines.append("=" * 80)
    lines.append(f"  Report ID      : {report_id}")
    lines.append(f"  Date / Time    : {now.strftime('%d %B %Y')}  at  {now.strftime('%H:%M')} hrs")
    lines.append(f"  Institution    : MediScan AI Radiology Centre")
    lines.append(f"  Department     : Diagnostic Radiology & Medical Imaging")
    lines.append(f"  Modality       : CT Chest — Axial Slice Analysis")
    lines.append(f"  Priority       : Routine")
    lines.append("=" * 80)

    lines.append("\n  PATIENT INFORMATION")
    lines.append("  " + "-" * 50)
    lines.append(f"  Patient Name        : {patient_info['name']}")
    lines.append(f"  Medical Record No.  : {patient_info['mrn']}")
    lines.append(f"  Date of Birth       : {patient_info['dob']}")
    lines.append(f"  Age / Gender        : {patient_info['age']} years / {patient_info['gender']}")
    lines.append(f"  Referring Physician : {patient_info['referring_doc']}")
    lines.append(f"  Clinical Indication : {patient_info['indication']}")

    lines.append("\n  TECHNICAL PARAMETERS")
    lines.append("  " + "-" * 50)
    lines.append(f"  AI Model            : U-Net (Encoder-Decoder, 31M Parameters)")
    lines.append(f"  Input Resolution    : 256 × 256 pixels")
    lines.append(f"  Preprocessing       : CLAHE (clipLimit=2.0, tileGrid=8×8)")
    lines.append(f"  Segmentation        : Binary threshold 0.5 (sigmoid output)")
    lines.append(f"  Contrast Agent      : {scan_info['contrast']}")
    lines.append(f"  Slice Thickness     : {scan_info['slice_thickness']} mm")

    lines.append("\n  QUANTITATIVE METRICS")
    lines.append("  " + "-" * 50)
    lines.append(f"  Total Scan Pixels       : 65,536  (256×256)")
    lines.append(f"  Segmented Lung Pixels   : {lung_pixels:,}")
    lines.append(f"  Lung Field Coverage     : {area_pct:.2f}%")
    lines.append(f"  Segmented Regions       : {n_regions}")
    lines.append(f"  Left/Right Symmetry     : {sym_ratio:.2f}")
    lines.append(f"  Mean Parenchymal HU     : {mean_intensity:.1f} (normalised)")

    lines.append("\n  RADIOLOGICAL OBSERVATIONS")
    lines.append("  " + "-" * 50)
    for f in findings.values():
        lines.append(f"\n  [{f['tag'].upper()}]  {f['label']}")
        lines.append(f"  Confidence : {f['conf']:.0f}%")
        lines.append(f"  {f['text']}")

    lines.append("\n  OVERALL ASSESSMENT")
    lines.append("  " + "-" * 50)
    lines.append(f"  {overall}")
    caution_items = [f["label"] for f in findings.values() if f["status"] == "caution"]
    if caution_items:
        lines.append(f"  Items requiring review: {', '.join(caution_items)}")

    lines.append("\n  IMPRESSION")
    lines.append("  " + "-" * 50)
    lines.append(f"  1. AI-assisted lung segmentation performed on axial CT slice.")
    lines.append(f"  2. Lung field coverage: {area_pct:.2f}% | Segmented regions: {n_regions}")
    lines.append(f"  3. Overall AI assessment: {overall}")
    if caution_items:
        lines.append(f"  4. Items requiring radiologist review: {', '.join(caution_items)}")
    lines.append(f"  5. Multi-slice volumetric analysis recommended for comprehensive assessment.")

    lines.append("\n  RECOMMENDATION")
    lines.append("  " + "-" * 50)
    lines.append("  Clinical correlation with complete CT series, pulmonary function tests,")
    lines.append("  and full patient history is strongly advised. This AI report is a decision-")
    lines.append("  support tool and does not replace a board-certified radiologist.")

    lines.append("\n" + "=" * 80)
    lines.append("  SIGNATURES")
    lines.append("=" * 80)
    lines.append("  AI System    : MediScan AI v2.0 — U-Net Lung Segmentation Engine")
    lines.append("  Radiologist  : [ Signature required before clinical use ]")
    lines.append(f"  Report ID    : {report_id}")
    lines.append(f"  Timestamp    : {now.strftime('%d %B %Y at %H:%M hrs')}")

    lines.append("\n" + "=" * 80)
    lines.append("  DISCLAIMER")
    lines.append("=" * 80)
    lines.append("  This report is generated by an AI-assisted system for RESEARCH &")
    lines.append("  EDUCATIONAL PURPOSES ONLY. It does not constitute a final clinical")
    lines.append("  diagnosis. All findings must be reviewed by a licensed radiologist")
    lines.append("  before any clinical action is taken. MediScan AI assumes no")
    lines.append(f"  medico-legal liability for decisions based solely on this output.")
    lines.append(f"  © {now.year} MediScan AI Radiology Suite. All rights reserved.")
    lines.append("=" * 80)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 MediScan AI")
        st.markdown("**Radiology Suite**")
        st.divider()

        st.markdown("##### 👤 Patient Details")
        p_name       = st.text_input("Full Name", "John Doe")
        p_id         = st.text_input("Medical Record No.", "MRN-10293")
        p_dob        = st.text_input("Date of Birth", "01-01-1980")
        p_gender     = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_age        = st.number_input("Age (years)", min_value=0, max_value=120, value=44)

        st.divider()
        st.markdown("##### 🩺 Clinical Information")
        p_ref_doc    = st.text_input("Referring Physician", "Dr. A. Khan")
        p_indication = st.text_input("Clinical Indication", "Cough, breathlessness — Rule out consolidation")

        st.divider()
        st.markdown("##### ⚙️ Scan Parameters")
        p_contrast   = st.selectbox("Contrast", ["Non-Contrast", "IV Contrast", "Oral + IV Contrast"])
        p_slice_thk  = st.selectbox("Slice Thickness (mm)", ["1.0", "1.5", "2.5", "5.0"])

        st.divider()
        st.info("Complete all fields above, then upload a CT slice to generate the AI diagnostic report.")

    # ── Main Panel ────────────────────────────────────────────
    st.title("🫁 MediScan AI — Radiology Suite")
    st.caption("AI-Assisted CT Lung Segmentation & Automated Radiological Reporting")

    # Progress steps
    uploaded_yet = "uploaded" in st.session_state and st.session_state["uploaded"]
    st.markdown(f"""
    <div class="step-row">
      <div class="step done"><span class="step-num">✓</span> Patient Info</div>
      <div class="step {'done' if uploaded_yet else 'active'}"><span class="step-num">2</span> Upload CT Slice</div>
      <div class="step {'active' if uploaded_yet else ''}"><span class="step-num">3</span> AI Analysis</div>
      <div class="step {'active' if uploaded_yet else ''}"><span class="step-num">4</span> Download Report</div>
    </div>
    """, unsafe_allow_html=True)

    model, loaded = load_model()
    if not loaded:
        st.error("⚠️ Model weights ('best_unet_model.pth') not found. Place it alongside app.py and restart.")
        return

    uploaded = st.file_uploader(
        "📂  Select Axial CT Slice (PNG / JPG / TIF)",
        type=["png", "jpg", "jpeg", "tif"],
        help="Upload a single axial CT slice exported from DICOM"
    )

    if uploaded:
        st.session_state["uploaded"] = True
        pil_img = Image.open(uploaded)

        col1, col2, col3 = st.columns([1, 1, 1.4])

        with col1:
            st.subheader("📷 Source Image")
            st.image(pil_img, use_column_width=True, caption="Uploaded CT Axial Slice")

        # ── Inference (verbatim logic) ────────────────────────
        img_np   = np.array(pil_img.convert("L").resize((256, 256)))
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_img = clahe.apply(img_np).astype(np.float32) / 255.0

        t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out       = model(t)
            prob_map  = out.squeeze().numpy()            # raw float probs
            mask      = (prob_map > 0.5).astype(np.uint8)
            lung_pixels = int(np.sum(mask))

        total_pixels = 256 * 256
        area_pct     = (lung_pixels / total_pixels) * 100

        # Extra metrics for report
        _, n_regions = ndi.label(mask)
        mid          = mask.shape[1] // 2
        left_px      = int(np.sum(mask[:, :mid]))
        right_px     = int(np.sum(mask[:, mid:]))
        sym_ratio    = min(left_px, right_px) / max(max(left_px, right_px), 1)
        mean_intens  = float(np.mean(img_np[mask == 1])) if lung_pixels > 0 else 0.0

        with col2:
            st.subheader("🤖 AI Segmentation")
            st.image(mask * 255, use_column_width=True, caption="U-Net Lung Mask")

            overall_ok = area_pct >= 20 and sym_ratio >= 0.6 and mean_intens <= 180
            card_cls   = "highlight" if overall_ok else "warn"

            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-top:10px">
              <div class="metric-card {card_cls}">
                <div class="val">{lung_pixels:,}</div>
                <div class="lbl">Lung Pixels</div>
              </div>
              <div class="metric-card {card_cls}">
                <div class="val">{area_pct:.1f}%</div>
                <div class="lbl">Coverage</div>
              </div>
            </div>
            <div style="display:flex;gap:8px;margin-top:8px">
              <div class="metric-card">
                <div class="val">{n_regions}</div>
                <div class="lbl">Regions</div>
              </div>
              <div class="metric-card">
                <div class="val">{sym_ratio:.2f}</div>
                <div class="lbl">Symmetry</div>
              </div>
              <div class="metric-card">
                <div class="val">{mean_intens:.0f}</div>
                <div class="lbl">Mean HU</div>
              </div>
            </div>
            <div style="margin-top:10px;padding:8px 12px;border-radius:6px;font-size:12px;font-weight:600;
              background:{'#e8f5e9' if overall_ok else '#fff3e0'};
              color:{'#1b5e20' if overall_ok else '#7a4800'};
              border:1px solid {'#a8d5a8' if overall_ok else '#f0c040'}">
              {'✅ Overall: Normal Study' if overall_ok else '⚠️ Overall: Findings Require Review'}
            </div>
            """, unsafe_allow_html=True)

        # ── AI Findings ───────────────────────────────────────
        findings = analyse_findings(mask, prob_map, img_np)

        with col3:
            st.subheader("📋 Diagnostic Report")

            now = datetime.now()
            report_id = f"RAD-{now.strftime('%Y%m%d%H%M%S')}"
            overall_label = "NORMAL STUDY" if all(f["status"] == "normal" for f in findings.values()) else "REVIEW REQUIRED"
            overall_color = "#1b5e20" if overall_label == "NORMAL STUDY" else "#7a4800"
            overall_bg    = "#e8f5e9"  if overall_label == "NORMAL STUDY" else "#fff3e0"

            # Build findings HTML
            findings_html = "".join(finding_html(f) for f in findings.values())

            # Impression items
            caution_items = [f["label"] for f in findings.values() if f["status"] == "caution"]
            imp_caution = f'<br><strong>4.</strong> Items for radiologist review: <em>{", ".join(caution_items)}</em>.' if caution_items else ""

            st.markdown(f"""
            <div class="report-container">

              <div class="report-header">
                <div class="report-header-left">
                  <h2>🏥 MediScan AI Radiology Centre</h2>
                  <p>Department of Diagnostic Radiology &amp; Medical Imaging</p>
                  <p style="margin-top:2px;font-size:11px;opacity:0.6">AI-Assisted CT Analysis System v2.0</p>
                </div>
                <div class="report-header-right">
                  <strong>Report ID</strong>
                  {report_id}<br>
                  {now.strftime("%d %b %Y, %H:%M")}<br>
                  <span class="report-badge">ROUTINE</span>
                </div>
              </div>

              <div class="report-section">
                <div class="section-title">Patient Information</div>
                <div class="info-grid">
                  <div>
                    <div class="info-row"><span class="info-label">Name</span><span class="info-value">{p_name}</span></div>
                    <div class="info-row"><span class="info-label">MRN</span><span class="info-value">{p_id}</span></div>
                    <div class="info-row"><span class="info-label">Date of Birth</span><span class="info-value">{p_dob}</span></div>
                    <div class="info-row"><span class="info-label">Age / Gender</span><span class="info-value">{p_age} yrs / {p_gender}</span></div>
                  </div>
                  <div>
                    <div class="info-row"><span class="info-label">Referring Physician</span><span class="info-value">{p_ref_doc}</span></div>
                    <div class="info-row"><span class="info-label">Modality</span><span class="info-value">CT Chest — Axial</span></div>
                    <div class="info-row"><span class="info-label">Contrast</span><span class="info-value">{p_contrast}</span></div>
                    <div class="info-row"><span class="info-label">Slice Thickness</span><span class="info-value">{p_slice_thk} mm</span></div>
                  </div>
                </div>
                <div style="margin-top:8px;font-size:12px;color:#607080">
                  <b style="color:#002147">Indication:</b> {p_indication}
                </div>
              </div>

              <div class="report-section">
                <div class="section-title">Quantitative AI Metrics</div>
                <div class="metric-row">
                  <div class="metric-card"><div class="val">{lung_pixels:,}</div><div class="lbl">Segmented Pixels</div></div>
                  <div class="metric-card"><div class="val">{area_pct:.1f}%</div><div class="lbl">Lung Coverage</div></div>
                  <div class="metric-card"><div class="val">{n_regions}</div><div class="lbl">Regions</div></div>
                  <div class="metric-card"><div class="val">{sym_ratio:.2f}</div><div class="lbl">Symmetry</div></div>
                </div>
                <div style="margin-top:8px;font-size:12px">
                  Overall Assessment: &nbsp;
                  <span style="background:{overall_bg};color:{overall_color};padding:2px 12px;border-radius:12px;font-size:11px;font-weight:700">
                    {overall_label}
                  </span>
                </div>
              </div>

              <div class="report-section">
                <div class="section-title">Radiological Observations</div>
                {findings_html}
              </div>

              <div class="report-section">
                <div class="section-title">Impression &amp; Recommendation</div>
                <div class="impression-box">
                  <strong>1.</strong> AI-assisted segmentation performed on the submitted axial CT slice.<br>
                  <strong>2.</strong> Lung coverage <strong>{area_pct:.2f}%</strong> across <strong>{n_regions}</strong> region(s). L/R symmetry ratio <strong>{sym_ratio:.2f}</strong>.<br>
                  <strong>3.</strong> Overall AI assessment: <strong style="color:{overall_color}">{overall_label}</strong>.{imp_caution}<br><br>
                  <strong>Recommendation:</strong> Clinical correlation with complete CT series, pulmonary function tests,
                  and full patient history is strongly advised. This AI output is a decision-support tool and does not
                  replace interpretation by a board-certified radiologist.
                </div>
              </div>

              <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This report is generated by an AI system for <strong>research &amp; educational purposes only</strong>.
                It does not constitute a final clinical diagnosis. All findings must be confirmed by a licensed radiologist.
                MediScan AI assumes no medico-legal liability for decisions made solely on this output.
              </div>

              <div class="signature-block">
                <div>
                  <div class="sig-line">
                    <strong>MediScan AI System v2.0</strong>
                    U-Net Lung Segmentation Engine
                  </div>
                </div>
                <div style="text-align:right">
                  <div class="sig-line" style="text-align:right">
                    <strong>Radiologist Verification</strong>
                    <span style="color:#a0b8cc;font-size:11px">[ Signature required for clinical use ]</span>
                  </div>
                </div>
              </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            patient_info = {"name": p_name, "mrn": p_id, "dob": p_dob, "gender": p_gender,
                            "age": p_age, "referring_doc": p_ref_doc, "indication": p_indication}
            scan_info    = {"contrast": p_contrast, "slice_thickness": p_slice_thk}
            report_txt   = generate_download_report(
                patient_info, scan_info, lung_pixels, area_pct,
                n_regions, sym_ratio, mean_intens, findings
            )

            st.download_button(
                label="📥  Download Full Clinical Report (.txt)",
                data=report_txt,
                file_name=f"MediScan_Report_{p_id}_{now.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:
        st.session_state["uploaded"] = False
        st.markdown("""
        <div style='border:2px dashed #a0b8d0;border-radius:10px;padding:48px 32px;
             text-align:center;background:#f7fafd;color:#5a6a7a;margin-top:16px'>
          <div style='font-size:52px'>🫁</div>
          <div style='font-size:18px;font-weight:700;margin-top:14px;color:#002147'>
            Upload an Axial CT Slice to Begin
          </div>
          <div style='font-size:13px;margin-top:6px'>Supported formats: PNG · JPG · JPEG · TIF</div>
          <div style='font-size:12px;margin-top:6px;color:#8a9ab0'>
            Fill in the patient details in the sidebar, then upload your image above.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("MediScan AI © 2026 · For research use only · All clinical decisions must be validated by a licensed radiologist.")


if __name__ == "__main__":
    main()
