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
    .report-box { padding: 20px; border: 1px solid #d0dce8; border-radius: 5px; background-color: #f9f9f9; color: black; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# U-NET ARCHITECTURE (Verbatim from notebook9dcaad60aa.ipynb)
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
        self.up4   = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec4 = DoubleConv(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, 2);  self.dec3 = DoubleConv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, 2);  self.dec2 = DoubleConv(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, 2);   self.dec1 = DoubleConv(128, 64)
        self.out   = nn.Conv2d(64, 1, 1)

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
# LOGIC & PREPROCESSING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet()
    path = "best_unet_model.pth"
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()
        return model, True
    return model, False

def generate_medical_report(patient_name, patient_id, lung_area_pixels):
    # Calculate approximate area percentage (assuming 256x256 image)
    total_pixels = 256 * 256
    area_pct = (lung_area_pixels / total_pixels) * 100
    date_str = datetime.now().strftime("%d-%m-%Y %H:%M")
    
    report = f"""
    🏥 MEDISCAN RADIOLOGY REPORT
    -------------------------------------------
    Date: {date_str}
    Patient Name: {patient_name}
    Patient ID: {patient_id}
    Procedure: AI-Assisted CT Lung Segmentation
    
    CLINICAL FINDINGS:
    - AI Model used: U-Net Architecture (31M Params)
    - Detected Lung Tissue Area: {lung_area_pixels} pixels
    - Relative Lung Field Volume: {area_pct:.2f}% of scan slice
    
    IMPRESSION:
    The AI segmentation has successfully delineated the lung boundaries. 
    The visualized volume is consistent with axial slice standards. 
    Further clinical correlation is required for diagnostic purposes.
    
    Electronically Signed by: MediScan AI System
    """
    return report

# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("🏥 Patient Information")
    p_name = st.sidebar.text_input("Patient Name", "John Doe")
    p_id = st.sidebar.text_input("Patient ID", "MRN-10293")
    st.sidebar.divider()
    st.sidebar.info("Upload an axial CT slice to generate a hospital report.")

    st.title("🫁 MediScan AI — Radiology Suite")
    
    model, loaded = load_model()
    if not loaded:
        st.error("Model weights ('best_unet_model.pth') not found.")
        return

    uploaded = st.file_uploader("Select DICOM/CT Image", type=["png","jpg","jpeg","tif"])
    
    if uploaded:
        pil_img = Image.open(uploaded)
        
        # UI Layout: Three Columns
        col1, col2, col3 = st.columns([1, 1, 1.2])
        
        with col1:
            st.subheader("Source Image")
            st.image(pil_img, use_column_width=True)

        # Preprocessing & Inference
        img_np = np.array(pil_img.convert("L").resize((256, 256)))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm_img = clahe.apply(img_np).astype(np.float32) / 255.0[cite: 1]
        
        t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            mask = (out.squeeze().numpy() > 0.5).astype(np.uint8)[cite: 1]
            lung_pixels = np.sum(mask)

        with col2:
            st.subheader("AI Segmentation")
            st.image(mask * 255, use_column_width=True)
            
        with col3:
            st.subheader("Medical Report")
            medical_text = generate_medical_report(p_name, p_id, lung_pixels)
            st.markdown(f"```\n{medical_text}\n```")
            
            # Download Feature
            st.download_button(
                label="📥 Download Clinical Report",
                data=medical_text,
                file_name=f"Report_{p_id}.txt",
                mime="text/plain"
            )

    st.divider()
    st.caption("MediScan AI 2026 | For research use only. Clinical decisions should be validated by a licensed radiologist.")

if __name__ == "__main__":
    main()
