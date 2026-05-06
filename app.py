import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os, warnings, cv2, io
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="MediScan AI", page_icon="🏥", layout="wide")

# Custom CSS for the Hospital Report Look
st.markdown("""
<style>
.report-container { background: white; border: 1px solid #c8d6e5; border-radius: 4px; padding: 20px; color: #1a1a2e; }
.report-header { background: #003366; color: white; padding: 15px; border-radius: 4px 4px 0 0; display: flex; justify-content: space-between; }
.section-title { font-weight: 700; color: #003366; border-bottom: 2px solid #003366; margin-bottom: 10px; text-transform: uppercase; font-size: 12px; }
.impression-box { background: #fff8e6; border: 1px solid #f0c040; padding: 10px; border-radius: 4px; font-size: 14px; }
.stButton>button { background-color: #003366 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# U-NET ARCHITECTURE (Matches notebook9dcaad60aa.ipynb)
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

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("🏥 Patient Intake")
    p_name = st.sidebar.text_input("Full Name", "John Doe")
    p_id = st.sidebar.text_input("MRN / Patient ID", "MRN-10293")
    
    st.title("🫁 MediScan AI Radiology Suite")
    
    model, loaded = load_model()
    if not loaded:
        st.error("Model weights not found.")
        return

    uploaded = st.file_uploader("Upload CT Axial Slice", type=["png","jpg","jpeg","tif"])
    
    if uploaded:
        raw_img = Image.open(uploaded)
        
        # Preprocess
        img_np = np.array(raw_img.convert("L").resize((256, 256)))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm_img = clahe.apply(img_np).astype(np.float32) / 255.0[cite: 1]
        
        # Predict
        t = torch.tensor(norm_img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            mask = (out.squeeze().numpy() > 0.5).astype(np.uint8)[cite: 1]
            lung_area = (np.sum(mask) / (256*256)) * 100

        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1:
            st.subheader("Source")
            # CHANGED: use_column_width instead of use_container_width to prevent TypeError
            st.image(raw_img, use_column_width=True)
            
        with col2:
            st.subheader("AI Mask")
            st.image(mask * 255, use_column_width=True)

        with col3:
            st.subheader("Clinical Report")
            report_id = datetime.now().strftime("RAD-%Y%H%M")
            report_text = f"Report ID: {report_id}\nPatient: {p_name}\nMRN: {p_id}\nLung Coverage: {lung_area:.2f}%"
            
            st.markdown(f"""
            <div class="report-container">
                <div class="report-header">
                    <span><b>MediScan AI Report</b></span>
                    <span>{datetime.now().strftime('%d %b %Y')}</span>
                </div>
                <div style="padding:10px">
                    <div class="section-title">Patient Info</div>
                    <p>Name: {p_name}<br>MRN: {p_id}</p>
                    <div class="section-title">AI Findings</div>
                    <p>Lung Coverage: {lung_area:.2f}%</p>
                    <div class="impression-box">
                        <strong>Impression:</strong> Segmentation successful. Borders delineated.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button("📥 Download (.txt)", report_text, file_name=f"{p_id}_Report.txt")

if __name__ == "__main__":
    main()
