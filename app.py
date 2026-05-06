import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os, warnings
import cv2

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | CT Lung Segmentation",
    page_icon="🫁",
    layout="wide"
)

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
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Use CPU for deployment compatibility
    device = torch.device("cpu")
    model = UNet()
    
    model_path = "best_unet_model.pth"
    if os.path.exists(model_path):
        # Load state_dict directly
        state_dict = torch.load(model_path, map_location=device)
        # If weights were saved with a 'model_state_dict' key, extract it
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        model.load_state_dict(state_dict)
        model.eval()
        return model, device, True
    return model, device, False

# ─────────────────────────────────────────────────────────────
# PREPROCESSING (Matches Training in notebook9dcaad60aa.ipynb)
# ─────────────────────────────────────────────────────────────
def preprocess(pil_img):
    # Convert to grayscale and resize to 256x256
    img = np.array(pil_img.convert("L").resize((256, 256)))
    
    # Apply CLAHE as used in training
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img).astype(np.float32) / 255.0[cite: 1]
    
    # Add batch and channel dimensions: (1, 1, 256, 256)
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🫁 MediScan AI — CT Lung Segmentation")
    
    model, device, loaded = load_model()
    
    if not loaded:
        st.error("❌ Model weights not found. Please ensure 'best_unet_model.pth' is in the app directory.")
        return

    uploaded = st.file_uploader("Upload a CT axial slice", type=["png","jpg","jpeg","tif"])
    
    if uploaded:
        pil_img = Image.open(uploaded)
        
        # UI Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(pil_img, caption="Original CT Scan", use_container_width=True)
            
        # Inference
        input_tensor = preprocess(pil_img)
        with torch.no_grad():
            prediction = model(input_tensor.to(device))
            # Binary threshold at 0.5[cite: 1]
            mask = (prediction.squeeze().numpy() > 0.5).astype(np.uint8) 
            
        with col2:
            # Multiply by 255 to make the mask visible as a white-on-black image
            st.image(mask * 255, caption="Predicted Lung Mask", use_container_width=True)
            
        st.success("Analysis complete.")

if __name__ == "__main__":
    main()
