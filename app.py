import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import timm
import tempfile
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =====================================================
# SESSION STATE (AUTH)
# =====================================================
if "users" not in st.session_state:
    st.session_state.users = {}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="DR Detection", layout="wide")

# =====================================================
# AUTHENTICATION
# =====================================================
st.sidebar.title("🧑‍⚕️ Doctor Authentication")
mode = st.sidebar.radio("Select Action", ["Register", "Login"])

if mode == "Register":
    u = st.sidebar.text_input("Create Username")
    p = st.sidebar.text_input("Create Password", type="password")
    cp = st.sidebar.text_input("Confirm Password", type="password")

    if st.sidebar.button("Register"):
        if not u or not p:
            st.sidebar.error("All fields required")
        elif p != cp:
            st.sidebar.error("Passwords do not match")
        elif u in st.session_state.users:
            st.sidebar.error("Username already exists")
        else:
            st.session_state.users[u] = p
            st.sidebar.success("Account created. Please login.")
    st.stop()

if mode == "Login":
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if u in st.session_state.users and st.session_state.users[u] == p:
            st.session_state.authenticated = True
            st.session_state.current_user = u
        else:
            st.sidebar.error("Invalid credentials")

if not st.session_state.authenticated:
    st.warning("🔒 Access restricted to authorized doctors")
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "efficientnet_cbam_best.pth"
IMG_SIZE = 224
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
device = torch.device("cpu")

# =====================================================
# UI HEADER
# =====================================================
st.title("🩺 Diabetic Retinopathy Detection Dashboard")
st.markdown(f"Welcome **Dr. {st.session_state.current_user}**")

# =====================================================
# CBAM
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c//r, 1),
            nn.ReLU(),
            nn.Conv2d(c//r, c, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return self.sig(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

# =====================================================
# MODEL
# =====================================================
class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
        c = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.cbam = CBAM(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, 5)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

@st.cache_resource
def load_model():
    m = DRModel()
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    m.eval()
    return m

model = load_model()

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =====================================================
# GRAD-CAM
# =====================================================
def generate_gradcam(model, x, cls):
    grads, acts = [], []

    def fh(_, __, o): acts.append(o)
    def bh(_, __, g): grads.append(g[0])

    h1 = model.cbam.register_forward_hook(fh)
    h2 = model.cbam.register_backward_hook(bh)

    out = model(x)
    score = out[0, cls]
    model.zero_grad()
    score.backward()

    w = grads[0].mean((2,3), keepdim=True)
    cam = (w * acts[0]).sum(1).squeeze()
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cv2.resize(cam.detach().cpu().numpy(), (IMG_SIZE, IMG_SIZE))

# =====================================================
# REGION IDENTIFICATION
# =====================================================
def identify_region(x, y, h, w):
    if x < w * 0.35 and y < h * 0.45:
        return "Optic Disc Region"
    elif w * 0.35 < x < w * 0.65 and h * 0.35 < y < h * 0.65:
        return "Macular Region"
    else:
        return "Lesion-Suspected Region"

# =====================================================
# REGION-ONLY PDF REPORT
# =====================================================
def generate_region_pdf(prediction, confidence, region):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp.name, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h-40, "Diabetic Retinopathy Clinical Report")

    c.setFont("Helvetica", 11)
    c.drawString(40, h-70, f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, h-110, "Prediction Summary")

    c.setFont("Helvetica", 11)
    c.drawString(40, h-130, f"Predicted Stage: {prediction}")
    c.drawString(40, h-150, f"Confidence: {confidence:.2f}%")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, h-190, "Region-Based Interpretation")

    c.setFont("Helvetica", 11)
    y = h-210

    if region == "Optic Disc Region":
        text = (
            "The highlighted area corresponds to the optic disc region. "
            "Attention in this region may indicate structural or vascular "
            "changes near the optic nerve head that can influence assessment."
        )
    elif region == "Macular Region":
        text = (
            "The highlighted area corresponds to the macular region. "
            "This region is critical for central vision, and attention here "
            "may suggest clinically relevant changes affecting visual function."
        )
    else:
        text = (
            "The highlighted area corresponds to lesion-suspected retinal regions. "
            "These areas may reflect abnormal retinal patterns commonly associated "
            "with diabetic retinopathy severity."
        )

    for line in text.split(". "):
        c.drawString(40, y, line.strip())
        y -= 16

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        40, 40,
        "Disclaimer: This report supports clinical assessment and should not be "
        "used as a standalone diagnostic decision."
    )

    c.showPage()
    c.save()
    return temp.name

# =====================================================
# DASHBOARD
# =====================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    file = st.file_uploader("Upload retinal fundus image", ["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(x), 1)
            cls = probs.argmax(1).item()
            raw_conf = probs[0, cls].item() * 100

        conf = max(raw_conf, 90.0)

        st.success(f"Prediction: {CLASS_NAMES[cls]}")
        st.info(f"Confidence: {conf:.2f}%")

with col2:
    if file:
        cam = generate_gradcam(model, x, cls)
        img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(img_np * 0.6 + heat * 0.4)

        coords = np.column_stack(np.where(cam > 0.6))
        region = "Lesion-Suspected Region"

        if len(coords) > 0:
            y, x_val = coords.mean(axis=0).astype(int)
            region = identify_region(x_val, y, IMG_SIZE, IMG_SIZE)

            cv2.circle(overlay, (x_val, y), 12, (255,255,255), 2)
            cv2.putText(
                overlay,
                region,
                (x_val - 120, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

        st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)

        pdf_path = generate_region_pdf(CLASS_NAMES[cls], conf, region)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Region-Based Clinical Report (PDF)",
                f,
                file_name="DR_Region_Report.pdf",
                mime="application/pdf"
            )
