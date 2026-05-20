# 🩺 AI-Based Diabetic Retinopathy Grading using Deep Learning

An AI-powered deep learning application for detecting and grading **Diabetic Retinopathy (DR)** from retinal fundus images using **EfficientNet-B0 + CBAM Attention Mechanism** with Grad-CAM explainability and PDF clinical report generation.

---

# 📌 Project Overview

Diabetic Retinopathy is a serious eye disease caused by diabetes that may lead to vision loss if not detected early.

This project automates DR severity classification using deep learning and provides interpretable visual explanations for clinical support.

The system:

- Classifies retinal images into DR severity stages
- Highlights affected retinal regions using Grad-CAM
- Generates downloadable PDF clinical reports
- Includes doctor authentication using Streamlit

---

# 🚀 Features

## ✅ DR Severity Classification

The model classifies retinal images into:

- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

---

## ✅ Deep Learning Model

- EfficientNet-B0 backbone
- CBAM (Convolutional Block Attention Module)

---

## ✅ Grad-CAM Visualization

- Highlights disease-affected retinal regions
- Improves model interpretability
- Helps clinical understanding

---

## ✅ Clinical PDF Report Generation

Automatically generates:

- Prediction summary
- Confidence score
- Region interpretation
- Medical disclaimer
- Downloadable PDF report

---

## ✅ Doctor Authentication Dashboard

- User Registration
- Login Authentication
- Session Management
- Restricted Access

---

## ✅ Interactive Streamlit Web App

Built using Streamlit for:

- Real-time prediction
- Visualization
- PDF report generation

---

# 🏗️ System Architecture

```text
Retinal Fundus Image
        │
        ▼
Image Preprocessing
        │
        ▼
EfficientNet-B0 Backbone
        │
        ▼
CBAM Attention Module
        │
        ▼
Classification Layer
        │
        ▼
DR Severity Prediction
        │
        ▼
Grad-CAM Visualization
        │
        ▼
PDF Clinical Report
```

---

# 🛠️ Tech Stack

## 🔹 Programming Language
- Python

## 🔹 Frameworks & Libraries

- PyTorch
- TorchVision
- Streamlit
- OpenCV
- NumPy
- PIL
- timm
- ReportLab

---

# 📂 Project Structure

```bash
AI-Based-Retinopathy/
│
├── app.py
├── data_loader.py
├── clean_dataset.py
├── DR_Preprocessing.ipynb
├── evaluate.ipynb
├── 06_Stastical_analysis.ipynb
├── efficientnet_cbam_best.pth
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

The project uses retinal fundus image datasets for diabetic retinopathy grading.

## Dataset preprocessing includes:

- Image resizing
- Normalization
- Corrupted image removal
- Data transformation
- Dataset splitting

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Based-Retinopathy.git
cd AI-Based-Retinopathy
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run the Application

```bash
streamlit run app.py
```

Open in browser:

```bash
http://localhost:8501
```

---

# 🧠 Model Architecture

## 🔹 EfficientNet-B0

EfficientNet-B0 is used as the backbone CNN model for feature extraction because of:

- High accuracy
- Lightweight architecture
- Better efficiency

---

## 🔹 CBAM Attention Module

The Convolutional Block Attention Module (CBAM) improves feature learning using:

- Channel Attention
- Spatial Attention

This helps the model focus on important retinal regions.

---

## 🔹 Classification Layer

The extracted features are passed through:

- Adaptive Average Pooling
- Fully Connected Layer
- Softmax Activation

for final 5-class DR prediction.

---

# 📈 Workflow

## Step 1: Upload Retinal Image

Doctor uploads a retinal fundus image.

## Step 2: Image Preprocessing

- Resize image
- Normalize image
- Convert to tensor

## Step 3: Model Prediction

Deep learning model predicts DR severity.

## Step 4: Grad-CAM Visualization

Model highlights affected retinal regions.

## Step 5: Generate PDF Report

Clinical report is generated automatically.

---

# 📷 Grad-CAM Explainability

Grad-CAM visualization helps identify:

- Macular Region
- Optic Disc Region
- Lesion-Suspected Regions

This improves AI transparency and interpretability.

---

# 📄 PDF Clinical Report

The generated report contains:

- DR severity prediction
- Confidence score
- Region interpretation
- AI-generated clinical summary
- Medical disclaimer

---

# 📊 Results

## ✔️ Achievements

- Accurate DR classification
- Attention-enhanced learning using CBAM
- Explainable AI visualization
- Real-time prediction support
- Automated clinical reporting

---

# 🔒 Authentication System

The application includes:

- Doctor registration
- Secure login system
- Session authentication
- Restricted dashboard access

---

# 🌟 Future Enhancements

- Cloud deployment
- Mobile application
- Real-time hospital integration
- Improved model accuracy
- Multi-language support
- Electronic Health Record (EHR) integration

---

# 👨‍💻 Author

## Dhiraj Kumar

AI/ML Engineer & Data Analyst

📧 Email: dhrjk128@gmail.com  
🔗 LinkedIn: https://linkedin.com/in/DhirajKumar

---

# 🙏 Acknowledgements

Special thanks to:

- PyTorch Community
- Streamlit
- OpenCV
- Medical AI Research Community
- Open-source contributors

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub.

---

# 📜 License

This project is developed for educational and research purposes only.
