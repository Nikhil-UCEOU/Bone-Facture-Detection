# 🩻 Bone Fracture Detection Using Vision Transformers

### 🚀 Automated & Explainable X-ray Fracture Detection System  
**By:** Eruva Nikhil, Avinash G., Suchit Boda  
**Under the guidance of:** Dr. P. V. Sudha, Department of CSE, Osmania University  
**Date:** July 2025  

---

## 🧭 Overview
This project presents an AI-powered system for **automated bone fracture detection** using **Vision Transformer (ViT)** and **Swin Transformer** architectures.

The system classifies X-ray images as **Normal** or **Fractured**, provides **Grad-CAM heatmaps** for explainability, and offers a **Streamlit-based web interface** for real-time diagnostics.

---

## 🧩 Key Highlights

| Component | Description |
|------------|-------------|
| **Dataset** | MURA-v1.1 (Musculoskeletal Radiographs) |
| **Models** | Vision Transformer (ViT), Swin Tiny, Swin Super |
| **Learning Type** | Binary classification (Normal vs Abnormal) |
| **Explainability** | Grad-CAM heatmaps for clinical transparency |
| **Deployment** | Streamlit app (PyTorch → ONNX → Quantized INT8) |
| **Efficiency Techniques** | Pruning · Knowledge Distillation · Quantization |
| **Output** | Model prediction · Confidence · Heatmap · PDF report |

---

## 🧱 Project Architecture
<code>
BoneFractureDetection/
│
├── app/
│   ├── app_streamlit.py          # Streamlit UI (PyTorch inference + Grad-CAM)
│   ├── app_onnx.py               # Streamlit UI (ONNX quantized inference)
│   └── assets/                   # CSS, icons, demo images
│
├── scripts/
│   ├── data_loader.py            # Preprocessing, label mapping, class weighting
│   ├── train_teacher.py          # Swin Super (Teacher) training
│   ├── train_distill.py          # Knowledge distillation (Student ← Teacher)
│   ├── prune_and_finetune.py     # Pruning redundant weights
│   ├── quantize_and_export.py    # ONNX export + INT8 quantization
│   ├── bbox_wrapper.py           # Bounding-box localization
│   └── cam_utils.py              # Grad-CAM visualization utilities
│
├── models/                       # .pth / .onnx (tracked via Git LFS / DVC)
│
├── data/                         # Placeholder for MURA dataset (ignored in Git)
│
├── docs/
│   ├── report.pdf                # Final project report
│   └── figures/                  # Architecture + Grad-CAM visuals
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── CODE_OF_CONDUCT.md            # Contributor behavior policy
├── LICENSE                       # Open-source license (MIT)
└── .gitignore                    # Ignored files and directories

</code>

---

## ⚙️ Training & Optimization Pipeline

1. **Teacher Model (Swin Super)**
   - Trained using class-weighted cross-entropy  
   - Balanced dataset labeling (positive = Abnormal, negative = Normal)

2. **Student Model**
   - Learned via knowledge distillation from the teacher  
   - Combined weighted CE + KL loss for stability and generalization

3. **Pruning & Quantization**
   - `apply_unstructured_pruning()` reduces redundant weights  
   - Quantized (INT8) for faster inference on CPUs

4. **ONNX Export**
   - Lightweight, production-ready model for deployment  
   - Fully compatible with `onnxruntime` and Streamlit apps

---

## 🧠 Explainable AI (Grad-CAM)

Grad-CAM visualizations reveal the exact bone regions influencing predictions, ensuring medical transparency and trustworthiness.

*Example Heatmap Output:*  
> Red regions → Fracture focus areas  
> Blue regions → Normal tissue  

---

## 💻 Streamlit Application Features
- 🩻 Upload any X-ray image (JPG/PNG)  
- 🧠 View predictions + confidence scores  
- 🔥 Visualize Grad-CAM heatmaps  
- 🕒 Compare ViT / Swin / Swin Super  
- 📄 Download auto-generated PDF reports  

---

## 📈 Results Summary

| Model | Accuracy | Confidence | Inference Speed |
|--------|-----------|-------------|----------------|
| **ViT** | Good | Medium | Moderate |
| **Swin Tiny** | Better | High | Fast |
| **Swin Super (Teacher)** | Excellent | Very High | Slightly Slower |
| **Distilled Student** | Near-Teacher Accuracy | High | **Fastest ✅** |

---

## 🌐 Future Enhancements
- Multi-class bone detection ( wrist · elbow · hand )
- Severity grading (minor/major)
- Bounding-box fracture localization (YOLO / DETR)
- PACS/RIS integration for hospitals
- Lightweight mobile app via ONNX Runtime Mobile

---

## 🪪 License
Released under the **MIT License** — see [`LICENSE`](LICENSE).

---

## 🤝 Contributing
Contributions are welcome!  
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines and setup instructions.

---

## 🧑‍⚖️ Code of Conduct
All contributors are expected to abide by our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## 📫 Contact
**Eruva Nikhil**  B.Tech AI & ML, Osmania University  
**Email:** <eruvaniku@gmail.com>

---
