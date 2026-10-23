# 🍅 TomatoAI: Autonomous Tomato Leaf Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![TFLite](https://img.shields.io/badge/TFLite-Edge%20Ready%20(200KB)-green.svg)](https://www.tensorflow.org/lite)
[![Paper](https://img.shields.io/badge/IEEE%20Format-Research%20Paper%20(4.4MB)-purple.svg)](docs/Tomato_Disease_Detection_Research_Paper.pdf)
[![Handbook](https://img.shields.io/badge/Team-Master%20Handbook-teal.svg)](docs/Tomato_Disease_Project_Team_Handbook.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **B.Tech 4th Year Major Project & Research Study**  
> *Empirical Evaluation of Deep Transfer Learning and Custom CNN Architectures for Automated Tomato Leaf Disease Classification*

---

## 📖 Executive Summary

Tomato (*Solanum lycopersicum*) is India's second largest vegetable crop with an annual market valuation exceeding **$2.1 Billion**. However, foliar diseases—principally **Late Blight**, **Bacterial Spot**, and **Tomato Yellow Leaf Curl Virus (TYLCV)**—routinely inflict **70% to 80% localized yield destruction**. In developing rural regions, certified agricultural extension officers are scarce, causing smallholder farmers to frequently misdiagnose diseases and over-apply chemical pesticides.

**TomatoAI** provides an autonomous, computer-vision-based agricultural diagnostic system trained on the 10 tomato pathology classes of the PlantVillage benchmark (**19,160 leaf images**). We performed a rigorous empirical comparative study between:
1. **Custom 4-Block CNN** trained from scratch with Batch Normalization and Dropout.
2. **MobileNetV2 Transfer Learning** leveraging inverted residual bottlenecks and depthwise separable convolutions.

Under identical computational constraints, **MobileNetV2 achieved 69.75% test accuracy and a 0.7117 macro F1-score**, outperforming the custom CNN by **+23.91%** while converging **3.1× faster**. The champion model was quantized into an ultra-compact **200 KB TensorFlow Lite (`.tflite`)** flatbuffer for zero-latency offline edge inference on low-cost Android smartphones and IoT devices.

---

## 🌟 Key Project Highlights

- 🎯 **Specialized 10-Class Tomato Taxonomy:** Unlike multi-crop models that achieve artificial accuracy by distinguishing distinct crops, TomatoAI focuses exclusively on differentiating subtle, overlapping foliar blights within tomato plants.
- ⚡ **Dual Architecture Benchmark:** Rigorous comparison between monolithic convolutional networks and depthwise separable transfer learning.
- 📊 **10 Empirical Evaluation Plots:** Complete telemetry including Class Distribution, Foliar Atlas, Training/Loss Curves, Confusion Matrices, Per-Class Precision/Recall/F1, and KPI Dashboards.
- 📱 **Edge & IoT Deployment:** Exported 200 KB TensorFlow Lite model for offline rural use without cellular or internet connectivity.
- 💻 **Interactive 3-Tab Streamlit App:** Real-time leaf diagnosis, curated test sample picker, interactive 10-plot telemetry gallery, and in-app research paper viewer.
- 📄 **Complete Publication Documentation:** Formally formatted 4.4 MB IEEE research paper and comprehensive Team Master Handbook ready for project approval committees and journal submissions.

---

## 🔬 Dataset Taxonomy (10 Classes | 19,160 Images)

| Class ID | Pathology / Foliar Condition | Causal Pathogen | Original Images | Taxonomic Group |
|:---:|:---|:---|:---:|:---:|
| `0` | **Bacterial Spot** | *Xanthomonas campestris* | 2,127 | Bacterial |
| `1` | **Early Blight** | *Alternaria solani* | 1,000 | Fungal |
| `2` | **Late Blight** | *Phytophthora infestans* | 1,909 | Oomycete (Critical) |
| `3` | **Leaf Mold** | *Passalora fulva* | 952 | Fungal (Polyhouse) |
| `4` | **Septoria Leaf Spot** | *Septoria lycopersici* | 1,771 | Fungal |
| `5` | **Spider Mites (Two-spotted)** | *Tetranychus urticae* | 1,676 | Arachnid Pest |
| `6` | **Target Spot** | *Corynespora cassiicola* | 1,404 | Fungal |
| `7` | **Yellow Leaf Curl Virus (TYLCV)** | *Begomovirus* (Whitefly-borne) | 5,357 | Viral |
| `8` | **Tomato Mosaic Virus (ToMV)** | *Tobamovirus* (Contact-borne) | 373 | Viral |
| `9` | **Healthy Leaf** | None (Control Baseline) | 1,591 | Physiological |
| — | **Total** | | **19,160** | |

### Data Augmentation Pipeline:
To resolve class imbalance (TYLCV contains 5,357 images vs Mosaic Virus with 373 images), synthetic data augmentation was integrated on the fly:
- Random horizontal reflection ($p = 0.5$)
- Random rotation ($\pm 20^\circ$)
- Zoom scaling ($0.85 - 1.15$)
- Rescaling normalization ($[0, 1]$ min-max)

---

## 📊 Quantitative Experimental Results

```
========================================================================================
  METRIC                        CUSTOM 4-BLOCK CNN       MOBILENETV2 (TL)     SUPERIORITY
========================================================================================
  Top-1 Test Accuracy           45.84%                   69.75%               +23.91% 🏆
  Test Loss (Cross-Entropy)     4.5074                   1.0142               3.4× lower error
  Macro Precision               72.02%                   77.34%               +5.32%
  Macro Recall                  46.72%                   70.65%               +23.93%
  Macro F1-Score                0.4608                   0.7117               +54.4% relative gain
  Training Time (CPU)           11.9 minutes             3.8 minutes          3.1× faster
  TFLite Model Size             2.11 MB                  0.20 MB (200 KB)     10.5× more compact
========================================================================================
```

### Why Did MobileNetV2 Outperform Custom CNN?
Standard convolutions compute spatial filtering and channel mixing simultaneously with computational cost $\mathcal{O}(D_k \cdot D_k \cdot M \cdot N \cdot D_f \cdot D_f)$. In contrast, **depthwise separable convolutions** split computation into:
1. **$3\times3$ Depthwise Convolution:** Single filter applied per input channel for spatial patterns.
2. **$1\times1$ Pointwise Convolution:** Linear combination of channels.

This reduces FLOPs by **~8.5× to 9×**, preserving representational capacity while eliminating gradient dissipation on limited training epochs.

---

## 🖼️ 10 Evaluation Plots Gallery

All plots are generated during execution and stored in [`tomato_output/plots/`](tomato_output/plots/):

1. **`01_class_distribution.png`**: Class taxonomy breakdown and frequency distribution.
2. **`02_sample_disease_images.png`**: 2×5 Foliar pathology visual atlas of all 10 conditions.
3. **`03_cnn_training_curves.png`**: Custom CNN training and validation accuracy/loss trajectory.
4. **`04_mnv2_training_curves.png`**: MobileNetV2 convergence and loss reduction curves.
5. **`05_cnn_confusion_matrix.png`**: Confusion matrix for custom CNN (raw counts & normalized).
6. **`06_mnv2_confusion_matrix.png`**: Confusion matrix for MobileNetV2 displaying strong diagonal density.
7. **`07_cnn_per_class_metrics.png`**: Per-class precision, recall, and F1 breakdown for custom CNN.
8. **`08_mnv2_per_class_metrics.png`**: Per-class precision, recall, and F1 breakdown for MobileNetV2.
9. **`09_model_comparison.png`**: Multi-panel KPI comparison dashboard.
10. **`10_comparison_curves.png`**: Side-by-side convergence overlay proving transfer learning stability.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/UjjwalSingh03/Early-Plant-Disease-Detection-System.git
cd Early-Plant-Disease-Detection-System
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Launch the Streamlit Web Application
```bash
cd app
streamlit run tomato_app.py
```
*(Alternative if streamlit is not in your system PATH: `python3 -m streamlit run tomato_app.py`)*

Open your browser at `http://localhost:8501`.

---

## 💻 Web Application Features

The interactive web application (`app/tomato_app.py`) provides 3 dedicated tabs:

1. **🌿 Disease Diagnosis & Inference:**
   - Upload custom leaf photos or pick from 10 pre-loaded test samples.
   - Top-3 diagnostic probability distribution with confidence progress bars.
   - Agronomic dossier with pathogen taxonomy, disease severity index, and chemical/biological treatment protocols.
2. **📊 Model Analytics & 10 Evaluation Plots:**
   - Full KPI performance comparison matrix.
   - Interactive high-resolution viewer for all 10 evaluation plots.
3. **📄 Research Paper & IEEE Specifications:**
   - In-app preview of the complete research paper.
   - One-click downloads for the **Compiled IEEE PDF Paper** and the **Team Master Handbook**.
   - Committee Viva Voce technical defense talking points.

---

## 📱 Edge Deployment with TensorFlow Lite

For rural edge deployment on low-cost Android smartphones or IoT camera traps (e.g., Raspberry Pi in greenhouses):
```python
import tensorflow as tf
import numpy as np

# Load the 200 KB quantized TFLite model
interpreter = tf.lite.Interpreter(model_path="tomato_output/models/tomato_mobilenetv2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run offline inference in < 45ms
# input_data is a preprocessed (1, 64, 64, 3) normalized float32 numpy array
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(predictions[0])
```

---

## 📂 Repository Architecture

```
Early-Plant-Disease-Detection-System/
├── app/
│   ├── tomato_app.py                       # Main Streamlit web application (3 tabs)
│   ├── requirements_tomato.txt             # App-specific runtime requirements
│   ├── class_indices.json                  # Class index mappings
│   └── trained_model/                      # Baseline model links
├── docs/
│   ├── Tomato_Disease_Detection_Research_Paper.pdf  # 4.4 MB IEEE publication paper
│   ├── Tomato_Disease_Project_Team_Handbook.pdf     # 4.4 MB Team onboarding dossier
│   ├── Research_Paper_Tomato_Disease_Detection_IEEE.md # Markdown paper source
│   ├── PROJECT_APPROVAL_COMMITTEE_DOSSIER.md       # Committee presentation dossier
│   ├── generate_research_paper_pdf.py              # PDF compiler for research paper
│   └── generate_team_handbook_pdf.py               # PDF compiler for team handbook
├── notebooks/
│   ├── Tomato_Disease_Detection_Complete.py        # Complete ML training pipeline
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb # Jupyter notebook
├── test_samples/
│   └── tomato/                             # 10 Curated test samples (1 per class)
├── tomato_output/
│   ├── models/
│   │   ├── tomato_mobilenetv2_model.h5     # Trained MobileNetV2 model
│   │   ├── tomato_mobilenetv2.tflite       # 200 KB ultra-compact edge model
│   │   └── tomato_cnn_model.h5             # Custom 4-block CNN baseline
│   ├── plots/                              # All 10 high-resolution evaluation plots
│   ├── reports/
│   │   └── final_summary.json              # Empirical metrics summary JSON
│   └── tomato_class_indices.json           # Tomato class taxonomy dictionary
├── .gitignore                              # Comprehensive exclusions for datasets & secrets
├── PROJECT_PHASE_TRACKER.md                # Phase checkpointing & progress tracker
├── README.md                               # Project documentation (This file)
└── requirement.txt                         # System-wide Python dependencies
```

---

## 🎯 Viva Voce / Project Committee Defense Points

| Question | Core Technical Defense |
|---|---|
| **Why focus strictly on Tomato?** | Multi-crop models distinguish plant species rather than subtle lesion patterns. A tomato farmer needs to separate Early Blight from Septoria and Bacterial Spot, which share similar brown spots. Narrowing to Tomato solves the real agricultural problem. |
| **Why did MobileNetV2 outperform Custom CNN?** | MobileNetV2 uses depthwise separable convolutions that factorize spatial filtering from channel combination. Combined with pre-trained ImageNet weights, it transfers rich generalized edge features without requiring massive epochs. |
| **How does this deploy in areas with no internet?** | The model was quantized into a **200 KB TensorFlow Lite flatbuffer** that runs locally on basic Android phones in under 45ms with zero cellular connectivity required. |
| **How was overfitting prevented?** | Data augmentation (rotation, flip, zoom), Batch Normalization, Dropout (0.4–0.5), EarlyStopping (patience=5), and ReduceLROnPlateau learning rate decay. |

---

## 👥 Contributors & Acknowledgements

- **Shubham Jaiswal** — Department of Computer Science & Engineering
- **Ujjwal Singh** — Project Collaborator & Maintainer
- Dataset: **PlantVillage Project** (David P. Hughes, Marcel Salathé et al.)
- Built with **TensorFlow / Keras**, **Streamlit**, and **ReportLab**.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
