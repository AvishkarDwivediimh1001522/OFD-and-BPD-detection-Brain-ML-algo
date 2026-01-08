#  Automated Fetal Brain Biometry from Ultrasound Images

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-brightgreen)
![Task](https://img.shields.io/badge/Task-Landmark%20%7C%20Segmentation-orange)
![Status](https://img.shields.io/badge/Status-Research--Grade-success)


> **A unified deep learning framework to estimate Biparietal Diameter (BPD) and Occipitofrontal Diameter (OFD) using fetal axial ultrasound images — via Landmark Detection *and* Segmentation-based approaches.**

---


## 📌 Project Overview (At a Glance)

Ultrasound is the safest and most widely used modality for fetal brain assessment. Accurate estimation of **BPD** and **OFD** is critical for:

* Gestational age estimation
* Monitoring fetal growth
* Early detection of CNS abnormalities

This project presents a **complete, research-grade solution** that tackles the problem using **two independent but complementary pipelines**:

1. **Task A – Landmark Detection Based Approach**
   → Directly predict 4 anatomical landmark points (2 for BPD, 2 for OFD)

2. **Task B – Segmentation Based Approach**
   → Segment the fetal cranium → fit an ellipse → derive BPD & OFD geometrically

Both approaches are implemented with **multiple hypotheses**, detailed experimentation, trained models, visualizations, and reports.

---

## 🧩 Problem Statement

Given a **fetal axial ultrasound image of the brain**, design an algorithm that can:

* Identify **4 anatomical landmark points**

  * Points **A & C** → Biparietal Diameter (BPD)
  * Points **B & D** → Occipitofrontal Diameter (OFD)

* OR segment the **cranial boundary** and compute:

  * BPD and OFD as the **major and minor axes** of the fitted ellipse

---

## 🛠 Engineering & Data Pipeline

The project follows a modular and reproducible research workflow designed for production environments:

* **Data Ingestion:** Automated loaders handle grayscale ultrasound frames and their corresponding annotation masks/coordinates.
* **The Trainer Engine:** Each hypothesis is trained via a dedicated `trainer.ipynb`. Upon completion, the engine serializes the best model state as a `.pth` file into the `Model Weights` folder.
* **The Tester Engine:** A standalone `tester.ipynb` re-imports the model architecture and weights. It executes inference, applies post-processing (Ellipse Fitting), and generates a performance report.
* **Persistence Layer:** Evaluation results are saved as `.pth` metadata and finalized as clinical-grade PDF reports within the `Report` directory.

---


##  Solution Architecture

```
Input Ultrasound Image
        │
        ├── Task A: Landmark Detection
        │      ├── CNN-based coordinate regression
        │      ├── Transfer learning + augmentation
        │      └── Heatmap-based localization
        │
        └── Task B: Segmentation
               ├── U-Net / Res-U-Net
               ├── Binary cranial mask prediction
               └── Ellipse fitting → BPD & OFD
```

---

#  Task A – Landmark Detection Based Approach

### 🎯 Objective

Predict **4 fetal cranial landmark points** directly from ultrasound images:

* 2 points → BPD
* 2 points → OFD

Each landmark is represented as an `(x, y)` coordinate.

---

## 🔬 Implemented Hypotheses

### 🔹 Hypothesis 1: Direct Coordinate Regression (ResNet18)

* Treats landmark detection as a **pure regression problem**
* Backbone: **ResNet18 (ImageNet pretrained)**
* Output: `8 continuous values` (x₁,y₁,…,x₄,y₄)
* Loss: **Mean Squared Error (MSE)**

**Pros:**

* Simple
* Fast to train
* Strong baseline

**Cons:**

* No explicit spatial constraint
* Sensitive to noise

---

### 🔹 Hypothesis 2: Transfer Learning + Data Augmentation

Enhances Hypothesis-1 by:

* Fine-tuning deeper layers
* Applying strong augmentations:

  * Rotation
  * Flip
  * Intensity & contrast jitter

**Outcome:**

* Better generalization
* Reduced overfitting
* Improved robustness on unseen images

---

### 🔹 Hypothesis 3: Heatmap-Based Landmark Localization

* Reformulates problem as **dense prediction**
* Model predicts one heatmap per landmark
* Landmark coordinates extracted from heatmap peaks

**Advantages:**

* Enforces spatial structure
* Smoother & more stable predictions
* Better visual interpretability

---

## 📊 Evaluation (Task A)

* Mean Radial Error (MRE)
* Pixel distance after rescaling
* Visual overlay of predicted landmarks

Quantitative metrics are always paired with **qualitative inspection**.

---

#  Task B – Segmentation Based Approach

### 🎯 Objective

1. Segment the **fetal cranium** from ultrasound image
2. Fit an **ellipse** to the segmentation mask
3. Compute:

   * **BPD** = minor axis
   * **OFD** = major axis

---

## 🔬 Implemented Hypotheses

### 🔹 Hypothesis 1: U-Net + Binary Cross Entropy (BCE)

* Baseline segmentation model
* Pixel-wise classification

---

### 🔹 Hypothesis 2: U-Net + Dice Loss

* Optimizes overlap between predicted & ground-truth masks
* Handles class imbalance better

---

### 🔹 Hypothesis 3: Res-U-Net (Residual U-Net)

* Introduces residual connections
* Better gradient flow
* Captures deeper anatomical patterns

---

## 🧮 Post-processing Pipeline

```
Segmentation Mask
      ↓
Thresholding
      ↓
Contour Extraction
      ↓
Ellipse Fitting (OpenCV)
      ↓
BPD & OFD Estimation
```

---

## 📊 Evaluation (Task B)

* Dice Coefficient
* Intersection over Union (IoU)
* Pixel Accuracy
* Visual comparison (GT vs Prediction vs Overlay)

---

# 🧪 Reproducibility & Usage

Each hypothesis includes:

* ✔ Trainer notebook
* ✔ Tester notebook
* ✔ Saved model weights (`.pth`)
* ✔ Visual outputs
* ✔ Detailed hypothesis-specific README

Simply open the **trainer notebook** to retrain or the **tester notebook** to reproduce results.

---

## 📊 Performance Evaluation

| Metric | Task A (Landmarks) | Task B (Segmentation) |
| :--- | :--- | :--- |
| **Primary Metric** | Mean Radial Error (MRE) | Dice Coefficient |
| **Secondary Metric** | Pixel distance (px) | IoU (Jaccard Index) |
| **Best Score** | **2.4 px** (Heatmap) | **0.88** (Res-U-Net) |

---

## 📁 Directory Structure

```text
Avishkar_Dwivedi_AI_Research_intern/
├── task_1_landmark/
│   ├── Model Weights/      # Serialized .pth weights (Hypo 1-3)
│   ├── Python Script/
│   │   └── Assets/         # Trainer and Tester Notebooks
│   ├── Report/             # Performance metadata (.pth) and PDF reports
│   └── README_task_1.md    # Task-specific documentation
│
└── task_2_segmentation/
    ├── Model Weights/      # Serialized .pth weights (Hypo 1-3)
    ├── Python Script/
    │   └── Assets/         # Trainer and Tester Notebooks
    ├── Model visualization/# Architecture plots and layer-flow
    ├── Report/             # Metrics and final clinical reports
    └── README_task_2.md    # Task-specific documentation

```
---

##  Why This Project Stands Out

* Dual independent pipelines (Landmark + Segmentation)
* Multiple hypotheses per task
* Medical imaging focused design
* Research-level structure & documentation
* Ready-to-run notebooks with trained weights

---

##  Final Note

This repository is designed so that **a single glance is enough** to understand:

* *What problem is solved*
* *How it is solved*
* *How to reproduce or extend the work*

Perfect for **research, academic evaluation, or medical AI portfolios**.

---

**Author:** Avishkar Dwivedi  
**Domain:** Medical Imaging · Deep Learning · Fetal Ultrasound
