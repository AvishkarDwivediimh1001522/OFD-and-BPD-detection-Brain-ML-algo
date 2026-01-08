

# Hypothesis 3: Landmark Detection via ResNet18 HeatMap Regression

This project implements a heatmap-based approach for identifying anatomical landmarks in fetal ultrasound images, utilizing a ResNet18 backbone for feature extraction.



##  The Approach
Instead of traditional coordinate regression (which can be unstable), Hypothesis 3 uses **HeatMap Regression**. The model predicts a 2D Gaussian distribution centered at the landmark location. This provides a spatial context that helps the model converge faster and more accurately.



## ⚙️ Configuration & Environment
- **Backbone:** Pre-trained ResNet18
- **Input Resolution:** 224 x 224
- **Heatmap Output:** 56 x 56
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metric:** Mean Radial Error (MRE)
- **Optimizer:** Adam ($LR = 1e-4$)
- **Hardware:** CPU-based execution

## 📁 Implementation Workflow

1. **Data Engineering:** - The `FetalHeatmapDataset` class transforms raw landmark coordinates from a CSV into 2D heatmaps.
    - Standardizes image inputs using normalization and resizing.
2. **Model Design:**
    - Leverages ResNet18 for deep feature extraction.
    - A custom decoder upsamples the feature maps to produce the final heatmap predictions.
3. **Training Protocol:**
    - **Backpropagation:** Computed via MSE loss between the predicted and ground-truth heatmaps.
    - **Validation:** Every epoch includes an MRE calculation, converting heatmaps back to $(x, y)$ coordinates to measure average pixel distance error.
4. **Checkpointing:**
    - The final model weights are persisted as `hypothesis_3_full_saved_mode.pth` in the designated Drive folder for easy access during the inference phase.

## 🏗️ Requirements
- PyTorch / Torchvision
- Pandas
- OpenCV (cv2)
- Pillow (PIL)
- Matplotlib
- NumPy

---

# HeatMap Inference & Clinical Reporting and tester file (Task 1)

This notebook handles the evaluation and automated report generation for the ResNet18-based landmark detection model. It converts abstract heatmap predictions into actionable biometric coordinates.



## 🛠️ Operational Workflow

- **Coordinate Reconstruction:** Features a `get_coords_from_hm` algorithm that identifies peak activations in the 56x56 heatmaps and scales them back to the original image resolution.
- **Backbone Integration:** Re-instantiates the ResNet18 + Decoder architecture to load the trained `hypothesis_3_full_saved_mode.pth` weights.
- **Device Flexibility:** Though configured for CPU, the script is written to be easily toggled for CUDA-based batch inference.

## 📊 Landmark Visualization
The notebook provides a visual validation suite:
1. **Heatmap Generation:** Displays the raw probability maps produced by the model.
2. **Point Overlay:** Draws the final predicted $(x, y)$ landmarks on the original ultrasound scan for manual review.



## 📝 Automated PDF Reporting
Unlike previous hypotheses, this notebook includes a professional reporting module using the `fpdf` library:
- **Project Metadata:** Includes the Hypothesis ID, model architecture details, and configuration settings.
- **Results Summary:** Documents the performance and saves the finalized weights into a dedicated `/Report` directory.
- **Visual Evidence:** Optionally attaches inference samples to the generated document.

## 🏗️ Requirements
- PyTorch / Torchvision
- FPDF (for PDF generation)
- OpenCV (cv2)
- Pandas
- Matplotlib
- PIL (Pillow)