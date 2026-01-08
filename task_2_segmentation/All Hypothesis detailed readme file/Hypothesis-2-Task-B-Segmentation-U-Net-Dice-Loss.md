

# Hypothesis 2: U-Net + Dice Loss Segmentation (Task B)

This repository contains the training implementation for the second hypothesis of Task B, focusing on improving fetal ultrasound segmentation using the Dice Loss metric.



##  The Approach
While Hypothesis 1 used standard cross-entropy, **Hypothesis 2** implements a custom **Dice Loss** function. This is specifically chosen to maximize the spatial overlap between the predicted segmentation and the actual anatomy, which is crucial for accurate fetal biometry.

##  Configuration & Hyperparameters
- **Input Resolution:** 256x256
- **Batch Size:** 16 (adjustable based on VRAM)
- **Loss Function:** Custom Dice Loss
- **Optimizer:** Adam
- **Data Split:** 80% Training / 20% Validation

## 📁 Implementation Details

1. **Dice Loss Module:** A dedicated `nn.Module` class that calculates the intersection over union to provide a more robust gradient for mask boundaries.
2. **U-Net Architecture:** High-resolution features are preserved using skip connections between the contracting and expansive paths.
3. **Data Loading:** The `FetalDatasetH2` class handles image normalization and ensures that masks are processed as binary tensors.
4. **Checkpointing:** The training loop includes a "Best Model" saver that monitors validation performance to prevent overfitting.
5. **Integrity Check:** Post-training scripts verify the `state_dict` to ensure the model weights are saved correctly for the tester notebook.

##  Getting Started
1. Ensure your dataset is structured with `/images` and `/masks` folders.
2. Update the paths in the `Configuration` cell of `trainer.ipynb`.
3. Run all cells to begin training. The best weights will be saved as `hypothesis_2_segmentation.pth`.

## 🏗️ Requirements
- Python 3.x
- PyTorch / Torchvision
- OpenCV
- Scikit-learn
- Matplotlib


---

# Biometric Inference & Ellipse Fitting and tester file work 

This notebook serves as the evaluation pipeline for the Hypothesis 2 segmentation model. It transitions from raw pixel-wise prediction to clinical biometric measurement extraction.



##  Operational Flow

- **Model Reconstruction:** Re-implements the `hypothesis_2_task_2_UNet` to ensure the notebook is self-contained for easy portability.
- **Weight Integration:** Loads the pre-trained weights optimized via Dice Loss to ensure high-overlap segmentation accuracy.
- **Structural Inspection:** Uses `torchinfo` to validate the layer-by-layer tensor flow and parameter distribution.

## 📏 Biometric Extraction Algorithm
The core of this tester is the `extract_biometry_points` function. It performs the following steps:
1. **Contour Detection:** Identifies the boundaries of the predicted fetal structure.
2. **Ellipse Fitting:** Uses least-squares optimization to fit a geometric ellipse to the segmentation.
3. **Metric Calculation:** Extracts the Biparietal Diameter (BPD) and Occipitofrontal Diameter (OFD) based on the major and minor axes of the fitted shape.

## 📁 Inference & Reporting
The notebook processes input images and generates a visual and textual report:
- **Visualization:** Side-by-side comparison of the original ultrasound and the model's segmentation with the biometric overlay.
- **Reporting:** Saves a final output file to the `/Report` directory containing the calculated diameters and model metadata.

## 🏗️ Requirements
- PyTorch / Torchvision
- OpenCV (cv2)
- Torchinfo
- Matplotlib
- NumPy