

# Hypothesis 3: Res-U-Net Segmentation (Task B)

This implementation explores the use of **Residual U-Net (Res-U-Net)** architecture to improve the segmentation of fetal ultrasound structures. 



##  The Hypothesis
Hypothesis 3 introduces **Residual Blocks** into the standard U-Net encoder-decoder paths. By using skip connections within the convolutional blocks themselves, we aim to mitigate the vanishing gradient problem and allow the network to learn more complex features from the ultrasound scans.

## ⚙️ Training Configuration
- **Architecture:** Res-U-Net
- **Input Size:** 256 x 256
- **Batch Size:** 8
- **Epochs:** 6
- **Loss Function:** Custom Dice Loss (with Smoothing)
- **Optimizer:** Adam ($LR = 0.0001$)

## 📁 Pipeline Overview

1. **Environment Setup:** Imports core PyTorch modules and configures the environment to use CUDA acceleration.
2. **Dataset Management:** - **FetalResDataset:** A custom class that manages image/mask pairs, performing real-time normalization and binarization.
    - **Data Splitting:** Uses an 80/20 split to ensure the model's generelization is tested on unseen data.
3. **Training Strategy:**
    - Uses a standard training loop with backpropagation.
    - Implements a validation check at the end of every epoch.
4. **Model Saving:** Only the "Best" weights are saved to the `WEIGHT_SAVE_DIR` based on the lowest recorded validation loss. This ensures the final `hypothesis_3_res_unet.pth` is the most optimized version.
5. **Progress Tracking:** Integrated with `tqdm` to provide real-time updates on training speed and loss reduction.

## 🏗️ Requirements
- PyTorch
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Tqdm



---

# Res-U-Net Inference & Evaluation and tester file 

This notebook provides the evaluation framework for the Res-U-Net model developed under Hypothesis 3. It focuses on quantifying the model's segmentation performance using standard medical imaging metrics.



## 🛠️ Features
- **Self-Contained Architecture:** The `hypothesis_3_task_2_ResUNet` is defined locally to ensure weight compatibility without external scripts.
- **Robust Evaluation:** Calculates high-level metrics including **Mean IoU** and **Dice Score** across the entire test subset.
- **Automated Reporting:** Generates a structured result summary, capturing the model's strengths and weaknesses on unseen ultrasound frames.

## 📐 Evaluation Metrics
The model is assessed based on:
1. **Intersection over Union (IoU):** Quantifies the overlap between the predicted mask and ground truth.
2. **Dice Coefficient:** Measures the similarity between two sets of pixels, emphasizing spatial overlap.
3. **Pixel-wise Accuracy:** Tracks correctly identified foreground and background pixels.



[Image of image segmentation metrics]


## 📁 Execution Workflow

1. **Initialization:** Configures the device (CUDA/CPU) and prepares the directory paths for the pre-trained weights.
2. **Weight Restoration:** Loads the `state_dict` from the training output and switches the network to `eval()` mode.
3. **Preprocessing Pipeline:** Resizes input images to 256x256 and applies standard normalization to ensure input consistency.
4. **Inference & Post-Processing:** - Executes a forward pass with gradient tracking disabled.
    - Applies a 0.5 probability threshold to generate final segmentation masks.
5. **Visualization:** Generates 4-panel comparison plots:
    - *Original Image* | *Ground Truth* | *Prediction* | *Overlay*
6. **Data Persistence:** Exports the final metric averages to a report file for peer review.

## 🏗️ Requirements
- PyTorch
- OpenCV
- Matplotlib
- Scikit-learn (for metrics)
- NumPy
- Torchvision