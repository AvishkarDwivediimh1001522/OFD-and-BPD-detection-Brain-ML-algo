# Master Report: Fetal Ultrasound Segmentation (Task B)

This project explores three different deep learning strategies for the automated segmentation of fetal ultrasound images. The goal is to produce high-accuracy masks that allow for the extraction of biometric data like Biparietal Diameter (BPD).

## The Three Hypotheses

| Model            | Architecture   | Loss Function | Purpose                                                         |
| :--------------- | :------------- | :------------ | :-------------------------------------------------------------- |
| **Hypothesis 1** | Standard U-Net | BCE           | Establishing a baseline for pixel-wise classification.          |
| **Hypothesis 2** | Standard U-Net | Dice Loss     | Optimizing for spatial overlap and boundary precision.          |
| **Hypothesis 3** | Res-U-Net      | Smooth Dice   | Using residual blocks to capture deep textures in noisy frames. |

## Why these approaches?

Medical ultrasounds are notoriously noisy. Hypothesis 1 helps us understand the data limits. Hypothesis 2 was used because standard loss functions often fail when the anatomical structure occupies only a small portion of the frame. Hypothesis 3 was introduced to allow the gradient to flow better through deeper layers, which helps the model focus on the actual fetal anatomy rather than the surrounding artifacts.

## 🛠️ Pipeline Details

1.  **Preprocessing:** All frames are normalized and resized to 256x256. Data is split into 80% training and 20% validation sets to ensure the weight files are generelized.
2.  **Training:** We use the Adam optimizer with a learning rate of $0.0001$. Training is monitored via validation loss, and only the "Best" weights are saved as `.pth` files.
3.  **Inference & Fitting:** After the model predicts a mask, we apply a binary threshold and use a custom `get_biometry_points` function to perform ellipse fitting.
4.  **Reporting:** The final output generates 4-panel plots comparing the Original Image, Ground Truth, Prediction, and the Final Overlay with biometric measurements.

## Metrics for Evaluation

We evaluate the models based on:

- **Dice Coefficient:** Measures similarity between prediction and truth.
- **Mean IoU:** Quantifies the intersection over union.
- **Pixel Accuracy:** General check of foreground/background identification.

## 🏗️ Technical Requirements

- PyTorch / Torchvision
- OpenCV (cv2)
- Scikit-learn
- Matplotlib & NumPy
- Torchinfo (for architecture inspection)

---

_Note: Ensure the dataset is organized into `/images` and `/masks` folders before initializing the trainer notebooks._
