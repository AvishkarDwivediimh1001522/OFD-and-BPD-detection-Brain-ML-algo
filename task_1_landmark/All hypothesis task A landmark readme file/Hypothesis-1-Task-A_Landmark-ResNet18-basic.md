# Task A: Fetal Landmark Detection (ResNet18 Regression)

This folder contains the training and testing pipelines for Hypothesis 1, focusing on direct coordinate regression to identify fetal biometric landmarks.

##  Hypothesis 1: ResNet18 Backbone

For this approach, we leverage a pre-trained ResNet18 model as a feature extractor, modifying the output head to predict 8 continuous values representing 4 sets of $(x, y)$ coordinates.

## 📁 Notebook 1: Trainer

The `trainer.ipynb` file handles the model optimization and weight persistence.

- **Data Scaling:** Coordinates are normalized by image dimensions to ensure the loss function treats all scans equally regardless of resolution.
- **Preprocessing:** Standardizes input to 224x224 pixels with ImageNet-based mean and std dev normalization.
- **Optimization:** Uses the Adam optimizer and Mean Squared Error (MSE) loss, which is ideal for this kind of coordinate regression task.
- **Output:** Saves the finalized state dictionary to Google Drive as `hypothesis_1_full_saved_mode.pth`.

---

## 📁 Notebook 2: Tester

The `tester.ipynb` file is used to validate the model on unseen data and visualize the predicted points.

- **Weight Verification:** Includes a diagnostic block to inspect the `.pth` file, ensuring all 122 weight layers (including the custom FC layer) are intact.
- **Inference Pipeline:** Performs the forward pass and applies the inverse scaling logic to map predictions back to the original pixel space.
- **Visual Validation:** Generates scatter plots over the raw ultrasound images:
  - **Lime Green:** BPD (Biparietal Diameter) landmarks.
  - **Yellow:** OFD (Occipitofrontal Diameter) landmarks.

## 🏗️ Requirements

- PyTorch / Torchvision
- OpenCV (cv2)
- Pandas (for CSV parsing)
- Matplotlib
- PIL (Pillow)
