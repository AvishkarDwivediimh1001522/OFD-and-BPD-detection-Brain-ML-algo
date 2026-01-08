# Hypothesis 2: ResNet18 Transfer Learning & Data Augmentation

This repository implements a landmark regression pipeline using a pre-trained ResNet18 backbone. This approach focuses on leveraging existing feature extractors and robust data augmentation to accurately locate anatomical points in fetal ultrasound scans.

## Approach & Methodology

Hypothesis 2 utilizes **Transfer Learning**. By freezing the convolutional layers of a ResNet18 model trained on ImageNet, we utilize its ability to recognize complex textures and edges, only training a custom fully-connected "head" for our specific coordinate regression task.

## ⚙️ Configuration

- **Backbone:** ResNet18 (Pre-trained)
- **Trainable Layers:** Final Fully Connected (FC) Layer only
- **Input Size:** 224 x 224
- **Augmentation:** Random Rotation, Color Jitter, and Normalization
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** 16
- **Epochs:** 25

## 📁 Pipeline Workflow

1. **Dataset Engineering:** Coordinates are normalized to a [0, 1] range to stabilize the gradient updates during training.
2. **Data Augmentation:**
   - **Training:** Implements `ColorJitter` and `RandomRotation` to simulate the variability found in clinical ultrasound environments.
   - **Validation:** Standard resizing and normalization ensure a consistent evaluation baseline.
3. **Training Protocol:**
   - The Adam optimizer is restricted to the parameters of the new FC layer.
   - Performance is monitored using **Mean Radial Error (MRE)** to measure the average pixel distance between predictions and ground truth.
4. **Hardware Acceleration:** Supports seamless switching between CUDA and CPU environments.
5. **Model Persistence:** The final state dictionary is exported to `hypothesis_2_weights.pth` for downstream testing.

## 📊 Evaluation

The primary success metric for this hypothesis is the minimization of the MRE on the validation set, ensuring that the landmarks are localized with high spatial precision.

## 🏗️ Requirements

- PyTorch / Torchvision
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

# ResNet18 Landmark Inference & Fitting Analysis & tester file

This notebook contains the testing and validation suite for the Transfer Learning-based landmark regression model (Hypothesis 2). It focuses on coordinate restoration and statistical error analysis.

## 🛠️ Key Functionalities

- **Coordinate Inverse Scaling:** Automatically maps the model's normalized [0, 1] outputs back to the original ultrasound pixel dimensions for precise measurement.
- **Biometric Visualization:** Draws color-coded landmarks (Green for BPD, Yellow for OFD) to visually assess the model's anatomical understanding.
- **Ground Truth Comparison:** Provides a side-by-side visual overlay of predicted points versus manually labeled ground truth.

## Performance Metrics

The primary evaluation metric used in this tester is the **Mean Radial Error (MRE)**.

- The script calculates the Euclidean distance between each predicted $(x, y)$ pair and the corresponding ground truth coordinate.
- Results are aggregated into a **Fitting Efficiency** bar graph, showing the pixel-wise deviation across a random sample of the test dataset.

## 📁 Evaluation Workflow

1. **Model Loading:** Instantiates the `LandmarkRegressor` and loads the pre-trained ResNet18 weights from the `.pth` file.
2. **Preprocessing:** Standardizes test images to 224x224 and applies ImageNet normalization.
3. **Inference Loop:** - Executes the forward pass on a batch of test images.
   - Scales the results to the original image resolution.
4. **Statistical Analysis:** Compares predictions against CSV-based labels to determine the accuracy of the landmark localization.
5. **Reporting:** Generates visual plots and a bar graph summarizing the average error in pixels.

## 🏗️ Requirements

- PyTorch / Torchvision
- OpenCV (cv2)
- Pandas (for ground truth CSV handling)
- Matplotlib (for error analysis graphs)
- NumPy
- PIL
