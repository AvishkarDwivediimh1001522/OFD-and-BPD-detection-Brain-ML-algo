# Task A – Fetal Landmark Detection (Landmark Based Approach)

This repository contain the complete implementation of **Task A – Landmark Detection** using three different hypothesis.  
All the hypothesis are implemented, trained and tested separately but follow the same dataset structure and evaluation logic.

The goal of Task A is to **detect 4 fetal biometric landmarks (8 coordinate values)** from ultrasound images.  
Each hypothesis explore a differnt modeling idea to understand the effect of architecture choice and learning strategy.

---

## Dataset and Problem Definition

The dataset consist of fetal ultrasound images with manually annotated landmark points.  
Each image has **4 landmark points**, stored as `(x, y)` coordinate pairs.

Before training, all coordinates are **scaled between 0 and 1** with respect to image width and height.  
This help the model to learn independent of absolute image resolution which sometimes vary.

Images are resized to **224 × 224** and normalized using ImageNet statistics for better convergence.

---

## Hypothesis 1 – Direct Coordinate Regression (ResNet18)

### Idea

In Hypothesis-1, landmark detection is treated as a **pure regression problem**.  
A pretrained **ResNet18** backbone is used and the final classification layer is replaced with a fully connected head that predict **8 continuous values**.

This approach is simple and fast, but sometimes sensitive to noisy predictions because no spatial constraint is applied.

### Model Architecture

- Backbone: ResNet18 (ImageNet pretrained)
- Output layer: Fully connected layer with 8 neurons
- Activation: Linear (no activation at output)
- Loss: Mean Squared Error (MSE)

### Training Pipeline

- Input images resized to `224×224`
- Landmark coordinates normalized
- Optimizer: Adam
- Loss function: MSE loss
- Weights saved as `.pth` file after training

The trainer notebook handles full optimization and weight saving for later testing.

### Testing Pipeline

- Loads saved `.pth` weights
- Verifies model weight integrity
- Runs inference on unseen images
- Scales predicted landmarks back to original pixel space
- Visualize predicted points over ultrasound images

Different landmark types are shown using different colors so errors are clearly visible.

This hypothesis provide a strong baseline and is easy to debug and understand. :contentReference[oaicite:1]{index=1}

---

## Hypothesis 2 – Transfer Learning + Data Augmentation

### Idea

Hypothesis-2 improves upon the first approach by adding **transfer learning and data augmentation** together.  
The backbone is still pretrained, but deeper layers are fine-tuned gradually.

Augmentation is used to improve generalization and reduce overfitting on small medical datasets.

### Key Improvements

- Random rotation and flipping
- Intensity jitter and contrast changes
- Partial layer unfreezing during training
- Slightly lower learning rate

These changes help the model learn more robust landmark positioning even when ultrasound quality is not perfect.

### Training Strategy

- Initial epochs: backbone frozen
- Later epochs: last convolution blocks unfrozen
- Same coordinate regression objective
- Weights saved separately for this hypothesis

Sometimes training become bit unstable in early epochs but stabilize later.

### Output Behavior

Compared to Hypothesis-1:

- Better generalization
- Reduced overfitting
- More stable predictions near image edges

This approach usually give better validation accuracy but take more time to train.

---

## Hypothesis 3 – Heatmap Based Landmark Localization

### Idea

In Hypothesis-3, the problem is reformulated as a **heatmap prediction task** instead of direct coordinate regression.  
The model predict a separate heatmap for each landmark where the peak indicate landmark location.

Coordinates are extracted from heatmap peaks during inference.

### Model Design

- Encoder-decoder style CNN
- Output shape: `(num_landmarks, H, W)`
- Loss: MSE or Smooth L1 on heatmaps
- Gaussian heatmaps generated from GT points

This approach enforce spatial structure and usually give smoother predictions.

### Advantages

- Better spatial consistency
- Robust against small image noise
- Easier to visualize model confidence

### Limitations

- Slightly higher computation
- Additional post-processing required
- Memory usage is bit more

Despite that, heatmap based approach perform very well on landmark localization tasks.

---

## Evaluation Metrics

All hypothesis are evaluated using:

- Mean Radial Error (MRE)
- Pixel distance error after rescaling
- Visual inspection of predicted landmarks

Numerical metrics are always supported by qualitative visualization to avoid misleading scores.

---
