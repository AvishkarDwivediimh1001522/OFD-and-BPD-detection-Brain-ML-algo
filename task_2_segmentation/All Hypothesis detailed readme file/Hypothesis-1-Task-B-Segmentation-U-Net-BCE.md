# Hypothesis 1 : U-Net + Binary Cross Entropy (BCE) for Fetal Ultrasound Segmentation  (Task B)

This folder contains the core training pipeline for segmenting anatomical structures in fetal ultrasound images using a PyTorch-based U-Net.

##  Overview
The `trainer.ipynb` notebook implements a full deep learning workflow, from custom dataset loading to model weight persistence and architecture visualization.



##  Configuration
- **Input Size:** 256 x 256 (Grayscale/RGB)
- **Architecture:** U-Net (Encoder-Decoder with Skip Connections)
- **Optimizer:** Adam ($LR = 0.0001$)
- **Loss Function:** Binary Cross-Entropy (BCE)
- **Hardware:** CUDA-enabled (with CPU fallback)

## 📁 Workflow Steps

1. **Data Prep:** Images and masks are loaded via a custom PyTorch `Dataset` class. We apply resizing and normalization before spliting the data into 80% training and 20% validation sets.
2. **Model Design:** The U-Net uses `ConvBlocks` with Dropout to prevent the model from just memorizing the noise in the ultrasound frames.
3. **Training:** The loop runs for 6 epochs. It monitors validation loss and saves the `state_dict` to Google Drive only when performance improves. This ensures we keep the most generelized version of the weights.
4. **Validation:** A weight verification step is included to ensure the `.pth` file is valid and can be reloaded for inference later.
5. **Visualization:** Generates a full model graph using `torchview` saved in the `/Report` directory to map out tensor shapes across layers.

## 🏗️ Requirements
- PyTorch
- OpenCV
- Scikit-learn
- Torchview
- Tqdm


---

# Fetal Biometry Tester & Inference Pipeline

This notebook is used to test the trained U-Net model and perform post-processing on the segmentation masks to extract biometric measurements.



##  Key Components

- **Architecture Re-definition:** Includes the `hypothesis_1_task_2_UNet` structure to allow for seamless weight loading without external dependencies.
- **Inference Engine:** Loads the pre-trained `.pth` weights and runs a forward pass on unseen test data.
- **Post-Processing:** Converts probability maps into binary masks using an optimal threshold.
- **Biometry Extraction:** Uses a custom `get_biometry_points` function to perform ellipse fitting—simulating real-world clinical measurements like head circumference.

## 📁 Evaluation Workflow

1. **Environment Setup:** Initializes the computing device (CUDA/CPU) and sets up file pointers for the test dataset.
2. **Weight Loading:** The model is compiled and the state dictionary is loaded from the report directory.
3. **Segmentation & Fitting:** - The model predicts a mask for the input ultrasound scan.
    - We apply a binary threshold to clean up the output.
    - The `get_biometry_points` logic extracts the contour and fits a geometrical shape to calculate measurements.
4. **Visualization:** Generates side-by-side plots of the original image, the predicted mask, and the final biometric overlay.

## 📊 Results Output
The final outputs, including visualized predictions and processed biometry data, are saved to the `/Report` directory for further analasys and peer review.

## 🏗️ Requirements
- PyTorch
- OpenCV (cv2)
- Matplotlib (for visualization)
- NumPy