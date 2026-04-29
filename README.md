DR Detection Pipeline
This notebook provides a pipeline for Diabetic Retinopathy (DR) detection, including training, evaluation, and explainability features using Grad-CAM, Integrated Gradients, and Guided Grad-CAM.

Table of Contents
Project Description
Setup
Usage
Training Mode
Evaluation Mode
Explainability Mode
Configuration
Outputs
Project Description
This pipeline is designed to:

Train a deep learning model for Diabetic Retinopathy classification.
Evaluate the trained model on a validation set.
Explain model predictions for a given image using various interpretability methods.
Setup
Before running the pipeline, ensure you have the necessary libraries installed. This typically includes torch, torchvision, numpy, pandas, matplotlib, seaborn, opencv-python, and argparse. You will also need a CFG (configuration) object defined elsewhere in your notebook that contains settings like MODEL_SAVE, RESULTS_DIR, IMG_SIZE, GRADCAM_LAYER, IG_STEPS, etc.

# Example of potential dependencies (install if not already present)
# !pip install torch torchvision opencv-python matplotlib
Usage
The pipeline can be run in three different modes: train, eval, and explain. You can specify the mode using command-line arguments. In a Colab environment, this typically means passing arguments to the parser.parse_args() function or setting them directly in the notebook.

Training Mode
To train the model, run the script with --mode train.

# In a Colab cell, you might simulate command-line arguments like this:
# import sys
# sys.argv = ['your_script_name.py', '--mode', 'train']
# (then run the main pipeline cell)

# Or, more directly for testing:
# args.mode = 'train'
# args.weights = None # Not needed for training
# args.img = None     # Not needed for training

# The main pipeline will then execute the 'train' block
Output: Generates and saves training curves (training_curves.png) showing Quadratic Weighted Kappa (QWK) and validation accuracy.

Evaluation Mode
To evaluate a pre-trained model, run the script with --mode eval and provide the path to the model weights using --weights.

# args.mode = 'eval'
# args.weights = 'path/to/your/model_weights.pth'
# args.img = None # Not needed for evaluation

# The main pipeline will then execute the 'eval' block
Output: Computes and saves evaluation metrics (e.g., QWK, accuracy, F1-score) to metrics.json.

Explainability Mode
To get explanations for a specific image, run the script with --mode explain and provide the image path using --img. You also need to specify the path to the pre-trained model weights with --weights.

# args.mode = 'explain'
# args.weights = 'path/to/your/model_weights.pth'
# args.img = 'path/to/your/image.jpg'

# The main pipeline will then execute the 'explain' block
Output: Visualizes Grad-CAM, Integrated Gradients, and Guided Grad-CAM heatmaps overlaid on the original image, along with AOPC and Map Entropy metrics. The resulting figure is saved as explanation.png.

Configuration
The pipeline relies on a CFG object (expected to be defined globally or passed around) for various configurations such as:

CFG.MODEL_SAVE: Default path for saving model weights.
CFG.RESULTS_DIR: Directory for saving output files (training curves, metrics, explanations).
CFG.IMG_SIZE: Image size for preprocessing.
CFG.GRADCAM_LAYER: The target layer for Grad-CAM.
CFG.IG_STEPS: Number of steps for Integrated Gradients computation.
Example CFG structure:

# Example CFG (should be defined in a preceding cell)
class Config:
    MODEL_SAVE = 'model_weights.pth'
    RESULTS_DIR = 'results'
    IMG_SIZE = 224
    GRADCAM_LAYER = 'layer4' # Example layer name for a ResNet-like model
    IG_STEPS = 50

CFG = Config()

# Ensure results directory exists
import os
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
Outputs
results/training_curves.png: Plot of training and validation QWK and accuracy (Train Mode).
results/metrics.json: JSON file containing evaluation metrics (Eval Mode).
results/explanation.png: Visualization of explainability methods (Explain Mode).
