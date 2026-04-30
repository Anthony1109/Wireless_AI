"""
CsiNet_onlytest.py
Inference Script to evaluate Generalization across 6 datasets (Exercise 2.15)

This script loads a pre-trained CsiNet autoencoder and tests its 
reconstruction performance (NMSE) across 6 different spatial user distributions.
It does not train the model; it only performs forward-pass inference.
"""

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import scipy.io as sio 
import numpy as np
import math
import os

# ────────────────────────  Configuration  ───────────────────────── #
# Select which trained model you want to evaluate:
# Set to 'B' to test the baseline model (trained only on Dataset 1)
# Set to 'C' to test the mixed-training model (trained on all 6 Datasets)
EVALUATE_MODEL_TASK = 'C'  
encoded_dim = 512      # Compression dimension (must match the saved model)

# CSI matrix parameters (must strictly match the training configuration)
img_height = 32        # Number of Base Station antennas        
img_width = 32         # Number of subcarriers 
img_channels = 2       # Real and Imaginary parts

# Construct the expected model filename based on the selected task
model_name = f'CsiNet_Task{EVALUATE_MODEL_TASK}_dim{encoded_dim}'

# ────────────────────────  Load Pre-trained Model ───────────────── #
print(f"Loading pre-trained model: {model_name}")

# Define file paths for the architecture, weights, and normalization factor
json_path = f"result/model_{model_name}.json"
weights_path = f"result/model_{model_name}.weights.h5"
max_abs_path = f"result/{model_name}_max_abs.npy"

# Safety check to ensure the model exists before running
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Cannot find {json_path}. Please run training first.")

# 1. Load Model Architecture
with open(json_path, 'r') as json_file:
    autoencoder = model_from_json(json_file.read())

# 2. Load Model Weights
autoencoder.load_weights(weights_path)

# 3. Load the global absolute maximum value used during training.
# This is strictly required to reverse the normalization (denormalize) 
# so NMSE is calculated on the true physical scale of the channel.
max_abs = np.load(max_abs_path)
print("Model loaded successfully.\n")

# ────────────────────────  Data Preprocessing Function ──────────── #
def load_and_preprocess_mat(dataset_idx):
    """
    Loads the raw COST2100 MATLAB dataset and reshapes it to match
    the input requirements of the CsiNet model.
    
    Args:
        dataset_idx (int): Dataset identifier (1 through 6).
    Returns:
        np.array: Formatted CSI data of shape [Samples, 2, 32, 32].
    """
    mat = sio.loadmat(f'data/channel_dataset_{dataset_idx}.mat')
    H_raw = mat['H_transfer'] 
    
    # Truncate to the first 32 subcarriers and antennas
    H_sliced = H_raw[:, :img_width, :, :img_height] 
    
    # Flatten snapshots and users into a single 'Samples' dimension
    H_samples = np.transpose(H_sliced, (0, 2, 1, 3)).reshape(-1, img_width, img_height)
    
    # Split the complex matrix into Real and Imaginary channels
    x_data = np.zeros((H_samples.shape[0], img_channels, img_height, img_width), dtype=np.float32)
    x_data[:, 0, :, :] = np.real(H_samples)
    x_data[:, 1, :, :] = np.imag(H_samples)
    
    return x_data

# ────────────────────────  Evaluate on ALL 6 Datasets ───────────── #
print("==================================================")
print(f" Evaluating Model trained on Task {EVALUATE_MODEL_TASK} ")
print("==================================================")

# Loop through all 6 spatial distribution datasets to evaluate generalization
for idx in range(1, 7):
    # 1. Load and reshape the raw dataset
    x_raw = load_and_preprocess_mat(idx)
    
    # 2. Apply the exact same normalization used during training
    # Maps data to [0, 1] range to match the model's Sigmoid output
    x_test = (x_raw / max_abs) * 0.5 + 0.5
    
    # 3. Perform Inference (Reconstruct the compressed CSI)
    x_hat = autoencoder.predict(x_test, verbose=0)
    
    # 4. Denormalize and convert back to the Complex Domain
    # We must undo the mapping: y = (x / max_abs) * 0.5 + 0.5 
    # Therefore: x = (y - 0.5) * 2 * max_abs
    
    # Denormalize Ground Truth
    x_test_real = (x_test[:, 0, :, :] - 0.5) * 2 * max_abs
    x_test_imag = (x_test[:, 1, :, :] - 0.5) * 2 * max_abs
    x_test_complex = x_test_real + 1j * x_test_imag
    
    # Denormalize Reconstructed Prediction
    x_hat_real = (x_hat[:, 0, :, :] - 0.5) * 2 * max_abs
    x_hat_imag = (x_hat[:, 1, :, :] - 0.5) * 2 * max_abs
    x_hat_complex = x_hat_real + 1j * x_hat_imag
    
    # 5. Calculate Normalized Mean Square Error (NMSE)
    # Power: Sum of the squared magnitudes of the original complex CSI
    # MSE: Sum of the squared magnitudes of the error (difference between original and reconstructed)
    power = np.sum(np.abs(x_test_complex)**2, axis=(1, 2))
    mse = np.sum(np.abs(x_test_complex - x_hat_complex)**2, axis=(1, 2))
    
    # Convert the ratio to logarithmic decibel (dB) scale
    nmse_db = 10 * math.log10(np.mean(mse / power))
    
    # Print the evaluation result for the current spatial distribution
    print(f"-> Dataset {idx} NMSE: {nmse_db:8.2f} dB")

print("==================================================\n")