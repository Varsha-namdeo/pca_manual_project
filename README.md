# PCA on Handwritten Digits Dataset

## Overview
This project demonstrates the application of **Principal Component Analysis (PCA)** on the **handwritten digits dataset** from Scikit-Learn. The goal is to reduce the dimensionality of the dataset while preserving as much variance as possible and reconstruct the images from the transformed data.

## Dataset
The dataset consists of 8x8 pixel grayscale images of handwritten digits (0-9). It contains **1,797** samples, each with **64 features** (8x8 pixel values).

## Steps Implemented

### 1. Standardization
- Compute mean and standard deviation of the dataset.
- Standardize the dataset to have zero mean and unit variance.

### 2. Compute Covariance Matrix
- Compute the covariance matrix of the standardized data.
- Visualize it using a heatmap.

### 3. Eigen Decomposition
- Compute eigenvalues and eigenvectors of the covariance matrix.
- Sort them in descending order of eigenvalues.

### 4. Explained Variance Analysis
- Calculate the **explained variance ratio**.
- Plot the **cumulative explained variance** to determine the optimal number of components.
- Identify the minimum number of principal components required to retain **95% variance**.

### 5. Dimensionality Reduction
- Select the top **k principal components** (default: 20).
- Transform the dataset using PCA.

### 6. Reconstruction
- Reconstruct images using the reduced PCA representation.
- Compare original vs. reconstructed images.

## Results
- **Visual Comparison**: The reconstructed images retain most of the essential features while reducing noise.
- **Dimensionality Reduction**: The dataset is significantly reduced while retaining most of its variance.
- **Explained Variance**: The first few principal components explain the majority of variance in the dataset.

## Dependencies
- Python
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## How to Run
1. Install dependencies using:
   ```sh
   pip install numpy matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```sh
   python pca_manual_project.py
   ```
## Author
Varsha Namdeo

