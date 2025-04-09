# PCA on Handwritten Digits Dataset (Scikit-learn)

This project demonstrates a manual implementation of **Principal Component Analysis (PCA)** using the `digits` dataset from Scikit-learn. It includes data preprocessing, PCA computation from scratch using NumPy, variance analysis, visualization, and image reconstruction.

---

## 📌 Objective

- Reduce the dimensionality of image data using PCA.
- Understand and visualize how much variance each principal component retains.
- Reconstruct original digit images from the reduced representation and evaluate the quality.

---

## 📊 Dataset

- **Name**: `digits`
- **Source**: `sklearn.datasets.load_digits()`
- **Shape**: `(1797 samples, 64 features)`
- **Image Size**: 8x8 grayscale images of handwritten digits (0–9)

---

## 🚀 Steps Performed

### 1. Standardization
- Standardized the data to have **zero mean** and **unit variance** for each feature.
- Handled zero standard deviation by replacing zeros with ones.

### 2. Covariance Matrix
- Computed the **covariance matrix** of the standardized data.
- Visualized the matrix using a **Seaborn heatmap**.

### 3. Eigen Decomposition
- Computed **eigenvalues** and **eigenvectors** from the covariance matrix.
- Sorted them in descending order based on explained variance.

### 4. Explained Variance Analysis
- Calculated the **explained variance ratio** for each component.
- Plotted **cumulative explained variance**.
- Identified the **minimum number of components** required to retain 95% of variance.

### 5. Dimensionality Reduction
- Selected top `k = 20` principal components.
- Projected the data to this lower-dimensional space.

### 6. Image Reconstruction
- Reconstructed the original images from the reduced representation.
- Compared original and reconstructed images side-by-side.

---

## 📈 Visualizations

- 🔥 Covariance Matrix Heatmap
- 📉 Cumulative Explained Variance Plot
- 🧠 Original vs. Reconstructed Digits (Grayscale Images)

---

## 🧮 Key Results

- **Total Principal Components**: 64
- **Minimum components for 95% variance**: *Printed in console*
- **Explained variance using top 20 components**: *Printed in console*

---

## 📦 Requirements

```bash
pip install numpy matplotlib seaborn scikit-learn
