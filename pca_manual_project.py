import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

# Load digits dataset (8x8 pixel images of handwritten digits)
digits = load_digits()
X = digits.data  # Shape (1797, 64)

# Step 1: Standardize the data to have zero mean and unit variance
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Avoid division by zero by replacing zero std values with 1
X_std[X_std == 0] = 1
X_standardized = (X - X_mean) / X_std

# Step 2: Compute Covariance Matrix
cov_matrix = np.cov(X_standardized.T)

# 🔍 Visualization: Heatmap of Covariance Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix, cmap='coolwarm', cbar=True)
plt.title("Covariance Matrix Heatmap")
plt.show()

# Step 3: Compute Eigenvalues and Eigenvectors of the Covariance Matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 4: Explained variance analysis
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)

# 🔍 Plot Explained Variance
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance vs. Components")
plt.axhline(y=0.95, color='r', linestyle='--', label="95% variance threshold")
plt.legend()
plt.grid()
plt.show()

# Find minimum k for 95% variance retention
k_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Minimum components needed for 95% variance retention: {k_95}")

# Step 5: Select top k principal components
k = 20  # Choose 20 components (adjustable)
top_eigenvectors = eigenvectors[:, :k]

# Step 6: Transform Data using PCA
X_pca = np.dot(X_standardized, top_eigenvectors)

# Step 7: Reconstruct images from reduced PCA representation
X_reconstructed = np.dot(X_pca, top_eigenvectors.T) * X_std + X_mean

# 🔍 Compare original and reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(10, 3))
for i in range(10):
    axes[0, i].imshow(X[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")
plt.show()

# Print final analysis summary
print(f"Total Components: {len(eigenvalues)}")
print(f"Explained Variance by First {k} Components: {np.sum(explained_variance_ratio[:k]):.2%}")

