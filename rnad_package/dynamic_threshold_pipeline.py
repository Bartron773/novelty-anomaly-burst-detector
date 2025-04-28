# RNAD Dynamic Threshold Starter

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Simulated burst embeddings (replace with real ones)
np.random.seed(42)
burst_embeddings = np.random.rand(50, 128)

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(burst_embeddings)

# Novelty Detection using PCA
pca = PCA(n_components=10, random_state=42)
pca.fit(burst_embeddings)
reconstructed = pca.inverse_transform(pca.transform(burst_embeddings))
reconstruction_error = np.mean((burst_embeddings - reconstructed) ** 2, axis=1)

# Dynamic Threshold based on Median Absolute Deviation (MAD)
median_error = np.median(reconstruction_error)
mad = np.median(np.abs(reconstruction_error - median_error))
dynamic_novelty_threshold = median_error + (3 * mad)

novelty_labels = (reconstruction_error > dynamic_novelty_threshold).astype(int)

# Visualization
plt.figure(figsize=(10,6))
plt.scatter(reconstruction_error, np.random.rand(50), c=['red' if a == -1 else 'blue' for a in anomaly_labels], s=80, edgecolors='k')
plt.axvline(dynamic_novelty_threshold, color='green', linestyle='--', label='Dynamic Novelty Threshold')
plt.title('Burst Landscape: Dynamic Threshold')
plt.xlabel('Reconstruction Error (Proxy for Novelty)')
plt.ylabel('Randomized Y-axis (Visual Separation)')
plt.legend()
plt.grid(True)
plt.show()
