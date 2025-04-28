# Recursive Novelty & Anomaly Discriminator (RNAD) Starter

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Simulated bursts embeddings
np.random.seed(42)
burst_embeddings = np.random.rand(50, 128)  # Example: 50 bursts, 128-dim embeddings

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.1)
anomaly_labels = iso_forest.fit_predict(burst_embeddings)

# Novelty Detection (PCA-based)
pca = PCA(n_components=10)
pca.fit(burst_embeddings)
reconstructed = pca.inverse_transform(pca.transform(burst_embeddings))
reconstruction_error = np.mean((burst_embeddings - reconstructed) ** 2, axis=1)

# Threshold for novelty
novelty_threshold = np.percentile(reconstruction_error, 90)
novelty_labels = (reconstruction_error > novelty_threshold).astype(int)

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(reconstruction_error, np.random.rand(50), c=anomaly_labels, cmap='coolwarm', label='Anomaly Label')
plt.axvline(novelty_threshold, color='green', linestyle='--', label='Novelty Threshold')
plt.title('Novelty vs Anomaly Scatter')
plt.xlabel('Reconstruction Error (Novelty Proxy)')
plt.ylabel('Random noise axis')
plt.legend()
plt.show()
