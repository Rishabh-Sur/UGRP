import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration
CAD = "cad100"
filepath = "data/data.txt"

print(f"\nLoading averaged PIV data from TXT file: {filepath}")

# =========================
# Load TXT data (skip header, use only first 2551 rows)
# =========================
# Columns in file:
# X (mm), Z (mm), U (m/s), V (m/s)

data = np.loadtxt(filepath, skiprows=1, max_rows=2551)

print(f"Loaded data shape: {data.shape}")

# Extract components (keeping variable names same as original code)
X_all = data[:, 0]   # X coordinates (mm)
Y_all = data[:, 1]   # Z coordinates (mm) -> kept as Y_all to keep rest identical
u_all = data[:, 2]   # U velocities (m/s)
v_all = data[:, 3]   # V velocities (m/s)

# Normalize velocity (direction only)
speed = np.sqrt(u_all**2 + v_all**2) + 1e-8
u_hat = u_all / speed
v_hat = v_all / speed

# Feature vector per point (from all ensembles)
features = np.column_stack([
    X_all,
    Y_all,
    u_hat,
    v_hat
])

print(f"Features shape: {features.shape}")

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

K = 2   

kmeans = KMeans(
    n_clusters=K,
    n_init=20,
    random_state=0
)

labels = kmeans.fit_predict(features_scaled)

# Get cluster centroids (in original feature space, before scaling)
# Transform centroids back from scaled space to original space
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Extract centroid coordinates (X, Y) for plotting
centroid_X = centroids_original[:, 0]
centroid_Y = centroids_original[:, 1]

print(f"\nNumber of clusters: {K}")
print(f"Cluster centroids (X, Y):")
for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    print(f"  Cluster {i}: ({cx:.2f}, {cy:.2f})")

# Create visualization with cluster centroids highlighted
plt.figure(figsize=(12, 10))

# Define distinct, vibrant colors for each cluster
# Using bright, easily distinguishable colors
cluster_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF']
# Red, Blue, Green, Magenta, Yellow, Cyan

# Plot each cluster with a distinct, vibrant color
for cluster_id in range(K):
    cluster_mask = labels == cluster_id
    cluster_X = X_all[cluster_mask]
    cluster_Y = Y_all[cluster_mask]
    
    plt.scatter(cluster_X, cluster_Y, c=cluster_colors[cluster_id], 
               s=8, alpha=0.9, edgecolors='none', 
               label=f'Cluster {cluster_id}', zorder=1)

# Highlight cluster centroids with larger, distinct markers
# Use matching colors for centroids with black borders for visibility
for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    plt.scatter(cx, cy, c=cluster_colors[i], marker='*', s=1200, 
               edgecolors='black', linewidths=4, 
               label=f'Centroid {i}' if i == 0 else '', zorder=10)
    
    # Add labels for centroids with matching colors
    plt.annotate(f'C{i}', (cx, cy), xytext=(10, 10), textcoords='offset points',
                fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=cluster_colors[i], 
                         edgecolor='black', linewidth=3, alpha=0.95))

plt.title(f"Vortex zones detected using K-means - {CAD} (From averaged TXT data)", 
          fontsize=14, fontweight='bold')
plt.xlabel("X (mm)", fontsize=12)
plt.ylabel("Y (mm)", fontsize=12)
plt.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nVisualization complete!")
