import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path

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

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(u_all**2 + v_all**2)

print(f"Velocity magnitude range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}] m/s")
print(f"X range: [{X_all.min():.2f}, {X_all.max():.2f}] mm")
print(f"Y range: [{Y_all.min():.2f}, {Y_all.max():.2f}] mm")

print("\n" + "="*60)
print("Step 0: Computing Vorticity and Swirling Strength")
print("="*60)

# Create regular grid for interpolation (needed for gradient computation)
x_min, x_max = X_all.min(), X_all.max()
y_min, y_max = Y_all.min(), Y_all.max()

# Create fine grid
grid_resolution = 200
x_grid = np.linspace(x_min, x_max, grid_resolution)
y_grid = np.linspace(y_min, y_max, grid_resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Points for interpolation
points = np.column_stack([X_all, Y_all])

# Create mask for valid data points (within convex hull)
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])
mask = hull_path.contains_points(np.column_stack([X_grid.ravel(), Y_grid.ravel()]))
mask = mask.reshape(X_grid.shape)

# Interpolate u and v onto grid
print("Interpolating velocity components onto grid...")
u_grid = griddata(points, u_all, (X_grid, Y_grid), method='linear')
v_grid = griddata(points, v_all, (X_grid, Y_grid), method='linear')

# Apply mask to interpolated data
u_grid[~mask] = np.nan
v_grid[~mask] = np.nan

# Grid spacing (assumes uniform grid)
dx = np.mean(np.diff(X_grid[0, :]))
dy = np.mean(np.diff(Y_grid[:, 0]))

print(f"Grid spacing: dx = {dx:.4f} mm, dy = {dy:.4f} mm")

# Compute velocity gradients
print("Computing velocity gradients...")
du_dx = np.gradient(u_grid, dx, axis=1)
du_dy = np.gradient(u_grid, dy, axis=0)

dv_dx = np.gradient(v_grid, dx, axis=1)
dv_dy = np.gradient(v_grid, dy, axis=0)

# 1. Compute Vorticity (out-of-plane for tumble plane)
# ω_y = ∂v/∂x - ∂u/∂y
print("Computing vorticity...")
vorticity_grid = dv_dx - du_dy  # Units: 1/s

# 2. Compute Swirling Strength (λ_ci)
# Swirling strength = imaginary part of complex eigenvalues of velocity gradient tensor
print("Computing swirling strength...")
swirling_strength_grid = np.zeros_like(vorticity_grid)

for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        # Skip NaNs
        if np.isnan(du_dx[i, j]) or np.isnan(du_dy[i, j]) or \
           np.isnan(dv_dx[i, j]) or np.isnan(dv_dy[i, j]):
            swirling_strength_grid[i, j] = np.nan
            continue
        
        # Velocity gradient tensor
        A = np.array([
            [du_dx[i, j], du_dy[i, j]],
            [dv_dx[i, j], dv_dy[i, j]]
        ])
        
        # Eigenvalues
        eigvals = np.linalg.eigvals(A)
        
        # Imaginary part gives swirling strength
        swirling_strength_grid[i, j] = np.max(np.abs(np.imag(eigvals)))

print(f"Vorticity range: [{np.nanmin(vorticity_grid):.4f}, {np.nanmax(vorticity_grid):.4f}] 1/s")
print(f"Swirling strength range: [{np.nanmin(swirling_strength_grid):.4f}, {np.nanmax(swirling_strength_grid):.4f}] 1/s")

# Interpolate vorticity and swirling strength back to original data points
print("Interpolating vorticity and swirling strength back to data points...")
vorticity_all = griddata((X_grid.ravel(), Y_grid.ravel()), 
                         vorticity_grid.ravel(), 
                         points, method='linear')
swirling_strength_all = griddata((X_grid.ravel(), Y_grid.ravel()), 
                                 swirling_strength_grid.ravel(), 
                                 points, method='linear')

# Handle any NaN values (points outside interpolation domain)
# Replace with median value
vorticity_all = np.nan_to_num(vorticity_all, nan=np.nanmedian(vorticity_all))
swirling_strength_all = np.nan_to_num(swirling_strength_all, nan=np.nanmedian(swirling_strength_all))

print(f"Vorticity at data points - range: [{np.min(vorticity_all):.4f}, {np.max(vorticity_all):.4f}] 1/s")
print(f"Swirling strength at data points - range: [{np.min(swirling_strength_all):.4f}, {np.max(swirling_strength_all):.4f}] 1/s")

# Prepare features for clustering: [X, Y, u, v, vorticity, swirling_strength]
print("\n" + "="*60)
print("Preparing feature vector: [X, Y, u, v, vorticity, swirling_strength]")
print("="*60)

features = np.column_stack([
    X_all,                  # X coordinates (mm)
    Y_all,                  # Y coordinates (mm)
    u_all,                  # U velocity (m/s)
    v_all,                  # V velocity (m/s)
    vorticity_all,          # Vorticity (1/s)
    swirling_strength_all   # Swirling strength (1/s)
])

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Features prepared and scaled.")

print("\n" + "="*60)
print("Step 1: GMM Clustering with K=2")
print("="*60)

# Set K=2
optimal_k = 2

print(f"\nUsing K = {optimal_k} for clustering")

# Fit GMM with K=2
optimal_gmm = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=42, max_iter=200)
optimal_gmm.fit(features_scaled)

# Calculate BIC and AIC for reference
optimal_bic = optimal_gmm.bic(features_scaled)
optimal_aic = optimal_gmm.aic(features_scaled)

print(f"BIC = {optimal_bic:.2f}, AIC = {optimal_aic:.2f}")

# Fit GMM and get initial cluster assignments
gmm_labels = optimal_gmm.predict(features_scaled)
gmm_probs = optimal_gmm.predict_proba(features_scaled)

# Get GMM means (centroids) in scaled space
gmm_centroids_scaled = optimal_gmm.means_

# Transform back to original space
gmm_centroids_original = scaler.inverse_transform(gmm_centroids_scaled)

print(f"\nGMM clustering completed with K={optimal_k}")
for i in range(optimal_k):
    count = np.sum(gmm_labels == i)
    cx, cy = gmm_centroids_original[i, 0], gmm_centroids_original[i, 1]
    print(f"  Cluster {i}: {count} points, centroid at ({cx:.2f}, {cy:.2f})")

print("\n" + "="*60)
print("Step 2: KMeans Refinement")
print("="*60)

# Use GMM centroids as initial centers for KMeans
kmeans = KMeans(n_clusters=optimal_k, init=gmm_centroids_scaled, n_init=1, 
                max_iter=300, random_state=42)
kmeans.fit(features_scaled)

# Get final KMeans labels and centroids
final_labels = kmeans.labels_
final_centroids_scaled = kmeans.cluster_centers_
final_centroids_original = scaler.inverse_transform(final_centroids_scaled)

# Calculate inertia (within-cluster sum of squares)
inertia = kmeans.inertia_

print(f"\nKMeans refinement completed")
print(f"Final inertia (within-cluster sum of squares): {inertia:.4f}")
for i in range(optimal_k):
    count = np.sum(final_labels == i)
    cx, cy = final_centroids_original[i, 0], final_centroids_original[i, 1]
    print(f"  Cluster {i}: {count} points, centroid at ({cx:.2f}, {cy:.2f})")

print("\n" + "="*60)
print("Step 3: Visualization")
print("="*60)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Define distinct, vibrant colors
cluster_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#FF8000', '#8000FF']

# Plot 1: GMM Results
ax1 = axes[0]
for i in range(optimal_k):
    mask = gmm_labels == i
    ax1.scatter(X_all[mask], Y_all[mask], c=cluster_colors[i % len(cluster_colors)], 
               label=f'Cluster {i}', alpha=0.6, s=20)

# Plot GMM centroids
for i in range(optimal_k):
    cx, cy = gmm_centroids_original[i, 0], gmm_centroids_original[i, 1]
    ax1.scatter(cx, cy, c='black', marker='x', s=200, linewidths=3, 
               label='GMM Centroid' if i == 0 else '', zorder=10)
    ax1.text(cx, cy, f'  {i}', fontsize=12, fontweight='bold', zorder=11)

ax1.set_xlabel('X (mm)', fontsize=12)
ax1.set_ylabel('Y (mm)', fontsize=12)
ax1.set_title(f'GMM Clustering (K={optimal_k})', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Final KMeans Results
ax2 = axes[1]
for i in range(optimal_k):
    mask = final_labels == i
    ax2.scatter(X_all[mask], Y_all[mask], c=cluster_colors[i % len(cluster_colors)], 
               label=f'Cluster {i}', alpha=0.6, s=20)

# Plot KMeans centroids
for i in range(optimal_k):
    cx, cy = final_centroids_original[i, 0], final_centroids_original[i, 1]
    ax2.scatter(cx, cy, c='black', marker='x', s=200, linewidths=3, 
               label='KMeans Centroid' if i == 0 else '', zorder=10)
    ax2.text(cx, cy, f'  {i}', fontsize=12, fontweight='bold', zorder=11)

ax2.set_xlabel('X (mm)', fontsize=12)
ax2.set_ylabel('Y (mm)', fontsize=12)
ax2.set_title(f'Final KMeans Clustering (K={optimal_k}, Inertia={inertia:.2f})', 
              fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.suptitle(f'Hybrid GMM + KMeans Clustering - {CAD}', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('gmm_kmeans_clustering.png', dpi=150, bbox_inches='tight')
print("\nSaved clustering visualization to 'gmm_kmeans_clustering.png'")
plt.show()

# Create detailed single plot with velocity field overlay
print("\n" + "="*60)
print("Step 4: Detailed Visualization with Velocity Field")
print("="*60)

fig, ax = plt.subplots(figsize=(14, 12))

# Plot clusters with colors
for i in range(optimal_k):
    mask = final_labels == i
    ax.scatter(X_all[mask], Y_all[mask], c=cluster_colors[i % len(cluster_colors)], 
              label=f'Cluster {i} ({np.sum(mask)} points)', alpha=0.7, s=30, edgecolors='black', linewidths=0.5)

# Plot centroids
for i in range(optimal_k):
    cx, cy = final_centroids_original[i, 0], final_centroids_original[i, 1]
    ax.scatter(cx, cy, c='black', marker='*', s=500, edgecolors='white', 
              linewidths=2, label='Centroid' if i == 0 else '', zorder=10)
    ax.text(cx, cy, f'  C{i}', fontsize=14, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=11)

# Overlay velocity vectors (subsampled)
skip = 8  # Show every 8th vector
ax.quiver(X_all[::skip], Y_all[::skip], u_all[::skip], v_all[::skip],
         scale=100, width=0.003, alpha=0.4, color='gray', zorder=5)

ax.set_xlabel('X (mm)', fontsize=14)
ax.set_ylabel('Y (mm)', fontsize=14)
ax.set_title(f'Hybrid GMM + KMeans Clustering with Velocity Vectors\n{CAD}, K={optimal_k}, BIC={optimal_bic:.1f}', 
            fontsize=15, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('gmm_kmeans_detailed.png', dpi=150, bbox_inches='tight')
print("Saved detailed visualization to 'gmm_kmeans_detailed.png'")
plt.show()

print("\n" + "="*60)
print("Clustering Complete!")
print("="*60)
print(f"Optimal K: {optimal_k}")
print(f"BIC: {optimal_bic:.2f}")
print(f"AIC: {optimal_aic:.2f}")
print(f"Final Inertia: {inertia:.4f}")
print(f"\nCluster Summary:")
for i in range(optimal_k):
    count = np.sum(final_labels == i)
    cx, cy = final_centroids_original[i, 0], final_centroids_original[i, 1]
    avg_vel = np.mean(velocity_magnitude[final_labels == i])
    print(f"  Cluster {i}: {count} points, centroid=({cx:.2f}, {cy:.2f}) mm, avg_vel={avg_vel:.4f} m/s")
