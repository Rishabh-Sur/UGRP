import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# Configuration
CAD = "cad100"
filepath = "data/data.txt"

print(f"\nLoading averaged PIV data from TXT file: {filepath}")

# Load TXT data (skip header)
# Columns in file:
# X (mm), Y (mm), U (m/s), V (m/s)

data = np.loadtxt(filepath, skiprows=1)

print(f"Loaded data shape: {data.shape}")

# Extract components
X_all = data[:, 0]   # X coordinates (mm)
Y_all = data[:, 1]   # Y coordinates (mm)
u_all = data[:, 2]   # U velocities (m/s)
v_all = data[:, 3]   # V velocities (m/s)

print(f"X range: [{X_all.min():.2f}, {X_all.max():.2f}] mm")
print(f"Y range: [{Y_all.min():.2f}, {Y_all.max():.2f}] mm")

# =========================
# Create mask based on convex hull of data points
# =========================
points = np.column_stack([X_all, Y_all])
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])

# =========================
# Create regular grid for interpolation
# =========================
x_min, x_max = X_all.min(), X_all.max()
y_min, y_max = Y_all.min(), Y_all.max()

grid_resolution = 200
x_grid = np.linspace(x_min, x_max, grid_resolution)
y_grid = np.linspace(y_min, y_max, grid_resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# =========================
# Interpolate velocity components onto grid
# =========================
u_grid = griddata(points, u_all, (X_grid, Y_grid), method='linear')
v_grid = griddata(points, v_all, (X_grid, Y_grid), method='linear')

# =========================
# Create mask for valid data points (within convex hull)
# =========================
mask = hull_path.contains_points(np.column_stack([X_grid.ravel(), Y_grid.ravel()]))
mask = mask.reshape(X_grid.shape)

# Apply mask
u_grid[~mask] = np.nan
v_grid[~mask] = np.nan

# =========================
# Calculate vorticity: ω_z = ∂v/∂x - ∂u/∂y
# =========================
print("\nCalculating vorticity...")

# Calculate spatial gradients
# Note: np.gradient returns gradients along each axis
# For 2D: gradient[0] is along first axis (y), gradient[1] is along second axis (x)
du_dy, du_dx = np.gradient(u_grid)
dv_dy, dv_dx = np.gradient(v_grid)

# Calculate grid spacing (assuming uniform spacing)
dx = (x_max - x_min) / (grid_resolution - 1)
dy = (y_max - y_min) / (grid_resolution - 1)

# Scale gradients by grid spacing
du_dx = du_dx / dx
du_dy = du_dy / dy
dv_dx = dv_dx / dx
dv_dy = dv_dy / dy

# Calculate vorticity (out-of-plane component for 2D flow)
# ω_z = ∂v/∂x - ∂u/∂y
vorticity = dv_dx - du_dy

# Calculate vorticity magnitude
vorticity_magnitude = np.abs(vorticity)

# Mask out invalid regions
vorticity_magnitude[~mask] = np.nan

print(f"Vorticity magnitude range: [{np.nanmin(vorticity_magnitude):.4f}, {np.nanmax(vorticity_magnitude):.4f}] 1/s")

# =========================
# Calculate threshold: 60th percentile (top 40%)
# =========================
# Get all valid vorticity values
valid_vorticity = vorticity_magnitude[~np.isnan(vorticity_magnitude)]
omega_threshold = np.percentile(valid_vorticity, 60)

print(f"\nVorticity threshold (60th percentile, top 40%): {omega_threshold:.4f} 1/s")
print(f"Number of valid points: {len(valid_vorticity)}")

# =========================
# Identify vortex regions: |ω| > ω_th
# =========================
vortex_mask = vorticity_magnitude > omega_threshold
vortex_mask[~mask] = False  # Ensure we only mark valid regions

num_vortex_points = np.sum(vortex_mask)
print(f"Number of points identified as vortices: {num_vortex_points}")

# =========================
# Create visualization similar to kmeans_2.py
# =========================
print("\nCreating visualization...")

fig, ax = plt.subplots(figsize=(14, 12))

# Plot vorticity magnitude as background colormap
im = ax.contourf(X_grid, Y_grid, vorticity_magnitude, levels=50, cmap='viridis', extend='both', alpha=0.7)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Vorticity Magnitude |ω| (1/s)', fontsize=12)

# Overlay vortex regions (where |ω| > ω_th) with distinct color
# Create a colored overlay for vortex regions
vortex_overlay = np.zeros_like(vorticity_magnitude)
vortex_overlay[vortex_mask] = 1.0
vortex_overlay[~mask] = np.nan

# Plot vortex regions with distinct color (red with transparency)
ax.contourf(X_grid, Y_grid, vortex_overlay, levels=[0.5, 1.5], colors=['red'], alpha=0.5, zorder=1)

# Draw threshold contour line (white dashed)
threshold_contour = ax.contour(X_grid, Y_grid, vorticity_magnitude, levels=[omega_threshold], 
                               colors='white', linewidths=2, linestyles='--', zorder=2)
ax.clabel(threshold_contour, inline=True, fontsize=10, fmt=f'|ω| = {omega_threshold:.2f} 1/s')

# Plot velocity vectors for context
skip = 6  # Show every 6th vector for clarity
ax.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
          u_grid[::skip, ::skip], v_grid[::skip, ::skip],
          color='white', scale=8, scale_units='xy', angles='xy', 
          width=0.003, alpha=0.8, zorder=3)

ax.set_xlabel('X (mm)', fontsize=12)
ax.set_ylabel('Y (mm)', fontsize=12)
ax.set_title(f'Vorticity Magnitude Criterion - {CAD}\n(Threshold: 60th percentile = {omega_threshold:.2f} 1/s)', 
             fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend for vortex regions
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.5, label='Vortex regions (|ω| > threshold)')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', frameon=True)

plt.tight_layout()
plt.show()

print(f"\nVisualization complete!")
print(f"Vortex regions identified: {num_vortex_points} points ({100*num_vortex_points/len(valid_vorticity):.1f}% of valid domain)")
