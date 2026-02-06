import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

# Configuration
test_point = 'r1300_p95'  # kept for consistency (not used now)
cad = "cad100"
filepath = "data/data.txt"   # UPDATED

print(f"\nLoading averaged PIV data from TXT file: {filepath}")

# =========================
# Load TXT data (skip header)
# =========================
# Columns in file:
# X (mm), Z (mm), U (m/s), V (m/s)

data = np.loadtxt(filepath, skiprows=1)

print(f"Loaded data shape: {data.shape}")

# Extract components (keeping variable names same as your code)
X_all = data[:, 0]   # X coordinates (mm)
Y_all = data[:, 1]   # Z coordinates (mm) -> kept as Y_all to keep rest identical
u_all = data[:, 2]   # U velocities (m/s)
v_all = data[:, 3]   # V velocities (m/s)

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(u_all**2 + v_all**2)

print(f"Velocity magnitude range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}] m/s")
print(f"X range: [{X_all.min():.2f}, {X_all.max():.2f}] mm")
print(f"Y range: [{Y_all.min():.2f}, {Y_all.max():.2f}] mm")

# =========================
# Create mask based on convex hull of data points
# =========================
points = np.column_stack([X_all, Y_all])
hull = ConvexHull(points)
hull_mask = np.zeros(len(points), dtype=bool)
hull_mask[hull.vertices] = True

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
# Interpolate onto grid
# =========================
velocity_mag_grid = griddata(points, velocity_magnitude, (X_grid, Y_grid), method='linear')
u_grid = griddata(points, u_all, (X_grid, Y_grid), method='linear')
v_grid = griddata(points, v_all, (X_grid, Y_grid), method='linear')

# =========================
# Create mask for valid data points (within convex hull)
# =========================
from matplotlib.path import Path
hull_path = Path(points[hull.vertices])
mask = hull_path.contains_points(np.column_stack([X_grid.ravel(), Y_grid.ravel()]))
mask = mask.reshape(X_grid.shape)

# Apply mask
velocity_mag_grid[~mask] = np.nan
u_grid[~mask] = np.nan
v_grid[~mask] = np.nan

# =========================
# Create the plot (UNCHANGED)
# =========================
fig, ax = plt.subplots(figsize=(12, 10))

# Plot velocity magnitude as colormap
im = ax.contourf(X_grid, Y_grid, velocity_mag_grid, levels=50, cmap='viridis', extend='both')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Velocity magnitude (m/s)', fontsize=12)

# Plot velocity vectors (white arrows) - uniform size, clearly visible
skip = 4  # Show every 4th vector
ax.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
          u_grid[::skip, ::skip], v_grid[::skip, ::skip],
          color='white', scale=8, scale_units='xy', angles='xy', 
          width=0.003, alpha=0.9, zorder=10)

ax.set_xlabel('X (mm)', fontsize=12)
ax.set_ylabel('Y (mm)', fontsize=12)
ax.set_title(f'Velocity Field - {cad} (From averaged TXT data)', 
             fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

print("\nPlotting complete!")
