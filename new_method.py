import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata

# Configuration
CAD = "cad100"
filepath = "data/data.txt"

print(f"\nLoading averaged PIV data from TXT file: {filepath}")


# Load TXT data (skip header)

# Columns in file:
# X (mm), Y (mm), U (m/s), V (m/s)

data = np.loadtxt(filepath, skiprows=1)

print(f"Loaded data shape: {data.shape}")

# Extract components (keeping variable names same as original code)
X_all = data[:, 0]   # X coordinates (mm)
Y_all = data[:, 1]   # Z coordinates (mm) -> kept as Y_all to keep rest identical
u_all = data[:, 2]   # U velocities (m/s)
v_all = data[:, 3]   # V velocities (m/s)

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(u_all**2 + v_all**2)

print(f"Velocity magnitude range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}] m/s")

# Step 1 & 2: Find top 5 points (at max) that satisfy all 4 checks, in increasing order of velocity
print("\n" + "="*60)
print("Step 1 & 2: Finding top 5 points (at max) that satisfy all 4 sign checks")
print("="*60)

# Get indices of points sorted by velocity magnitude (ascending)
sorted_indices = np.argsort(velocity_magnitude)

# Check all points until we find 5 valid points or exhaust all points
total_points = len(sorted_indices)
print(f"Checking all {total_points} points in increasing order of velocity...")
print(f"Will stop when 5 valid points are found or all points are checked.")

def calculate_tan_theta(u, v):
    """Calculate tan(theta) where theta is angle with x-axis"""
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tan_theta = np.where(np.abs(u) > 1e-10, v / u, np.sign(v) * np.inf)
    return tan_theta

def calculate_theta_degrees(u, v):
    """Calculate theta in degrees where theta is angle with x-axis"""
    theta_rad = np.arctan2(v, u)
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def find_closest_points(x, y, X_all, Y_all, direction, n=2):
    """
    Find n closest points in a given direction
    direction: 'top_right', 'bottom_left', 'top_left', 'bottom_right'
    """
    # Calculate relative positions
    dx = X_all - x
    dy = Y_all - y
    
    if direction == 'top_right':
        mask = (dx > 0) & (dy > 0)
        distances = np.sqrt(dx**2 + dy**2)
    elif direction == 'bottom_left':
        mask = (dx < 0) & (dy < 0)
        distances = np.sqrt(dx**2 + dy**2)
    elif direction == 'top_left':
        mask = (dx < 0) & (dy > 0)
        distances = np.sqrt(dx**2 + dy**2)
    elif direction == 'bottom_right':
        mask = (dx > 0) & (dy < 0)
        distances = np.sqrt(dx**2 + dy**2)
    else:
        return []
    
    # Set invalid points to infinity
    distances[~mask] = np.inf
    
    # Get n closest indices
    closest_indices = np.argsort(distances)[:n]
    closest_indices = closest_indices[distances[closest_indices] < np.inf]
    
    return closest_indices.tolist()

def validate_point(point_idx, X_all, Y_all, u_all, v_all):
    """Validate a point based on velocity vector orientation checks"""
    x = X_all[point_idx]
    y = Y_all[point_idx]
    
    # Get 3 closest points for each corner
    tr_indices = find_closest_points(x, y, X_all, Y_all, 'top_right', n=3)
    bl_indices = find_closest_points(x, y, X_all, Y_all, 'bottom_left', n=3)
    tl_indices = find_closest_points(x, y, X_all, Y_all, 'top_left', n=3)
    br_indices = find_closest_points(x, y, X_all, Y_all, 'bottom_right', n=3)
    
    if len(tr_indices) == 0 or len(bl_indices) == 0 or len(tl_indices) == 0 or len(br_indices) == 0:
        return False
    
    # Check 1: First closest top-right and bottom-left
    tr_idx = tr_indices[0]
    bl_idx = bl_indices[0]
    
    tan_theta_tr = calculate_tan_theta(u_all[tr_idx], v_all[tr_idx])
    tan_theta_bl = calculate_tan_theta(u_all[bl_idx], v_all[bl_idx])
    theta_tr = calculate_theta_degrees(u_all[tr_idx], v_all[tr_idx])
    theta_bl = calculate_theta_degrees(u_all[bl_idx], v_all[bl_idx])
    
    sign_match_1 = np.sign(tan_theta_tr) == np.sign(tan_theta_bl)
    theta_diff_1 = np.abs(theta_tr - theta_bl)
    if theta_diff_1 > 180:
        theta_diff_1 = 360 - theta_diff_1
    theta_diff_check_1 = theta_diff_1 > 90.0
    
    if not (sign_match_1 and theta_diff_check_1):
        return False
    
    # Check 2: First closest top-left and bottom-right
    tl_idx = tl_indices[0]
    br_idx = br_indices[0]
    
    tan_theta_tl = calculate_tan_theta(u_all[tl_idx], v_all[tl_idx])
    tan_theta_br = calculate_tan_theta(u_all[br_idx], v_all[br_idx])
    theta_tl = calculate_theta_degrees(u_all[tl_idx], v_all[tl_idx])
    theta_br = calculate_theta_degrees(u_all[br_idx], v_all[br_idx])
    
    sign_match_2 = np.sign(tan_theta_tl) == np.sign(tan_theta_br)
    theta_diff_2 = np.abs(theta_tl - theta_br)
    if theta_diff_2 > 180:
        theta_diff_2 = 360 - theta_diff_2
    theta_diff_check_2 = theta_diff_2 > 90.0
    
    if not (sign_match_2 and theta_diff_check_2):
        return False
    
    # Check 3: Second closest top-right and bottom-left
    if len(tr_indices) < 2 or len(bl_indices) < 2:
        return False
    
    tr_idx_2 = tr_indices[1]
    bl_idx_2 = bl_indices[1]
    
    tan_theta_tr_2 = calculate_tan_theta(u_all[tr_idx_2], v_all[tr_idx_2])
    tan_theta_bl_2 = calculate_tan_theta(u_all[bl_idx_2], v_all[bl_idx_2])
    theta_tr_2 = calculate_theta_degrees(u_all[tr_idx_2], v_all[tr_idx_2])
    theta_bl_2 = calculate_theta_degrees(u_all[bl_idx_2], v_all[bl_idx_2])
    
    sign_match_3 = np.sign(tan_theta_tr_2) == np.sign(tan_theta_bl_2)
    theta_diff_3 = np.abs(theta_tr_2 - theta_bl_2)
    if theta_diff_3 > 180:
        theta_diff_3 = 360 - theta_diff_3
    theta_diff_check_3 = theta_diff_3 > 90.0
    
    if not (sign_match_3 and theta_diff_check_3):
        return False
    
    # Check 4: Second closest top-left and bottom-right
    if len(tl_indices) < 2 or len(br_indices) < 2:
        return False
    
    tl_idx_2 = tl_indices[1]
    br_idx_2 = br_indices[1]
    
    tan_theta_tl_2 = calculate_tan_theta(u_all[tl_idx_2], v_all[tl_idx_2])
    tan_theta_br_2 = calculate_tan_theta(u_all[br_idx_2], v_all[br_idx_2])
    theta_tl_2 = calculate_theta_degrees(u_all[tl_idx_2], v_all[tl_idx_2])
    theta_br_2 = calculate_theta_degrees(u_all[br_idx_2], v_all[br_idx_2])
    
    sign_match_4 = np.sign(tan_theta_tl_2) == np.sign(tan_theta_br_2)
    theta_diff_4 = np.abs(theta_tl_2 - theta_br_2)
    if theta_diff_4 > 180:
        theta_diff_4 = 360 - theta_diff_4
    theta_diff_check_4 = theta_diff_4 > 90.0
    
    if not (sign_match_4 and theta_diff_check_4):
        return False
    
    # Check 5: Third closest top-right and bottom-left
    if len(tr_indices) < 3 or len(bl_indices) < 3:
        return False
    
    tr_idx_3 = tr_indices[2]
    bl_idx_3 = bl_indices[2]
    
    tan_theta_tr_3 = calculate_tan_theta(u_all[tr_idx_3], v_all[tr_idx_3])
    tan_theta_bl_3 = calculate_tan_theta(u_all[bl_idx_3], v_all[bl_idx_3])
    theta_tr_3 = calculate_theta_degrees(u_all[tr_idx_3], v_all[tr_idx_3])
    theta_bl_3 = calculate_theta_degrees(u_all[bl_idx_3], v_all[bl_idx_3])
    
    sign_match_5 = np.sign(tan_theta_tr_3) == np.sign(tan_theta_bl_3)
    theta_diff_5 = np.abs(theta_tr_3 - theta_bl_3)
    if theta_diff_5 > 180:
        theta_diff_5 = 360 - theta_diff_5
    theta_diff_check_5 = theta_diff_5 > 90.0
    
    if not (sign_match_5 and theta_diff_check_5):
        return False
    
    # Check 6: Third closest top-left and bottom-right
    if len(tl_indices) < 3 or len(br_indices) < 3:
        return False
    
    tl_idx_3 = tl_indices[2]
    br_idx_3 = br_indices[2]
    
    tan_theta_tl_3 = calculate_tan_theta(u_all[tl_idx_3], v_all[tl_idx_3])
    tan_theta_br_3 = calculate_tan_theta(u_all[br_idx_3], v_all[br_idx_3])
    theta_tl_3 = calculate_theta_degrees(u_all[tl_idx_3], v_all[tl_idx_3])
    theta_br_3 = calculate_theta_degrees(u_all[br_idx_3], v_all[br_idx_3])
    
    sign_match_6 = np.sign(tan_theta_tl_3) == np.sign(tan_theta_br_3)
    theta_diff_6 = np.abs(theta_tl_3 - theta_br_3)
    if theta_diff_6 > 180:
        theta_diff_6 = 360 - theta_diff_6
    theta_diff_check_6 = theta_diff_6 > 90.0
    
    # All 6 checks must pass (both sign match and theta difference > 90°)
    return sign_match_6 and theta_diff_check_6

# Validate ALL points in order of increasing velocity
# Only consider points with y < -10
valid_points = []
checked_count = 0
skipped_count = 0

for idx in sorted_indices:
    # Filter: Only consider points with y < -10
    if Y_all[idx] >= -10:
        skipped_count += 1
        continue
    
    checked_count += 1
    is_valid = validate_point(idx, X_all, Y_all, u_all, v_all)
    
    if is_valid:
        valid_points.append(idx)
        if len(valid_points) % 10 == 0:  # Print every 10th valid point
            print(f"Point {idx}: VALID (x={X_all[idx]:.2f}, y={Y_all[idx]:.2f}, vel={velocity_magnitude[idx]:.4f}) - Found {len(valid_points)} valid so far...")
    else:
        if checked_count % 100 == 0:  # Print progress every 100 points
            print(f"Checked {checked_count} points (skipped {skipped_count} with y >= -10), found {len(valid_points)} valid so far...")

print(f"\nTotal points checked (y < -10): {checked_count}")
print(f"Total points skipped (y >= -10): {skipped_count}")
print(f"Total valid points found: {len(valid_points)}")

# Step 3: Select top 5 valid points (lowest velocity magnitude)
# Sort valid points by velocity magnitude and take top 5
if len(valid_points) > 0:
    # Sort valid points by their velocity magnitude
    valid_points_sorted = sorted(valid_points, key=lambda idx: velocity_magnitude[idx])
    final_points = valid_points_sorted[:5]  # Take top 5 (lowest velocity)
    
    print(f"Valid point indices (all): {valid_points}")
    print(f"Velocity magnitudes (all valid): {velocity_magnitude[valid_points]}")
    print(f"\nTop 5 valid points selected (lowest velocity):")
    print(f"  Indices: {final_points}")
    print(f"  Velocity magnitudes: {velocity_magnitude[final_points]}")
    print(f"  Y coordinates: {Y_all[final_points]}")
else:
    final_points = []
print("\n" + "="*60)
print(f"Step 3: Final points: {len(final_points)}")
print("="*60)

if len(final_points) < 2:
    print("Error: Need at least 2 valid points for clustering!")
    exit(1)

# Step 4: Fixed centroid clustering for each pair of final points
print("\n" + "="*60)
print("Step 4: Fixed centroid clustering for each pair")
print("="*60)

# Normalize velocity (direction only) for features
speed = np.sqrt(u_all**2 + v_all**2) + 1e-8
u_hat = u_all / speed
v_hat = v_all / speed

# Feature vector per point
features = np.column_stack([
    X_all,
    Y_all,
    u_hat,
    v_hat
])

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Generate all pairs from final points
from itertools import combinations
pairs = list(combinations(final_points, 2))
print(f"Number of pairs to test: {len(pairs)}")

best_error = np.inf
best_pair = None
best_labels = None
best_centroids = None

for i, (p1_idx, p2_idx) in enumerate(pairs):
    # Use the two points as fixed centroids (in scaled feature space)
    centroid1 = features_scaled[p1_idx].reshape(1, -1)
    centroid2 = features_scaled[p2_idx].reshape(1, -1)
    fixed_centroids = np.vstack([centroid1, centroid2])
    
    # Assign each point to the nearest fixed centroid
    # Calculate distances from all points to both centroids
    distances_to_centroid1 = np.linalg.norm(features_scaled - centroid1, axis=1)
    distances_to_centroid2 = np.linalg.norm(features_scaled - centroid2, axis=1)
    
    # Assign labels based on nearest centroid
    labels = np.where(distances_to_centroid1 < distances_to_centroid2, 0, 1)
    
    # Calculate error (sum of squared distances from points to their assigned centroids)
    error = 0.0
    for point_idx in range(len(features_scaled)):
        assigned_centroid = fixed_centroids[labels[point_idx]]
        dist_sq = np.sum((features_scaled[point_idx] - assigned_centroid) ** 2)
        error += dist_sq
    
    print(f"Pair {i+1}: Points ({p1_idx}, {p2_idx}) - Error: {error:.4f}")
    
    if error < best_error:
        best_error = error
        best_pair = (p1_idx, p2_idx)
        best_labels = labels
        best_centroids = fixed_centroids

print(f"\nBest pair: {best_pair} with error: {best_error:.4f}")

# Step 5: New clustering method based on sin(theta) rules
print("\n" + "="*60)
print("Step 5: Applying sin(theta) based clustering rules")
print("="*60)

# Get cluster centroids (in original feature space)
centroids_original = scaler.inverse_transform(best_centroids)

centroid_X = centroids_original[:, 0]
centroid_Y = centroids_original[:, 1]

print(f"Final cluster centroids (X, Y):")
for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    print(f"  Cluster {i}: ({cx:.2f}, {cy:.2f})")

def find_second_closest_in_quadrant(centroid_x, centroid_y, X_all, Y_all, direction):
    """Find the second closest point in a given quadrant to a centroid"""
    dx = X_all - centroid_x
    dy = Y_all - centroid_y
    
    if direction == 'top_right':
        mask = (dx > 0) & (dy > 0)
    elif direction == 'top_left':
        mask = (dx < 0) & (dy > 0)
    elif direction == 'bottom_left':
        mask = (dx < 0) & (dy < 0)
    elif direction == 'bottom_right':
        mask = (dx > 0) & (dy < 0)
    else:
        return None
    
    if not np.any(mask):
        return None
    
    distances = np.sqrt(dx**2 + dy**2)
    distances[~mask] = np.inf
    
    # Get the two closest points
    sorted_indices = np.argsort(distances)
    valid_indices = sorted_indices[distances[sorted_indices] < np.inf]
    
    if len(valid_indices) < 2:
        return None
    
    # Return the second closest (index 1)
    return valid_indices[1]

def get_quadrant(x, y, centroid_x, centroid_y):
    """Determine which quadrant a point is in relative to centroid"""
    dx = x - centroid_x
    dy = y - centroid_y
    if dx > 0 and dy > 0:
        return 'top_right'
    elif dx < 0 and dy > 0:
        return 'top_left'
    elif dx < 0 and dy < 0:
        return 'bottom_left'
    elif dx > 0 and dy < 0:
        return 'bottom_right'
    else:
        return 'on_axis'  # On x or y axis

def calculate_sin_theta(u, v):
    """Calculate sin(theta) where theta is angle with x-axis"""
    magnitude = np.sqrt(u**2 + v**2) + 1e-8
    return v / magnitude  # sin(theta) = v / |v|

# For each centroid, find second closest point in each quadrant and calculate sin(theta)
# Structure: centroid_quadrant_sin_thetas[centroid_id][quadrant] = sin(theta)
centroid_quadrant_sin_thetas = []
quadrants = ['top_right', 'top_left', 'bottom_left', 'bottom_right']

for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    quadrant_sin_thetas = {}
    print(f"\nCentroid {i} ({cx:.2f}, {cy:.2f}):")
    
    for quadrant in quadrants:
        second_closest_idx = find_second_closest_in_quadrant(cx, cy, X_all, Y_all, quadrant)
        if second_closest_idx is None:
            print(f"  Warning: No second closest {quadrant} point found, using default sin(theta)=0")
            quadrant_sin_thetas[quadrant] = 0.0
        else:
            sin_theta = calculate_sin_theta(u_all[second_closest_idx], v_all[second_closest_idx])
            quadrant_sin_thetas[quadrant] = sin_theta
            print(f"  Second closest {quadrant} point index={second_closest_idx}, sin(theta)={sin_theta:.4f}")
    
    centroid_quadrant_sin_thetas.append(quadrant_sin_thetas)

# Step 6: Find closest boundary points and determine circular boundaries
print("\n" + "="*60)
print("Step 6: Finding closest boundary points for circular boundaries")
print("="*60)

def calculate_angle_between_vectors(v1, v2):
    """Calculate the angle (in degrees) between two vectors using dot product"""
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0
    
    # Dot product
    dot_product = np.dot(v1, v2)
    
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)
    
    # Calculate angle in degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Find boundary points (where sin(theta) sign changes OR angle < 30 deg or > 150 deg) for each centroid
centroid_radii = []

for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    quadrant_refs = centroid_quadrant_sin_thetas[i]
    boundary_points = []
    
    # Find all points that satisfy boundary conditions
    for point_idx in range(len(X_all)):
        x, y = X_all[point_idx], Y_all[point_idx]
        quadrant = get_quadrant(x, y, cx, cy)
        
        if quadrant == 'on_axis':
            continue
        
        is_boundary = False
        boundary_reason = ""
        
        # Criterion 1: Check if sin(theta) sign changes
        sin_theta_ref = quadrant_refs[quadrant]
        sin_theta_point = calculate_sin_theta(u_all[point_idx], v_all[point_idx])
        
        if np.sign(sin_theta_point) != np.sign(sin_theta_ref):
            is_boundary = True
            boundary_reason = "sin(theta) sign change"
        
        # Criterion 2: Check if angle between radius vector and velocity vector < 30 degrees or > 150 degrees
        # Radius vector: from centroid to point
        radius_vector = np.array([x - cx, y - cy])
        # Velocity vector at this point
        velocity_vector = np.array([u_all[point_idx], v_all[point_idx]])
        
        # Calculate angle between radius and velocity vectors
        angle_deg = calculate_angle_between_vectors(radius_vector, velocity_vector)
        
        if angle_deg < 30.0 or angle_deg > 150.0:
            is_boundary = True
            if boundary_reason:
                boundary_reason += f" OR angle < 30° or > 150°"
            else:
                if angle_deg < 30.0:
                    boundary_reason = f"angle < 30° ({angle_deg:.1f}°)"
                else:
                    boundary_reason = f"angle > 150° ({angle_deg:.1f}°)"
        
        # If either criterion is met, this is a boundary point
        if is_boundary:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            boundary_points.append((point_idx, distance, x, y, boundary_reason))
    
    # Find the closest boundary point
    if len(boundary_points) > 0:
        boundary_points.sort(key=lambda x: x[1])  # Sort by distance
        closest_boundary = boundary_points[0]
        radius = closest_boundary[1]
        centroid_radii.append(radius)
        print(f"Centroid {i}: Closest boundary point at distance {radius:.4f} (point {closest_boundary[0]}, x={closest_boundary[2]:.2f}, y={closest_boundary[3]:.2f}, reason: {closest_boundary[4]})")
    else:
        # If no boundary points found, use a default radius (e.g., mean distance to all points)
        distances_to_centroid = np.sqrt((X_all - cx)**2 + (Y_all - cy)**2)
        radius = np.mean(distances_to_centroid) * 0.5  # Use half of mean distance as fallback
        centroid_radii.append(radius)
        print(f"Centroid {i}: No boundary points found, using default radius {radius:.4f}")

# Apply circular boundary clustering
new_labels = np.full(len(X_all), -1)  # -1 means outside/not assigned

for point_idx in range(len(X_all)):
    x, y = X_all[point_idx], Y_all[point_idx]
    distances_to_centroids = []
    
    for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
        radius = centroid_radii[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Check if point is inside this circle
        if distance <= radius:
            distances_to_centroids.append((i, distance))
    
    # If point is inside one or more circles, assign to the closest centroid
    if len(distances_to_centroids) > 0:
        distances_to_centroids.sort(key=lambda x: x[1])  # Sort by distance
        new_labels[point_idx] = distances_to_centroids[0][0]

# Count points in each cluster
for i in range(2):
    count = np.sum(new_labels == i)
    print(f"Cluster {i}: {count} points")

# Step 7: Final plotting with circular boundaries
print("\n" + "="*60)
print("Step 7: Final plotting with circular boundaries")
print("="*60)

# Create visualization
plt.figure(figsize=(14, 12))

# Define distinct, vibrant colors
cluster_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF']

# Create a grid for boundary visualization
x_min, x_max = X_all.min(), X_all.max()
y_min, y_max = Y_all.min(), Y_all.max()
x_range = x_max - x_min
y_range = y_max - y_min
margin_x = x_range * 0.1
margin_y = y_range * 0.1

# Create fine grid for circular boundary visualization
grid_resolution = 300
x_grid = np.linspace(x_min - margin_x, x_max + margin_x, grid_resolution)
y_grid = np.linspace(y_min - margin_y, y_max + margin_y, grid_resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Interpolate velocity components onto grid for PIV field visualization
points = np.column_stack([X_all, Y_all])
u_grid = griddata(points, u_all, (X_grid, Y_grid), method='linear')
v_grid = griddata(points, v_all, (X_grid, Y_grid), method='linear')

# Assign grid points to clusters based on circular boundaries
grid_labels = np.full(X_grid.shape, -1)

for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    radius = centroid_radii[i]
    
    # Calculate distance from each grid point to centroid
    distances = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
    
    # Points inside the circle belong to this cluster
    inside_circle = distances <= radius
    grid_labels[inside_circle] = i

# Plot cluster regions with circular shading
from matplotlib.patches import Circle

# Plot velocity vectors underneath clusters (lower zorder)
skip = 5  # Show every 5th vector for clarity
plt.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
          u_grid[::skip, ::skip], v_grid[::skip, ::skip],
          color='black', scale=6, scale_units='xy', angles='xy', 
          width=0.001, alpha=0.9, zorder=1)

for cluster_id in range(2):
    cx, cy = centroid_X[cluster_id], centroid_Y[cluster_id]
    radius = centroid_radii[cluster_id]
    
    # Shade the circular cluster region
    circle = Circle((cx, cy), radius, color=cluster_colors[cluster_id], 
                   alpha=0.3, zorder=0)
    plt.gca().add_patch(circle)
    
    # Draw circular boundary line
    boundary_circle = Circle((cx, cy), radius, fill=False, 
                            edgecolor=cluster_colors[cluster_id], 
                            linewidth=2, linestyle='-', zorder=3)
    plt.gca().add_patch(boundary_circle)
    
    # Plot data points in this cluster (only those inside the circle)
    data_mask = new_labels == cluster_id
    if np.any(data_mask):
        cluster_X = X_all[data_mask]
        cluster_Y = Y_all[data_mask]
        
        # Also filter by circular boundary
        distances_to_centroid = np.sqrt((cluster_X - cx)**2 + (cluster_Y - cy)**2)
        inside_mask = distances_to_centroid <= radius
        
        if np.any(inside_mask):
            plt.scatter(cluster_X[inside_mask], cluster_Y[inside_mask], 
                       c=cluster_colors[cluster_id], s=8, alpha=0.7, 
                       edgecolors='none', label=f'Cluster {cluster_id}', zorder=2)

# Highlight cluster centroids
for i, (cx, cy) in enumerate(zip(centroid_X, centroid_Y)):
    plt.scatter(cx, cy, c=cluster_colors[i], marker='*', s=1200, 
               edgecolors='black', linewidths=4, 
               label=f'Centroid {i}' if i == 0 else '', zorder=10)
    
    # Add labels for centroids
    plt.annotate(f'C{i}', (cx, cy), xytext=(10, 10), textcoords='offset points',
                fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=cluster_colors[i], 
                         edgecolor='black', linewidth=3, alpha=0.95))

plt.title(f"Vortex zones - {CAD} (Best pair: {best_pair})", 
          fontsize=14, fontweight='bold')
plt.xlabel("X (mm)", fontsize=12)
plt.ylabel("Y (mm)", fontsize=12)
plt.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
plt.axis('equal')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print(f"\nVisualization complete!")
