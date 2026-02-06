# Vortex Detection Methods for PIV Data

This repository implements multiple approaches for detecting and identifying vortex regions in 2D Particle Image Velocimetry (PIV) data from engine tumble plane measurements.

## Dataset

### Data Format
- **File**: `data/data.txt`
- **Format**: Space-separated text file with header row
- **Columns**: 
  - Column 0: X coordinates (mm)
  - Column 1: Y coordinates (mm)
  - Column 2: U velocity component (m/s)
  - Column 3: V velocity component (m/s)
- **Data**: Ensemble-averaged PIV velocity field for CAD 100 (1300 rpm, 95 kPa)

### Data Loading
All scripts load data using:
```python
data = np.loadtxt("data/data.txt", skiprows=1)
X_all = data[:, 0]  # X coordinates (mm)
Y_all = data[:, 1]  # Y coordinates (mm)
u_all = data[:, 2]  # U velocities (m/s)
v_all = data[:, 3]  # V velocities (m/s)
```

## Implemented Methods

### 1. Velocity Field Visualization (`plotting.py`)

**Purpose**: Visualize the raw PIV velocity field with velocity magnitude colormap and velocity vectors.

**Methodology**:
- Interpolates scattered PIV data onto a regular 200×200 grid using `griddata`
- Creates a convex hull mask to identify valid measurement domain
- Computes velocity magnitude: `|v| = √(u² + v²)`
- Visualizes velocity magnitude as background colormap (viridis)
- Overlays white velocity vectors (quiver plot) for flow direction

**Key Features**:
- Handles irregular measurement domain using convex hull
- Linear interpolation for smooth visualization
- Configurable vector density (default: every 4th vector)

**Usage**:
```bash
python plotting.py
```

---

### 2. Vorticity Magnitude Criterion (`omega.py`)

**Purpose**: Identify vortex regions based on vorticity magnitude threshold.

**Mathematical Formulation**:
- **Vorticity**: `ω_z = ∂v/∂x - ∂u/∂y` (out-of-plane component for 2D flow)
- **Vorticity Magnitude**: `|ω| = |ω_z|`
- **Vortex Condition**: `|ω| > ω_th` where `ω_th` is the 60th percentile (top 40%)

**Methodology**:
1. Interpolate velocity components onto regular grid
2. Compute spatial gradients using `np.gradient`
3. Calculate vorticity: `ω = dv/dx - du/dy`
4. Compute vorticity magnitude: `|ω|`
5. Set threshold as 60th percentile of all valid vorticity values
6. Identify vortex regions where `|ω| > ω_th`

**Physical Interpretation**:
- Measures local angular velocity
- Indicates rotation intensity
- Regions of strong rotational activity

**Advantages**:
- Simple and computationally efficient
- Good for initial visualization
- Provides approximate indication of vortex strength

**Limitations**:
- Cannot distinguish shear from rotation
- Produces false positives in pure shear flows
- Threshold-dependent
- No vortex center or boundary definition

**Usage**:
```bash
python omega.py
```

**Output**: Visualization with vorticity magnitude colormap, red overlay for vortex regions (|ω| > threshold), and white dashed contour at threshold boundary.

---

### 3. Q-Criterion (`Q.py`)

**Purpose**: Identify vortex regions where rotation dominates over strain.

**Mathematical Formulation**:
- **Velocity Gradient Tensor** (2×2 for 2D): 
  ```
  A = [∂u/∂x  ∂u/∂y]
      [∂v/∂x  ∂v/∂y]
  ```
- **Symmetric Part** (Strain-rate tensor): `S = (1/2)(A + Aᵀ)`
- **Antisymmetric Part** (Rotation tensor): `Ω = (1/2)(A - Aᵀ)`
- **Q-Criterion**: `Q = (1/2)(||Ω||² - ||S||²)`
- **Vortex Condition**: `Q > 0`

**Methodology**:
1. Compute velocity gradient tensor A
2. Decompose into symmetric (S) and antisymmetric (Ω) parts
3. Calculate norms: `||Ω||² = 2×Ω_xy²`, `||S||² = S_xx² + S_yy² + 2×S_xy²`
4. Compute Q: `Q = 0.5 × (||Ω||² - ||S||²)`
5. Identify vortex regions where `Q > 0`

**Physical Interpretation**:
- Rotation dominates over strain
- Associated with low-pressure regions
- Clear separation between rotational and strain-dominated zones

**Advantages**:
- Distinguishes rotation from pure strain
- Provides vortex regions (connected Q > 0 zones)
- Approximate vortex cores via Q-maxima
- Relative comparison of vortex intensity

**Limitations**:
- Threshold dependent (Q > 0 is standard)
- Q-max ≠ guaranteed vortex center
- Sensitive to noise in PIV data
- No temporal coherence

**Usage**:
```bash
python Q.py
```

**Output**: Visualization with Q values as colormap, red overlay for vortex regions (Q > 0), and white dashed contour at Q = 0 boundary.

---

### 4. Δ (Discriminant) Criterion (`delta.py`)

**Purpose**: Confirm true swirling motion by detecting complex conjugate eigenvalues.

**Mathematical Formulation**:
- **Characteristic Equation**: `λ³ + Qλ + R = 0` (for 3D incompressible flow)
- **Invariants**:
  - `Q = -(1/2) × trace(A²)` (second invariant)
  - `R = -det(A)` (third invariant)
- **Discriminant**: `Δ = (Q/3)³ + (R/2)²`
- **Vortex Condition**: `Δ > 0`

**Methodology**:
1. Compute velocity gradient tensor A (2×2)
2. Calculate Q: `Q = -0.5 × trace(A²)`
3. Calculate R: `R = -det(A)`
4. Compute discriminant: `Δ = (Q/3)³ + (R/2)²`
5. Identify vortex regions where `Δ > 0`

**Physical Interpretation**:
- Presence of complex conjugate eigenvalues
- Indicates spiralling streamlines
- Confirms true swirling motion
- Distinguishes vortex-type critical points from saddles/nodes

**Advantages**:
- Strong theoretical basis for vortex existence
- Confirms true swirling motion
- Distinguishes vortices from other critical points

**Limitations**:
- Highly sensitive to gradient noise
- Computationally expensive
- No vortex size or strength measure

**Usage**:
```bash
python delta.py
```

**Output**: Visualization with Δ values as colormap, red overlay for vortex regions (Δ > 0), and white dashed contour at Δ = 0 boundary.

---

### 5. K-Means Clustering (`kmeans.py`)

**Purpose**: Detect vortex zones using unsupervised clustering based on spatial location and velocity direction.

**Methodology**:
1. **Data Loading**: Load first 2551 rows from `data/data.txt`
2. **Feature Extraction**:
   - Spatial coordinates: `[X, Y]`
   - Normalized velocity direction: `[û, v̂]` where `û = u/|v|`, `v̂ = v/|v|`
   - Feature vector: `[X, Y, û, v̂]`
3. **Feature Scaling**: Standardize features using `StandardScaler` (zero mean, unit variance)
4. **Clustering**: Apply K-Means with `K = 2` clusters
   - Algorithm: scikit-learn `KMeans`
   - Initializations: 20 random initializations (`n_init=20`)
   - Random state: 0 (for reproducibility)

**Key Features**:
- Uses normalized velocity direction (invariant to velocity magnitude)
- Includes spatial coordinates for spatial coherence
- Identifies distinct vortex zones based on flow patterns

**Usage**:
```bash
python kmeans.py
```

**Output**: Scatter plot with color-coded clusters, star-shaped centroids with labels (C0, C1), and legend.

---

### 6. Hybrid GMM + K-Means Clustering (`gmm_kmeans.py`)

**Purpose**: Combine probabilistic clustering (GMM) with deterministic refinement (K-Means) for robust vortex detection.

**Methodology**:

**Step 0: Feature Computation**
- Compute vorticity: `ω = ∂v/∂x - ∂u/∂y`
- Compute swirling strength (λ_ci): imaginary part of eigenvalues of velocity gradient tensor
- Feature vector: `[X, Y, u, v, vorticity, swirling_strength]`

**Step 1: GMM Clustering**
- Fit Gaussian Mixture Model with `K = 2` components
- Covariance type: full (allows ellipsoidal clusters)
- Get initial cluster assignments and centroids
- Calculate BIC and AIC for model evaluation

**Step 2: K-Means Refinement**
- Use GMM centroids as initial centers for K-Means
- Refine cluster assignments using deterministic K-Means
- Calculate final inertia (within-cluster sum of squares)

**Step 3-4: Visualization**
- Side-by-side comparison of GMM and K-Means results
- Detailed plot with velocity vectors overlaid
- Save visualizations as PNG files

**Key Features**:
- Combines probabilistic and deterministic approaches
- Uses physically meaningful features (vorticity, swirling strength)
- Provides both GMM and refined K-Means results
- Includes velocity field context in visualization

**Usage**:
```bash
python gmm_kmeans.py
```

**Output**: 
- Two side-by-side plots (GMM vs K-Means)
- Detailed plot with velocity vectors
- Saved as `gmm_kmeans_clustering.png` and `gmm_kmeans_detailed.png`

---

### 7. Custom Vortex Detection Method (`new_method.py`)

**Purpose**: Multi-step physics-based approach combining velocity-based point validation with fixed-centroid clustering and circular boundary definition.

**Methodology**:

**Step 1-2: Point Validation & Selection**
- Sort all points by increasing velocity magnitude: `|v| = √(u² + v²)`
- Filter: Only consider points with `y < -10`
- **Validation Criteria** (6 checks per point):
  - For each of 3 closest points in 4 quadrants (top-right, bottom-left, top-left, bottom-right):
    - **Sign check**: `tan(θ)` signs must match between opposite quadrants (TR↔BL, TL↔BR)
    - **Angle check**: Absolute difference of `θ` (angle with x-axis) must be > 90°
  - Point is **valid** only if all 6 checks pass
- **Output**: Top 5 valid points with lowest velocity magnitude

**Step 3: Fixed-Centroid Clustering**
- Feature vector: `[X, Y, û, v̂]` where `(û, v̂)` are normalized velocity components
- For each pair of the 5 valid points:
  - Use pair as **fixed centroids** (no iteration)
  - Assign all points to nearest centroid
  - Calculate inertia (sum of squared distances)
- **Selection**: Choose pair with **lowest inertia** as final cluster centers

**Step 4-5: Circular Boundary Definition**
- For each centroid, find 2nd closest point in each quadrant
- **Boundary Criteria**: A point is a boundary if:
  1. `sin(θ)` sign changes relative to quadrant reference, OR
  2. Angle between radius vector (centroid→point) and velocity vector is < 30° or > 150°
- **Radius**: Distance to **closest boundary point** defines circular cluster region

**Step 6: Visualization**
- Shaded circular regions around centroids with computed radii
- Points inside circles assigned to nearest centroid
- Black velocity vectors overlaid for flow context
- Star-shaped centroids with labels (C0, C1)

**Key Features**:
- No arbitrary thresholds: Uses physical flow characteristics
- Rotation-based validation: Ensures true swirling motion
- Fixed centroids: Prevents convergence issues
- Circular boundaries: Physically meaningful vortex regions

**Usage**:
```bash
python new_method.py
```

**Output**: Visualization with circular cluster regions, velocity vectors, and labeled centroids.

---

## Requirements

```bash
pip install numpy matplotlib scikit-learn scipy
```

## File Structure

```
.
├── data/
│   └── data.txt              # Ensemble-averaged PIV data
├── plotting.py                # Velocity field visualization
├── omega.py                   # Vorticity magnitude criterion
├── Q.py                       # Q-criterion
├── delta.py                   # Δ (Discriminant) criterion
├── kmeans.py                  # K-Means clustering
├── gmm_kmeans.py              # Hybrid GMM + K-Means
├── new_method.py              # Custom vortex detection method
└── README.md                  # This file
```

## Comparison of Methods

| Method | Basis | Threshold | Strengths | Limitations |
|--------|-------|-----------|-----------|------------|
| **Vorticity Magnitude** | Local angular velocity | 60th percentile | Simple, fast | Cannot distinguish shear from rotation |
| **Q-Criterion** | Rotation vs. strain | Q > 0 | Distinguishes rotation from strain | Sensitive to noise |
| **Δ Criterion** | Complex eigenvalues | Δ > 0 | Strong theoretical basis | Highly sensitive to noise |
| **K-Means** | Spatial + velocity direction | K = 2 | Simple clustering | Requires predefined K |
| **GMM + K-Means** | Multi-feature clustering | K = 2 | Probabilistic + deterministic | Computationally expensive |
| **Custom Method** | Physics-based validation | None | No arbitrary thresholds | Complex implementation |

## Notes

- All methods operate on 2D PIV data (tumble plane)
- Data is ensemble-averaged for CAD 100 (1300 rpm, 95 kPa)
- Grid interpolation uses 200×200 resolution for gradient-based methods
- Convex hull masking ensures only valid measurement domain is analyzed
- Velocity vectors are visualized for flow context in all methods

## References

- EngineBench Dataset: [Dataset](https://eng.ox.ac.uk/tpsrg/research/enginebench)
- Q-Criterion: Hunt et al. (1988) "Eddies, streams, and convergence zones in turbulent flows"
- Δ Criterion: Chong et al. (1990) "A general classification of three-dimensional flow fields"
