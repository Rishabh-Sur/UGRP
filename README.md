# Vortex Zone Detection using K-Means Clustering on EngineBench Dataset

This project implements K-means clustering to detect vortex zones in engine tumble plane velocity data from the EngineBench dataset.

## Dataset

### Download
The dataset can be downloaded from Kaggle:
- **Dataset Link**: [EngineBench on Kaggle](https://www.kaggle.com/datasets/samueljbaker/enginebench?select=TP.h5)
- **File Used**: `TP.h5` (Tumble Plane data)

### Dataset Structure
The EngineBench dataset contains Particle Image Velocimetry (PIV) data from engine experiments. The tumble plane data is stored in HDF5 format with the following structure:

- **Shape**: `(33385, 4131, 4)`
- **Dimensions**:
  - **Dimension 0 (33385)**: Snapshots/Timesteps - different time instances or measurement cycles
  - **Dimension 1 (4131)**: Spatial measurement points - measurement locations in the tumble plane
  - **Dimension 2 (4)**: Components - `[X-coordinate, Y-coordinate, U-velocity, V-velocity]`

### Vector Encoding
The dataset includes a `vector_encoding` attribute that maps Crank Angle Degree (CAD) values to indices:
```python
{
    "cad040": 0, "cad050": 1, "cad060": 2, "cad070": 3, "cad080": 4,
    "cad090": 5, "cad100": 6, "cad180": 7, "cad220": 8, "cad260": 9, "cad300": 10
}
```

## Methodology

### 1. Data Partitioning

The dataset is partitioned along dimension 0 (snapshots) into **11 equal parts**:

- **Total snapshots**: 33,385
- **Number of partitions**: 11
- **Partition size**: 33,385 ÷ 11 ≈ 3,035 snapshots per partition

Each partition represents a different phase or condition in the engine cycle, corresponding to different CAD values.

### 2. Ensemble Creation

For each of the 11 partitions, an **ensemble velocity field** is created by averaging the **first 10 snapshots** of that partition:

- **Snapshots per ensemble**: 10
- **Total ensembles**: 11
- **Ensemble calculation**: 
  ```python
  ensemble = mean(partition_data[0:10, :, :], axis=0)
  ```

This ensemble approach:
- Reduces noise by averaging multiple snapshots
- Captures the characteristic flow pattern for each partition
- Provides a representative velocity field for each CAD phase

### 3. Feature Extraction

For each spatial point in all ensembles, features are extracted:

1. **Spatial coordinates**: X and Y coordinates (components 0 and 1)
2. **Velocity components**: U and V velocities (components 2 and 3)
3. **Normalized velocity direction**: 
   - Speed: `sqrt(u² + v²)`
   - Normalized U: `u / speed`
   - Normalized V: `v / speed`

The feature vector for each point is:
```python
features = [X, Y, u_hat, v_hat]
```

Where:
- `X, Y`: Spatial coordinates
- `u_hat, v_hat`: Normalized velocity direction (unit vectors)

### 4. Feature Scaling

All features are standardized using `StandardScaler` from scikit-learn:
- Centers features to zero mean
- Scales features to unit variance
- Ensures all features contribute equally to clustering

### 5. K-Means Clustering

K-means clustering is applied to identify vortex zones:

- **Algorithm**: K-Means from scikit-learn
- **Number of clusters**: Configurable (default: 2-4)
- **Initialization**: 20 random initializations (`n_init=20`)
- **Random state**: 0 (for reproducibility)

The clustering groups spatial points based on:
- Their spatial location (X, Y)
- Their velocity direction (normalized u, v)

Points with similar flow patterns and locations are grouped together, identifying distinct vortex zones.

## Implementation

### Requirements
```bash
pip install numpy h5py matplotlib scikit-learn
```

### Usage
```bash
python kmeans.py
```

### Key Parameters
- `NUM_PARTITIONS = 11`: Number of partitions along dimension 0
- `SNAPSHOTS_PER_ENSEMBLE = 10`: Number of snapshots to average for each ensemble
- `K = 2`: Number of clusters for K-means (adjustable: 3-6 recommended)

## Output

The script generates two visualizations:

1. **Grid of Individual Ensembles**: 11 subplots showing clustering results for each ensemble (labeled with CAD numbers from vector encoding)
2. **Combined Visualization**: All 11 ensembles plotted together with cluster assignments

Each visualization uses color coding to represent different vortex zones identified by K-means clustering.

## Advantages of This Approach

1. **Noise Reduction**: Ensemble averaging reduces measurement noise and captures stable flow patterns
2. **Temporal Coverage**: 11 partitions cover different phases of the engine cycle
3. **Robust Clustering**: Using normalized velocity directions makes clustering invariant to velocity magnitude
4. **Spatial Awareness**: Including X, Y coordinates ensures spatial coherence in cluster assignments

## References

- EngineBench Dataset: [Kaggle](https://www.kaggle.com/datasets/samueljbaker/enginebench)
- EngineBench Documentation: [Oxford TPSRG](https://eng.ox.ac.uk/tpsrg/research/enginebench/)

## Notes

- The dataset structure is `(Snapshots, Spatial_Points, Components)`, not a regular 2D grid
- Each spatial point stores its own coordinates, allowing for irregular measurement grids
- The ensemble approach provides a balance between temporal resolution and noise reduction

