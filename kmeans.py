import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

H5_PATH = "data/TP.h5"
GROUP = "r1300_p40"

# Partition parameters
NUM_PARTITIONS = 11
SNAPSHOTS_PER_ENSEMBLE = 10

with h5py.File(H5_PATH, "r") as f:
    group = f[GROUP]
    piv = group["PIV"]
    
    # Read vector encoding to get CAD numbers
    vector_encoding_raw = group.attrs.get('vector_encoding', {})
    
    # Create mapping from ensemble index to CAD number
    # vector_encoding format: {"cad040": 0, "cad050": 1, ...}
    # We need to map index -> CAD number
    cad_mapping = {}
    if vector_encoding_raw:
        # Handle if it's stored as a string (JSON-like)
        import json
        if isinstance(vector_encoding_raw, str):
            vector_encoding = json.loads(vector_encoding_raw)
        else:
            vector_encoding = vector_encoding_raw
        
        # Sort by value to get sequential order
        sorted_encoding = sorted(vector_encoding.items(), key=lambda x: x[1])
        for cad_key, idx in sorted_encoding:
            # Extract CAD number from "cad040" -> "040"
            cad_number = cad_key.replace("cad", "")
            cad_mapping[idx] = cad_number
        print(f"CAD mapping: {cad_mapping}")
    else:
        # Fallback if vector_encoding not found
        for i in range(NUM_PARTITIONS):
            cad_mapping[i] = str(i).zfill(3)
    
    total_snapshots = piv.shape[0]  # 33385
    num_spatial_points = piv.shape[1]  # 4131
    
    partition_size = total_snapshots // NUM_PARTITIONS  # 33385 / 11 ≈ 3035
    
    print(f"Total snapshots: {total_snapshots}")
    print(f"Partition size: {partition_size}")
    print(f"Creating {NUM_PARTITIONS} ensembles from first {SNAPSHOTS_PER_ENSEMBLE} snapshots of each partition...")
    
    # Store ensembles: each ensemble is averaged over first 10 snapshots of a partition
    ensembles = []
    
    for partition_idx in range(NUM_PARTITIONS):
        start_idx = partition_idx * partition_size
        end_idx = start_idx + SNAPSHOTS_PER_ENSEMBLE
        
        # Extract first 10 snapshots from this partition
        partition_data = piv[start_idx:end_idx, :, :]  # Shape: (10, 4131, 4)
        
        # Average over snapshots to create ensemble
        # Components: [X-coord, Y-coord, U-velocity, V-velocity]
        ensemble = np.mean(partition_data, axis=0)  # Shape: (4131, 4)
        
        ensembles.append(ensemble)
        print(f"  Partition {partition_idx}: snapshots {start_idx} to {end_idx-1}, ensemble shape: {ensemble.shape}")
    
    # Stack all ensembles together
    # Shape: (11 * 4131, 4) = (45441, 4)
    all_ensembles = np.vstack(ensembles)
    print(f"\nStacked ensembles shape: {all_ensembles.shape}")
    
    # Extract components from all ensembles
    # Components: [X-coord, Y-coord, U-velocity, V-velocity]
    X_all = all_ensembles[:, 0]  # X coordinates
    Y_all = all_ensembles[:, 1]  # Y coordinates
    u_all = all_ensembles[:, 2]  # U velocities
    v_all = all_ensembles[:, 3]  # V velocities
    
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
    
    # Reshape for visualization
    # Since we have 11 ensembles × 4131 points = 45441 points total
    # We'll visualize each ensemble separately or create a combined visualization
    num_points_per_ensemble = num_spatial_points
    
    # Visualize each ensemble's clustering results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for ensemble_idx in range(NUM_PARTITIONS):
        start_point = ensemble_idx * num_points_per_ensemble
        end_point = start_point + num_points_per_ensemble
        
        ensemble_labels = labels[start_point:end_point]
        ensemble_X = X_all[start_point:end_point]
        ensemble_Y = Y_all[start_point:end_point]
        ensemble_u = u_all[start_point:end_point]
        ensemble_v = v_all[start_point:end_point]
        
        # Create scatter plot colored by cluster
        ax = axes[ensemble_idx]
        scatter = ax.scatter(ensemble_X, ensemble_Y, c=ensemble_labels, 
                            cmap="tab10", s=1, alpha=0.6)
        # Use CAD number from vector encoding
        cad_number = cad_mapping.get(ensemble_idx, str(ensemble_idx).zfill(3))
        ax.set_title(f"CAD {cad_number}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(NUM_PARTITIONS, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Vortex zones detected using K-means (11 ensembles)", y=1.02)
    plt.show()
    
    # Also create a combined visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_all, Y_all, c=labels, cmap="tab10", s=1, alpha=0.5)
    plt.colorbar(scatter, label="K-means zone ID")
    plt.title("Vortex zones detected using K-means (All 11 ensembles combined)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()
