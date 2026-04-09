"""
02_kmeans_clustering.py
=======================
K-Means clustering with silhouette analysis, PCA exploration,
cluster merging, and spectral response of clusters.

Inputs
------
checkpoint_01.pkl

Outputs
-------
checkpoint_02.pkl :
    pca, X_pca, cluster_map, loaded_models
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    DATA_DIR, PLOT_DIR, MODEL_DIR, HEIGHT, WIDTH,
    ALL_BANDS, REFLECTIVE_BANDS, EMISSIVE_BANDS, BAND_WAVELENGTHS,
    apply_plot_style, add_geo_ticks, save_checkpoint, load_checkpoint,
)
from functions_project2 import (
    plot_pca_scree, pca_eigen_table, plot_pca_discrete, plot_pca_rgb,
    plot_pca_loadings, plot_rgb_subset,
    merge_and_relabel, plot_kmeans_spectral_response,
)

apply_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
ckpt = load_checkpoint('checkpoint_01.pkl')
X_raw    = ckpt['X_raw']
X_scaled = ckpt['X_scaled']
rgb      = ckpt['rgb']
lon      = ckpt['lon']
lat      = ckpt['lat']

ds = xr.open_dataset(ckpt['ds_path'])
ds = ds.isel(y=slice(2935-256, 2935+256), x=slice(1448-320, 1448+320))
ds = ds.isel(y=slice(None, None, -1), x=slice(None, None, -1))

# ══════════════════════════════════════════════════════════════════════════════
# 2. K-MEANS CLUSTERING  (k = 2..12)
# ══════════════════════════════════════════════════════════════════════════════
# -- Train (uncomment to re-run, otherwise load saved) --
# results = {}
# for k in range(2, 13):
#     model = KMeans(n_clusters=k, random_state=670).fit(X_scaled)
#     results[k] = {
#         'model':   model,
#         'labels':  model.labels_,
#         'inertia': model.inertia_,
#     }
#     print(f"  k={k:>2d}  inertia={model.inertia_:.0f}")
# joblib.dump(results, MODEL_DIR / 'viirs_all_clusters_2_to_12.pkl')

loaded_models = joblib.load('viirs_all_clusters_2_to_12.pkl')
print(f"Loaded K-Means models for k = {list(loaded_models.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTER MAP VISUALIZATION  (loop over selected k values)
# ══════════════════════════════════════════════════════════════════════════════
for k_val in [6, 8, 10]:
    model_data = loaded_models[k_val]
    k = model_data['model'].n_clusters
    cluster_map = model_data['labels'].reshape(HEIGHT, WIDTH)

    n_clusters  = k
    cmap_base   = plt.get_cmap('tab20')
    colors      = [cmap_base(i) for i in range(n_clusters)]
    custom_cmap = mcolors.ListedColormap(colors)
    norm        = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5,
                                       n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(cluster_map, cmap=custom_cmap, norm=norm,
                   interpolation='nearest', origin='upper')
    ax.set_title(f'VIIRS Cluster Classification (k = {k})',
                 fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_clusters),
                        fraction=0.046, pad=0.04)
    cbar.set_label('Cluster ID')
    cbar.ax.set_yticklabels([f'Cluster {i+1}' for i in range(n_clusters)])

    add_geo_ticks(ax, lon, lat)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'cluster_map_kmeans{k}.png',
                bbox_inches='tight')
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 4. PRINCIPAL COMPONENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance  = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Scree plot
plot_pca_scree(explained_variance, cumulative_variance,
               save_path=PLOT_DIR / 'pca_scree.png')

# Eigenvalue / eigenvector tables
band_names = ALL_BANDS
df_eigen, df_load = pca_eigen_table(pca, feature_names=band_names)
df_eigen_top, df_load_top = pca_eigen_table(
    pca, feature_names=band_names, top_n=5,
    latex_path=str(PLOT_DIR / 'pca_tables.tex')
)

# Discrete PC maps
for pc_idx, title in [(0, 'PC1: Albedo/Brightness'),
                       (1, 'PC2: Thermal Contrast'),
                       (2, 'PC3'), (3, 'PC4')]:
    plot_pca_discrete(X_pca[:, pc_idx], HEIGHT, WIDTH,
                      n_levels=4, title=title)

# PCA RGB composite
rgbstack = plot_pca_rgb(X_pca, HEIGHT, WIDTH,
                        pc_indices=(1, 2, 0), stretch=2)

# PCA loadings bar chart
plot_pca_loadings(pca, band_names)

# ══════════════════════════════════════════════════════════════════════════════
# 5. CLUSTER MERGING  (k=8 example)
# ══════════════════════════════════════════════════════════════════════════════
model_data  = loaded_models[8]
k           = model_data['model'].n_clusters
cluster_map = model_data['labels'].reshape(HEIGHT, WIDTH).copy()

merge_dict = {
    7: 2,   # cluster 8 → cluster 3  (0-indexed)
    3: 1,   # cluster 4 → cluster 2  (0-indexed)
}

cluster_map, label_map = merge_and_relabel(cluster_map, merge_dict)
n_clusters_new = len(np.unique(cluster_map))

# Color list for merged clusters
cc = ['#4dac26', '#a6cee3', '#d6a86b', '#2166ac', '#6a3d9a', '#f0f0f0']
custom_cmap = mcolors.ListedColormap(cc[:n_clusters_new])
norm = mcolors.BoundaryNorm(np.arange(n_clusters_new + 1) - 0.5,
                             n_clusters_new)

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
im = ax.imshow(cluster_map, cmap=custom_cmap, norm=norm,
               interpolation='nearest', origin='upper')
ax.set_title(f'VIIRS Merged Clusters (k={k} → {n_clusters_new})',
             fontsize=12, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_clusters_new),
                    fraction=0.046, pad=0.04)
cbar.set_label('Cluster ID')
cbar.ax.set_yticklabels([f'Cluster {i+1}' for i in range(n_clusters_new)])
add_geo_ticks(ax, lon, lat)
fig.tight_layout()
fig.savefig(PLOT_DIR / f'cluster_map_kmeans{k}_merged.png',
            bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 6. K-MEANS SPECTRAL RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
plot_kmeans_spectral_response(
    X_raw,
    cluster_map=cluster_map,
    k=8,
    all_band_names=ALL_BANDS,
    reflective_bands=REFLECTIVE_BANDS,
    emissive_bands=EMISSIVE_BANDS,
    save_dir=PLOT_DIR,
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
save_checkpoint({
    'pca':            pca,
    'X_pca':          X_pca,
    'rgbstack':       rgbstack,
    'cluster_map':    cluster_map,
    'loaded_models':  loaded_models,
}, 'checkpoint_02.pkl')

print("\n[02] Done — K-Means and PCA complete.")