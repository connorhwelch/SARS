"""
01_data_load_and_explore.py
===========================
Load VIIRS NetCDF, subset to study area, visualize true color
and zenith angles, build the band-stacked feature arrays.

Outputs
-------
checkpoint_01.pkl :
    ds_path, X_raw, X_scaled, nan_mask, rgb, lon, lat
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from satpy.writers import get_enhanced_image
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, PLOT_DIR, MODEL_DIR, HEIGHT, WIDTH,
    Y_SLICE, X_SLICE, ALL_BANDS,
    apply_plot_style, add_geo_ticks, save_checkpoint,
)
from functions_project2 import plot_zenith_angles, plot_rgb_subset

apply_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SUBSET
# ══════════════════════════════════════════════════════════════════════════════
ds_path = sorted(DATA_DIR.rglob('*VIIRS_corrected*2024*.nc*'))[0]
ds = xr.open_dataset(ds_path)

# Crop to study area and flip both axes
ds = ds.isel(y=Y_SLICE, x=X_SLICE)
ds = ds.isel(y=slice(None, None, -1), x=slice(None, None, -1))

print(ds)
print(f"Image dimensions: {HEIGHT} × {WIDTH}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRUE COLOR VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
img = get_enhanced_image(ds['true_color'])
rgb = img.data.transpose('y', 'x', 'bands').values
rgb = np.clip(rgb, 0, 1)

lon = ds['longitude'].values
lat = ds['latitude'].values

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb, origin='upper')
add_geo_ticks(ax, lon, lat)
ax.set_title("VIIRS True Color", fontsize=13, fontweight='bold', pad=10)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'true_color.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 3. ZENITH ANGLE GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
plot_zenith_angles(ds, save_path=PLOT_DIR / 'zenith_angles.png')

# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD FEATURE ARRAYS
# ══════════════════════════════════════════════════════════════════════════════
# Stack all 16 bands → (height, width, n_bands)
X = np.stack([ds[b].values for b in ALL_BANDS], axis=-1)
ny, nx, nb = X.shape
X = X.reshape(-1, nb)                 # (n_pixels, 16)

# NaN mask → valid pixels only
nan_mask = np.any(np.isnan(X), axis=1)
X_valid  = X[~nan_mask]

print(f"Total pixels : {len(X):,}")
print(f"Valid pixels : {len(X_valid):,}")
print(f"NaN pixels   : {nan_mask.sum():,}")

# Raw (unscaled) feature matrix — same pixel order
band_arrays = [ds[b].values.ravel() for b in ALL_BANDS]
X_raw = np.stack(band_arrays, axis=1)

# Standardize
X_scaled = StandardScaler().fit_transform(X_valid)

print(f"X_raw shape    : {X_raw.shape}")
print(f"X_scaled shape : {X_scaled.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
save_checkpoint({
    'ds_path':  str(ds_path),
    'X_raw':    X_raw,
    'X_scaled': X_scaled,
    'nan_mask': nan_mask,
    'rgb':      rgb,
    'lon':      lon,
    'lat':      lat,
}, 'checkpoint_01.pkl')

print("\n[01] Done — data loaded and feature arrays built.")