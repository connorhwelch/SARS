"""
03_training_samples.py
======================
Build training/testing samples using PCA-RGB masking, spectral
indices (NDWI, NDSI, NDVI), and manual selection for each class.

Inputs
------
checkpoint_01.pkl, checkpoint_02.pkl

Outputs
-------
checkpoint_03.pkl :
    X_train, X_test, X_train_raw, X_test_raw,
    y_train, y_test, a_train, a_test, pixel_areas,
    class_index_map, water_mask, cloud_mask, snow_mask, smoke_mask
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from satpy.writers import get_enhanced_image

from p2_config import *
from functions_project2 import *

apply_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
ckpt01 = load_checkpoint('checkpoint_01.pkl')
ckpt02 = load_checkpoint('checkpoint_02.pkl')

X_raw    = ckpt01['X_raw']
X_scaled = ckpt01['X_scaled']
rgb      = ckpt01['rgb']

rgbstack = ckpt02['rgbstack']      # PCA RGB composite

ds = xr.open_dataset(ckpt01['ds_path'])
ds = ds.isel(y=slice(2935-256, 2935+256), x=slice(1448-320, 1448+320))
ds = ds.isel(y=slice(None, None, -1), x=slice(None, None, -1))

# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE PIXEL AREA GRID
# ══════════════════════════════════════════════════════════════════════════════
pixel_areas, pixel_cross, pixel_along = compute_pixel_area_grid(
    sat_data=ds,
    nadir_along_track_resolution=NADIR_ALONG_TRACK,
    nadir_cross_track_resolution=NADIR_CROSS_TRACK,
    sat_orb_height=ORBITAL_HEIGHT,
    correct_for_earth_curvature=True,
)

print(f"Pixel area grid shape : {pixel_areas.shape}")
print(f"Nadir pixel area      : {NADIR_ALONG_TRACK * NADIR_CROSS_TRACK / 1e6:.4f} km²")
print(f"Scene min pixel area  : {pixel_areas.min() / 1e6:.4f} km²")
print(f"Scene max pixel area  : {pixel_areas.max() / 1e6:.4f} km²")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASS MASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. WATER ─────────────────────────────────────────────────────────────────
R, G, B = rgbstack[:,:,0], rgbstack[:,:,1], rgbstack[:,:,2]

# Aggressive water mask
mask_blue_green = (R < 0.2) & (G < 0.5) & (B > 0.5)
water_mask = mask_blue_green.copy()

# Conservative water selection
mask_blue = (R < 0.2) & (G < 0.2) & (B > 0.5)
blue_only = np.zeros_like(rgbstack)
blue_only[mask_blue] = rgbstack[mask_blue]

# Verify with NDWI
ndwi = (ds['M04'] - ds['M07']) / (ds['M04'] + ds['M07'])
ndwi_np = ndwi.values

confident_water = blue_only.copy()
y_indx_water, x_indx_water, _ = np.where(confident_water > 0)
water_indx_lst = list(zip(y_indx_water, x_indx_water))
print(f"Water pixels: {len(water_indx_lst):,}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))
plot_rgb_subset(ds, ax=ax, overlay=blue_only, show=False,
                make_transparent_zero=True)
ax.set_title('Water Mask Overlay', fontsize=13, fontweight='bold')
fig.savefig(PLOT_DIR / 'mask_water.png', bbox_inches='tight')
plt.show()

# ── 3b. CLOUD ─────────────────────────────────────────────────────────────────
dsfrgb = viirs_day_snow_fog_rgb(ds['M07'], ds['M10'], ds['M13'] - ds['M15'])
R_sf, G_sf, B_sf = dsfrgb[:,:,0], dsfrgb[:,:,1], dsfrgb[:,:,2]

mask_white = (R_sf > 0.9) & (G_sf > 0.9) & (B_sf > 0.9)
cloud_mask = mask_white.copy()

cloud = np.zeros_like(rgbstack)
cloud[mask_white] = dsfrgb[mask_white]

y_indx_cloud, x_indx_cloud, _ = np.where(cloud > 0)
cloud_indx_lst = list(zip(y_indx_cloud, x_indx_cloud))
print(f"Cloud pixels: {len(cloud_indx_lst):,}")

fig, ax = plt.subplots(figsize=(10, 8))
plot_rgb_subset(rgb=cloud, ax=ax, show=False, make_transparent_zero=True)
ax.set_title('Cloud Mask', fontsize=13, fontweight='bold')
fig.savefig(PLOT_DIR / 'mask_cloud.png', bbox_inches='tight')
plt.show()

# ── 3c. SNOW ──────────────────────────────────────────────────────────────────
ndsi = (ds['M04'] - ds['M10']) / (ds['M04'] + ds['M10'])
ndsi_np = ndsi.values

ndsi_threshold = 0.7
confident_snow = ndsi_np.copy()
confident_snow[ndsi_np < ndsi_threshold] = np.nan
snow_mask = confident_snow >= 0.01

# Remove water overlap
snow_water_removed = confident_snow.copy()
snow_water_removed[water_mask] = np.nan

y_indx_snow, x_indx_snow = np.where(snow_water_removed > 0)
snow_water_removed_indx = list(zip(y_indx_snow, x_indx_snow))
print(f"Snow pixels (water removed): {len(snow_water_removed_indx):,}")

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(snow_water_removed, cmap='gray')
ax.set_title('Snow Mask (water removed)', fontsize=13, fontweight='bold')
fig.savefig(PLOT_DIR / 'mask_snow.png', bbox_inches='tight')
plt.show()

# ── 3d. SMOKE ─────────────────────────────────────────────────────────────────
R, G, B = rgbstack[:,:,0], rgbstack[:,:,1], rgbstack[:,:,2]

mask_green = (R < 0.3) & (G > 0.7) & (B < 0.7)
green_only = np.zeros_like(rgbstack)
green_only[mask_green] = rgbstack[mask_green]
smoke_mask = mask_green.copy()

confident_smoke = green_only.copy()

# Smoke is only visible in a CROPPED region of the PCA-RGB
smoke_y_slice = slice(100, 400)
smoke_x_slice = slice(0, 100)

fig, ax = plt.subplots(figsize=(10, 8))
plot_rgb_subset(ds.sel(x=smoke_x_slice, y=smoke_y_slice), ax=ax,
                overlay=green_only[smoke_y_slice, smoke_x_slice],
                show=False, make_transparent_zero=True)
ax.set_title('Smoke Mask (cropped region)', fontsize=13, fontweight='bold')
fig.savefig(PLOT_DIR / 'mask_smoke.png', bbox_inches='tight')
plt.show()

# Extract indices from cropped region
y_indx_smoke, x_indx_smoke, _ = np.where(
    green_only[smoke_y_slice, smoke_x_slice] >= 0.5
)
smoke_indx_lst = list(zip(y_indx_smoke, x_indx_smoke))

# ── OFFSET FIX: convert local crop indices → full-image coordinates ───────
smoke_y_offset = 100   # = smoke_y_slice.start
smoke_x_offset = 0     # = smoke_x_slice.start

smoke_indx_lst_global = [
    (y + smoke_y_offset, x + smoke_x_offset)
    for y, x in smoke_indx_lst
]
print(f"Smoke pixels (global): {len(smoke_indx_lst_global):,}")

# ── 3e. BARE SOIL ────────────────────────────────────────────────────────────
ndvi = (ds['M07'] - ds['M05']) / (ds['M07'] + ds['M05'])
ndvi_np = ndvi.values

vegetation_mask = ndvi_np > 0.15
confident_bare_soil = ndvi_np.copy()
confident_bare_soil[vegetation_mask] = np.nan
confident_bare_soil[smoke_mask]      = np.nan
confident_bare_soil[cloud_mask]      = np.nan
confident_bare_soil[water_mask]      = np.nan
confident_bare_soil[snow_mask]       = np.nan

# Visualization
img = get_enhanced_image(ds['true_color'])
rgb_tc = img.data.transpose('y', 'x', 'bands').values
rgb_tc = np.clip(rgb_tc, 0, 1)

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(rgb_tc)
ax.imshow(confident_bare_soil, cmap='Grays', alpha=0.7)
ax.set_title('Bare Soil Mask', fontsize=13, fontweight='bold')
fig.savefig(PLOT_DIR / 'mask_bare_soil.png', bbox_inches='tight')
plt.show()

y_indx_bare, x_indx_bare = np.where(confident_bare_soil >= 0)
bare_indx_lst = list(zip(y_indx_bare, x_indx_bare))
print(f"Bare Soil pixels: {len(bare_indx_lst):,}")

# ── 3f. VEGETATION ───────────────────────────────────────────────────────────
ndvi = (ds['M07'] - ds['M05']) / (ds['M07'] + ds['M05'])
ndvi_np = ndvi.values

non_vegetation_mask = ndvi_np < 0.4
confident_vegetation = ndvi_np.copy()
confident_vegetation[non_vegetation_mask] = np.nan
confident_vegetation[smoke_mask]          = np.nan
confident_vegetation[cloud_mask]          = np.nan
confident_vegetation[water_mask]          = np.nan

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(rgb_tc)
ax.imshow(confident_vegetation, cmap='Greens', alpha=0.7)
ax.set_title('Vegetation Mask', fontsize=13, fontweight='bold')
fig.savefig(PLOT_DIR / 'mask_vegetation.png', bbox_inches='tight')
plt.show()

y_indx_veg, x_indx_veg = np.where(confident_vegetation >= 0)
veg_indx_lst = list(zip(y_indx_veg, x_indx_veg))
print(f"Vegetation pixels: {len(veg_indx_lst):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD CLASS INDEX MAP
# ══════════════════════════════════════════════════════════════════════════════
class_index_map = [
    (water_indx_lst,          CLASS_LABELS['water'],      'water'),
    (cloud_indx_lst,          CLASS_LABELS['cloud'],      'cloud'),
    (snow_water_removed_indx, CLASS_LABELS['snow'],       'snow'),
    (smoke_indx_lst_global,   CLASS_LABELS['smoke'],      'smoke'),
    (bare_indx_lst,           CLASS_LABELS['bare_soil'],  'bare_soil'),
    (veg_indx_lst,            CLASS_LABELS['vegetation'], 'vegetation'),
]

print("\n" + "=" * 55)
print(f"{'Class':<14} {'Pixels':>10}")
print("-" * 55)
for indx_lst, label, name in class_index_map:
    print(f"  {name:<12} {len(indx_lst):>10,}")
print("=" * 55)

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT  (per class, with pixel areas)
# ══════════════════════════════════════════════════════════════════════════════
X_train_list,     X_test_list     = [], []
X_train_raw_list, X_test_raw_list = [], []
y_train_list,     y_test_list     = [], []
a_train_list,     a_test_list     = [], []

print(f"\n{'Class':<14} {'Total':>8} {'Train':>8} {'Test':>7}")
print("-" * 42)

for indx_lst, label, name in class_index_map:
    if len(indx_lst) == 0:
        print(f"  WARNING: No indices for '{name}', skipping.")
        continue

    # Convert (y, x) → flat indices
    ys = np.array([idx[0] for idx in indx_lst])
    xs = np.array([idx[1] for idx in indx_lst])
    flat_indices = ys * WIDTH + xs

    # Guard against out-of-bounds
    valid_mask   = (flat_indices >= 0) & (flat_indices < len(X_scaled))
    flat_indices = flat_indices[valid_mask]
    ys = ys[valid_mask]
    xs = xs[valid_mask]

    # Extract features + labels + pixel areas
    X_class     = X_scaled[flat_indices]
    X_class_raw = X_raw[flat_indices]
    y_class     = np.full(len(flat_indices), label, dtype=int)
    a_class     = pixel_areas[ys, xs]

    # Stratified split — same seed for all arrays
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_class, y_class,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    X_tr_raw, X_te_raw, _, _ = train_test_split(
        X_class_raw, y_class,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    a_tr, a_te, _, _ = train_test_split(
        a_class, y_class,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    X_train_list.append(X_tr);         X_test_list.append(X_te)
    X_train_raw_list.append(X_tr_raw); X_test_raw_list.append(X_te_raw)
    y_train_list.append(y_tr);         y_test_list.append(y_te)
    a_train_list.append(a_tr);         a_test_list.append(a_te)

    print(f"  {name:<12} {len(flat_indices):>8,} {len(y_tr):>8,} {len(y_te):>7,}")

# Concatenate all classes
X_train     = np.vstack(X_train_list)
X_test      = np.vstack(X_test_list)
X_train_raw = np.vstack(X_train_raw_list)
X_test_raw  = np.vstack(X_test_raw_list)
y_train     = np.hstack(y_train_list)
y_test      = np.hstack(y_test_list)
a_train     = np.hstack(a_train_list)
a_test      = np.hstack(a_test_list)

print("-" * 42)
print(f"  {'TOTAL':<12} {len(y_train)+len(y_test):>8,} "
      f"{len(y_train):>8,} {len(y_test):>7,}")
print(f"\nX_train: {X_train.shape},  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},   y_test:  {y_test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
save_checkpoint({
    'X_train':     X_train,
    'X_test':      X_test,
    'X_train_raw': X_train_raw,
    'X_test_raw':  X_test_raw,
    'y_train':     y_train,
    'y_test':      y_test,
    'a_train':     a_train,
    'a_test':      a_test,
    'pixel_areas': pixel_areas,
    'class_index_map': class_index_map,
    'water_mask':  water_mask,
    'cloud_mask':  cloud_mask,
    'snow_mask':   snow_mask,
    'smoke_mask':  smoke_mask,
}, 'checkpoint_03.pkl')

print("\n[03] Done — training samples built.")