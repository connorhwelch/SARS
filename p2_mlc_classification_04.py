"""
04_mlc_classification.py
========================
Train a Maximum Likelihood Classifier (GaussianNB), predict on
test set and full image, generate confusion matrices, accuracy
metrics, spectral responses, and the classification map.

Inputs
------
checkpoint_01.pkl, checkpoint_03.pkl

Outputs
-------
checkpoint_04.pkl :
    mlc, y_pred, class_map, y_pred_full
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from satpy.writers import get_enhanced_image

from p2_config import *
from functions_project2 import *

apply_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
ckpt01 = load_checkpoint('checkpoint_01.pkl')
ckpt03 = load_checkpoint('checkpoint_03.pkl')

X_scaled    = ckpt01['X_scaled']
X_raw       = ckpt01['X_raw']
rgb         = ckpt01['rgb']
lon         = ckpt01['lon']
lat         = ckpt01['lat']

X_train     = ckpt03['X_train']
X_test      = ckpt03['X_test']
X_train_raw = ckpt03['X_train_raw']
X_test_raw  = ckpt03['X_test_raw']
y_train     = ckpt03['y_train']
y_test      = ckpt03['y_test']

ds = xr.open_dataset(ckpt01['ds_path'])
ds = ds.isel(y=slice(2935-256, 2935+256), x=slice(1448-320, 1448+320))
ds = ds.isel(y=slice(None, None, -1), x=slice(None, None, -1))

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN MLC  (GaussianNB)
# ══════════════════════════════════════════════════════════════════════════════
mlc = GaussianNB()
mlc.fit(X_train, y_train)
print("[MLC] Model trained.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
y_pred = mlc.predict(X_test)

print("\n" + "=" * 65)
print("  CLASSIFICATION REPORT")
print("=" * 65)
print(classification_report(
    y_test, y_pred,
    target_names=list(CLASS_LABELS.keys())
))

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
plot_confusion_matrix(
    y_test, y_pred, CLASS_LABELS,
    save_path=PLOT_DIR / 'confusion_matrix_mlc.png',
)

plot_per_class_confusion(
    y_test, y_pred, CLASS_LABELS,
    save_path=PLOT_DIR / 'per_class_confusion_mlc.png',
)
plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# 5. ACCURACY METRICS  (+ LaTeX export)
# ══════════════════════════════════════════════════════════════════════════════
df_acc = calculate_accuracy_metrics(
    y_test, y_pred, CLASS_LABELS,
    latex_path=str(TABLE_DIR / 'accuracy_mlc.tex'),
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. SPECTRAL STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
df_wide, df_long = build_spectral_statistics_table(
    X_raw=X_train_raw,
    y=y_train,
    all_band_names=ALL_BANDS,
    class_labels=CLASS_LABELS,
    save_dir=TABLE_DIR,
)
print("\nSpectral statistics (wide):")
print(df_wide.head())

# ══════════════════════════════════════════════════════════════════════════════
# 7. SPECTRAL RESPONSE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
# All classes overlaid
plot_spectral_response(
    X_raw=X_train_raw,
    y=y_train,
    all_band_names=ALL_BANDS,
    reflective_bands=REFLECTIVE_BANDS,
    emissive_bands=EMISSIVE_BANDS,
    class_labels=CLASS_LABELS,
    mode='all_classes',
    title_prefix='VIIRS Spectral Response (Training)',
    save_dir=PLOT_DIR,
)

# Per-class individual traces
plot_spectral_response(
    X_raw=X_train_raw,
    y=y_train,
    all_band_names=ALL_BANDS,
    reflective_bands=REFLECTIVE_BANDS,
    emissive_bands=EMISSIVE_BANDS,
    class_labels=CLASS_LABELS,
    mode='per_class',
    show_individual=True,
    title_prefix='VIIRS Spectral Response (Training)',
    save_dir=PLOT_DIR,
    ylim_reflective=(0, 100),
    ylim_emissive=(260, 340),
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FULL IMAGE CLASSIFICATION MAP
# ══════════════════════════════════════════════════════════════════════════════
y_pred_full = mlc.predict(X_scaled)
class_map   = y_pred_full.reshape(HEIGHT, WIDTH)

# ── Setup class colormap ──────────────────────────────────────────────────────
n_classes   = len(CLASS_LABELS)
custom_cmap = mcolors.ListedColormap(CLASS_COLORS_LIST)
norm        = mcolors.BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)

# ── True Color RGB ────────────────────────────────────────────────────────────
img_full = get_enhanced_image(ds['true_color'])
rgb_full = img_full.data.transpose('y', 'x', 'bands').values
rgb_full = np.clip(rgb_full, 0, 1)

# ── Side-by-side plot ─────────────────────────────────────────────────────────
plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

# Left: True Color
axes[0].imshow(rgb_full, origin='upper')
add_geo_ticks(axes[0], lon, lat)
axes[0].set_title('VIIRS True Color', fontsize=13, fontweight='bold')

# Right: Classified Map
im = axes[1].imshow(class_map, cmap=custom_cmap, norm=norm,
                    interpolation='nearest', origin='upper')
add_geo_ticks(axes[1], lon, lat)
axes[1].set_title('Maximum Likelihood Classification',
                  fontsize=13, fontweight='bold')

# Discrete colorbar
tick_locs = np.arange(n_classes)
cbar = fig.colorbar(im, ax=axes[1], ticks=tick_locs,
                    fraction=0.046, pad=0.04)
cbar.set_label('Land Cover Class', fontsize=10)
cbar.ax.set_yticklabels(CLASS_DISPLAY_NAMES, fontsize=9)

fig.suptitle('VIIRS Surface Classification — Maximum Likelihood',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'mlc_classification_map.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
save_checkpoint({
    'mlc':         mlc,
    'y_pred':      y_pred,
    'y_pred_full': y_pred_full,
    'class_map':   class_map,
}, 'checkpoint_04.pkl')

print("\n[04] Done — MLC trained, evaluated, and mapped.")

df_pixel_info, df_spectral_mean, df_spectral_std = build_class_summary_tables(
    X_raw            = X_train_raw,
    y                = y_train,
    ds               = ds,
    pixel_areas      = pixel_areas,
    pixel_cross      = pixel_cross,
    pixel_along      = pixel_along,
    class_labels     = CLASS_LABELS,
    all_band_names   = ALL_BANDS,
    reflective_bands = REFLECTIVE_BANDS,
    emissive_bands   = EMISSIVE_BANDS,
    satellite_name   = "NOAA-20 VIIRS",
    save_dir         = TABLE_DIR,
    latex_path       = TABLE_DIR / 'class_summary_tables.tex',
)