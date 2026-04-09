"""
05_chi2_threshold.py
====================
Chi-squared threshold classification: reject pixels whose
Mahalanobis distance exceeds the chi-squared critical value.
Compare standard MLC vs chi-squared at multiple confidence levels.

Inputs
------
checkpoint_01.pkl, checkpoint_03.pkl, checkpoint_04.pkl

Outputs
-------
checkpoint_05.pkl :
    class_map_chi2, y_pred_chi2, chi2_thresh, n_unc, min_dist
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, PLOT_DIR, TABLE_DIR, MODEL_DIR, HEIGHT, WIDTH,
    ALL_BANDS, REFLECTIVE_BANDS, EMISSIVE_BANDS,
    CLASS_LABELS, CLASS_LABELS_EXT, CLASS_DISPLAY_NAMES,
    CHI2_CONFIDENCE,
    apply_plot_style, save_checkpoint, load_checkpoint,
)
from functions_project2 import (
    chi2_classify, plot_chi2_classification,
    chi2_accuracy_report, calculate_accuracy_metrics,
    plot_spectral_response,
)

apply_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
ckpt01 = load_checkpoint('checkpoint_01.pkl')
ckpt03 = load_checkpoint('checkpoint_03.pkl')
ckpt04 = load_checkpoint('checkpoint_04.pkl')

X_scaled    = ckpt01['X_scaled']
X_raw       = ckpt01['X_raw']

X_train     = ckpt03['X_train']
X_test      = ckpt03['X_test']
X_train_raw = ckpt03['X_train_raw']
X_test_raw  = ckpt03['X_test_raw']
y_train     = ckpt03['y_train']
y_test      = ckpt03['y_test']

mlc         = ckpt04['mlc']
y_pred_std  = ckpt04['y_pred']       # standard MLC predictions (no rejection)
class_map   = ckpt04['class_map']

ds = xr.open_dataset(ckpt01['ds_path'])
ds = ds.isel(y=slice(2935-256, 2935+256), x=slice(1448-320, 1448+320))
ds = ds.isel(y=slice(None, None, -1), x=slice(None, None, -1))

n_classes = len(CLASS_LABELS)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CHI-SQUARED CLASSIFICATION  (primary confidence)
# ══════════════════════════════════════════════════════════════════════════════
class_labels_display = {
    "Water": 0, "Cloud": 1, "Snow": 2, "Smoke": 3,
    "Bare Soil": 4, "Vegetation": 5,
}

class_map_chi2, chi2_thresh, min_dist, n_unc = chi2_classify(
    X_scaled, mlc, class_labels_display,
    height=HEIGHT, width=WIDTH,
    confidence=CHI2_CONFIDENCE,
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT CHI-SQUARED CLASSIFICATION MAP
# ══════════════════════════════════════════════════════════════════════════════
plot_chi2_classification(
    ds, class_map_chi2, class_labels_display,
    chi2_threshold=chi2_thresh,
    confidence=CHI2_CONFIDENCE,
    n_unclassified=n_unc,
    save_path=str(PLOT_DIR / f'mlc_chi2_{CHI2_CONFIDENCE:.0%}.png'),
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. CHI-SQUARED ACCURACY ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
# Predict on test set with chi-squared threshold
# Re-use the trained class statistics from chi2_classify internals:
# We need per-pixel Mahalanobis distances for test set only.

from scipy.stats import chi2 as chi2_dist

n_bands    = X_test.shape[1]
threshold  = chi2_dist.ppf(CHI2_CONFIDENCE, df=n_bands)

# Compute class means and covariance inverses from training data
class_means   = []
class_cov_inv = []
for k in range(n_classes):
    X_k    = X_train[y_train == k]
    mean_k = np.mean(X_k, axis=0)
    cov_k  = np.cov(X_k, rowvar=False) + np.eye(n_bands) * 1e-6
    class_means.append(mean_k)
    class_cov_inv.append(np.linalg.inv(cov_k))

# Mahalanobis distance for each test pixel to each class
distances_test = np.zeros((len(X_test), n_classes))
for k_idx in range(n_classes):
    diff = X_test - class_means[k_idx]
    left = diff @ class_cov_inv[k_idx]
    distances_test[:, k_idx] = np.sum(left * diff, axis=1)

min_dist_test   = np.min(distances_test, axis=1)
best_class_test = np.argmin(distances_test, axis=1)

# Apply threshold
y_pred_chi2 = best_class_test.copy()
y_pred_chi2[min_dist_test > threshold] = n_classes   # Unclassified

print(f"\nTest set chi² rejection:")
print(f"  Threshold (χ² @ {CHI2_CONFIDENCE:.0%}): {threshold:.4f}")
print(f"  Classified:   {np.sum(y_pred_chi2 < n_classes):,}")
print(f"  Unclassified: {np.sum(y_pred_chi2 == n_classes):,} "
      f"({np.mean(y_pred_chi2 == n_classes):.2%})")

# Full accuracy report with rejection
df_chi2_report, cm_chi2 = chi2_accuracy_report(
    y_test, y_pred_chi2, CLASS_LABELS,
    n_classes=n_classes,
    confidence=CHI2_CONFIDENCE,
    chi2_threshold=threshold,
    latex_path=str(TABLE_DIR / f'accuracy_chi2_{CHI2_CONFIDENCE:.0%}.tex'),
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTION A: ACCURACY ON CLASSIFIED PIXELS ONLY
# ══════════════════════════════════════════════════════════════════════════════
classified_mask      = y_pred_chi2 < n_classes
y_test_classified    = y_test[classified_mask]
y_pred_classified    = y_pred_chi2[classified_mask]

df_chi2_classified, _ = calculate_accuracy_metrics(
    y_test_classified, y_pred_classified, CLASS_LABELS,
    latex_path=str(TABLE_DIR / 'accuracy_chi2_classified_only.tex'),
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. STANDARD VS CHI-SQUARED COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
df_standard, _ = calculate_accuracy_metrics(
    y_test, y_pred_std, CLASS_LABELS,
)

print("\n" + "=" * 70)
print("  COMPARISON: Standard MLC vs Chi-squared MLC")
print("=" * 70)
print(f"\n  Standard MLC:")
overall_std = df_standard[df_standard['Class'] == '** OVERALL **']
print(f"    Overall Accuracy: "
      f"{overall_std[\"Producer's Acc (%)\"].values[0]:.2f}%")

print(f"\n  Chi-squared MLC ({CHI2_CONFIDENCE:.0%}):")
print(f"    Overall (all pixels):       "
      f"{df_chi2_report.iloc[-1]['Producer\\'s Acc (%)']:.2f}%" if 'Producer\'s Acc (%)' in df_chi2_report.columns else "    See report above")
print(f"    Overall (classified only):  "
      f"{df_chi2_report.iloc[-1]['User\\'s Acc (%)']:.2f}%" if 'User\'s Acc (%)' in df_chi2_report.columns else "    See report above")

# ══════════════════════════════════════════════════════════════════════════════
# 7. MULTI-CONFIDENCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
confidence_levels = [0.90, 0.95, 0.99, 0.999]

print(f"\n{'Confidence':>12} {'χ² Thresh':>12} {'Rejected':>10} {'Rej %':>8}")
print("-" * 48)

for conf in confidence_levels:
    thresh_c = chi2_dist.ppf(conf, df=n_bands)
    n_reject = np.sum(min_dist_test > thresh_c)
    pct      = n_reject / len(min_dist_test) * 100
    print(f"  {conf:>10.1%} {thresh_c:>12.4f} {n_reject:>10,} {pct:>7.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SPECTRAL RESPONSE — CHI-SQUARED PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
# Only plot classified (non-rejected) test pixels
classified_idx = y_pred_chi2 < n_classes

plot_spectral_response(
    X_raw=X_test_raw[classified_idx],
    y=y_pred_chi2[classified_idx],
    all_band_names=ALL_BANDS,
    reflective_bands=REFLECTIVE_BANDS,
    emissive_bands=EMISSIVE_BANDS,
    class_labels=CLASS_LABELS,
    mode='all_classes',
    title_prefix=f'VIIRS Spectral Response — χ² Classified ({CHI2_CONFIDENCE:.0%})',
    save_dir=PLOT_DIR,
)

plot_spectral_response(
    X_raw=X_test_raw[classified_idx],
    y=y_pred_chi2[classified_idx],
    all_band_names=ALL_BANDS,
    reflective_bands=REFLECTIVE_BANDS,
    emissive_bands=EMISSIVE_BANDS,
    class_labels=CLASS_LABELS,
    mode='per_class',
    show_individual=True,
    title_prefix=f'VIIRS Spectral Response — χ² Classified ({CHI2_CONFIDENCE:.0%})',
    save_dir=PLOT_DIR,
    ylim_reflective=(0, 100),
    ylim_emissive=(260, 340),
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. FULL IMAGE CHI-SQUARED AT MULTIPLE CONFIDENCE LEVELS
# ══════════════════════════════════════════════════════════════════════════════
for conf in confidence_levels:
    cmap_chi2, thresh_c, mdist_c, n_unc_c = chi2_classify(
        X_scaled, mlc, class_labels_display,
        height=HEIGHT, width=WIDTH,
        confidence=conf,
    )

    plot_chi2_classification(
        ds, cmap_chi2, class_labels_display,
        chi2_threshold=thresh_c,
        confidence=conf,
        n_unclassified=n_unc_c,
        save_path=str(PLOT_DIR / f'mlc_chi2_{conf:.0%}.png'),
    )
    plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
save_checkpoint({
    'class_map_chi2':  class_map_chi2,
    'y_pred_chi2':     y_pred_chi2,
    'chi2_thresh':     chi2_thresh,
    'n_unclassified':  n_unc,
    'min_dist':        min_dist,
    'threshold_test':  threshold,
    'class_means':     class_means,
    'class_cov_inv':   class_cov_inv,
}, 'checkpoint_05.pkl')

print("\n[05] Done — chi-squared threshold classification complete.")