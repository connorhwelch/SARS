"""
06_accuracy_and_area.py
=======================
Final accuracy comparisons, pixel-area-based class area summaries,
spectral statistics tables (mean/std) for training, testing, K-Means,
MLC predictions, and chi-squared classification.

Inputs
------
checkpoint_01.pkl, checkpoint_02.pkl, checkpoint_03.pkl,
checkpoint_04.pkl, checkpoint_05.pkl

Outputs
-------
Area summary tables     → TABLE_DIR / area_*.tex
Spectral stats tables   → TABLE_DIR / spectral_*.tex  +  .csv
Comparison figures      → PLOT_DIR  / area_comparison_*.png
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from p2_config import (
    DATA_DIR, PLOT_DIR, TABLE_DIR, MODEL_DIR, HEIGHT, WIDTH,
    ALL_BANDS, REFLECTIVE_BANDS, EMISSIVE_BANDS, BAND_WAVELENGTHS,
    CLASS_LABELS, CLASS_LABELS_EXT, CLASS_DISPLAY_NAMES,
    CLASS_COLORS_LIST, CLASS_COLORS_EXT,
    NADIR_ALONG_TRACK, NADIR_CROSS_TRACK, ORBITAL_HEIGHT,
    CHI2_CONFIDENCE,
    apply_plot_style, save_checkpoint, load_checkpoint,
)
from functions_project2 import (
    summarize_class_areas, calculate_accuracy_metrics,
    chi2_accuracy_report, extract_spectral_stats,
)

apply_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════
ckpt01 = load_checkpoint('checkpoint_01.pkl')
ckpt02 = load_checkpoint('checkpoint_02.pkl')
ckpt03 = load_checkpoint('checkpoint_03.pkl')
ckpt04 = load_checkpoint('checkpoint_04.pkl')
ckpt05 = load_checkpoint('checkpoint_05.pkl')

X_raw        = ckpt01['X_raw']
X_scaled     = ckpt01['X_scaled']

cluster_map  = ckpt02['cluster_map']

X_train_raw  = ckpt03['X_train_raw']
X_test_raw   = ckpt03['X_test_raw']
y_train      = ckpt03['y_train']
y_test       = ckpt03['y_test']
a_train      = ckpt03['a_train']
a_test       = ckpt03['a_test']
pixel_areas  = ckpt03['pixel_areas']

mlc          = ckpt04['mlc']
y_pred       = ckpt04['y_pred']
y_pred_full  = ckpt04['y_pred_full']
class_map    = ckpt04['class_map']

class_map_chi2 = ckpt05['class_map_chi2']
y_pred_chi2    = ckpt05['y_pred_chi2']

n_classes        = len(CLASS_LABELS)
pixel_areas_flat = pixel_areas.ravel()

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPECTRAL STATISTICS — ALL DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "#" * 80)
print("#  SPECTRAL STATISTICS TABLES")
print("#" * 80)

# ── 2a. Training data ────────────────────────────────────────────────────────
print("\n── Training Data ──")
df_mean_train, df_std_train = extract_spectral_stats(
    X_raw              = X_train_raw,
    y                  = y_train,
    all_band_names     = ALL_BANDS,
    class_labels       = CLASS_LABELS,
    band_wavelengths   = BAND_WAVELENGTHS,
    reflective_bands   = REFLECTIVE_BANDS,
    emissive_bands     = EMISSIVE_BANDS,
    save_dir           = TABLE_DIR,
    latex_path         = TABLE_DIR / 'spectral_train.tex',
    caption_prefix     = "Training Data",
    table_label_prefix = "train",
)

# ── 2b. Testing data ─────────────────────────────────────────────────────────
print("\n── Testing Data ──")
df_mean_test, df_std_test = extract_spectral_stats(
    X_raw              = X_test_raw,
    y                  = y_test,
    all_band_names     = ALL_BANDS,
    class_labels       = CLASS_LABELS,
    band_wavelengths   = BAND_WAVELENGTHS,
    reflective_bands   = REFLECTIVE_BANDS,
    emissive_bands     = EMISSIVE_BANDS,
    save_dir           = TABLE_DIR,
    latex_path         = TABLE_DIR / 'spectral_test.tex',
    caption_prefix     = "Testing Data",
    table_label_prefix = "test",
)

# ── 2c. MLC predictions on test set ──────────────────────────────────────────
print("\n── MLC Predictions (Test Set) ──")
df_mean_mlc, df_std_mlc = extract_spectral_stats(
    X_raw              = X_test_raw,
    y                  = y_pred,
    all_band_names     = ALL_BANDS,
    class_labels       = CLASS_LABELS,
    band_wavelengths   = BAND_WAVELENGTHS,
    reflective_bands   = REFLECTIVE_BANDS,
    emissive_bands     = EMISSIVE_BANDS,
    save_dir           = TABLE_DIR,
    latex_path         = TABLE_DIR / 'spectral_mlc_pred.tex',
    caption_prefix     = "MLC Predicted",
    table_label_prefix = "mlc_pred",
)

# ── 2d. Chi-squared classified pixels only ────────────────────────────────────
print("\n── Chi-squared Classified (Test Set) ──")
chi2_mask = y_pred_chi2 < n_classes

df_mean_chi2, df_std_chi2 = extract_spectral_stats(
    X_raw              = X_test_raw[chi2_mask],
    y                  = y_pred_chi2[chi2_mask],
    all_band_names     = ALL_BANDS,
    class_labels       = CLASS_LABELS,
    band_wavelengths   = BAND_WAVELENGTHS,
    reflective_bands   = REFLECTIVE_BANDS,
    emissive_bands     = EMISSIVE_BANDS,
    save_dir           = TABLE_DIR,
    latex_path         = TABLE_DIR / 'spectral_chi2.tex',
    caption_prefix     = f"Chi-squared ({CHI2_CONFIDENCE:.0%})",
    table_label_prefix = "chi2",
)

# ── 2e. K-Means clusters ─────────────────────────────────────────────────────
print("\n── K-Means Clusters ──")
cluster_y = cluster_map.ravel()
cluster_labels_dict = {
    f"Cluster {c+1}": int(c)
    for c in sorted(np.unique(cluster_y))
}

df_mean_km, df_std_km = extract_spectral_stats(
    X_raw              = X_raw,
    y                  = cluster_y,
    all_band_names     = ALL_BANDS,
    class_labels       = cluster_labels_dict,
    band_wavelengths   = BAND_WAVELENGTHS,
    reflective_bands   = REFLECTIVE_BANDS,
    emissive_bands     = EMISSIVE_BANDS,
    save_dir           = TABLE_DIR,
    latex_path         = TABLE_DIR / 'spectral_kmeans.tex',
    caption_prefix     = "K-Means",
    table_label_prefix = "kmeans",
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. AREA SUMMARIES — PER DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "#" * 80)
print("#  CLASS AREA SUMMARIES")
print("#" * 80)

# ── 3a. Training data ────────────────────────────────────────────────────────
df_area_train = summarize_class_areas(
    y_train, a_train, CLASS_LABELS,
    source_name="Training Data",
)

# ── 3b. Testing data ─────────────────────────────────────────────────────────
df_area_test = summarize_class_areas(
    y_test, a_test, CLASS_LABELS,
    source_name="Testing Data",
)

# ── 3c. MLC full scene ───────────────────────────────────────────────────────
df_area_mlc = summarize_class_areas(
    class_map.ravel(), pixel_areas_flat, CLASS_LABELS,
    source_name="MLC Full Scene",
)

# ── 3d. K-Means clusters ─────────────────────────────────────────────────────
df_area_km = summarize_class_areas(
    cluster_y, pixel_areas_flat, cluster_labels_dict,
    source_name="K-Means Clusters",
)

# ── 3e. Chi-squared classification ───────────────────────────────────────────
class_labels_chi2 = {**CLASS_LABELS, "Unclassified": n_classes}

df_area_chi2 = summarize_class_areas(
    class_map_chi2.ravel(), pixel_areas_flat, class_labels_chi2,
    source_name=f"Chi² Classification ({CHI2_CONFIDENCE:.0%})",
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMBINED AREA COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
all_dfs = []
for df, tag in [(df_area_train, 'Training'),
                (df_area_test,  'Testing'),
                (df_area_mlc,   'MLC'),
                (df_area_km,    'K-Means'),
                (df_area_chi2,  'Chi²')]:
    tmp = df.copy()
    # Rename the area columns so they don't collide on merge
    rename_map = {}
    for col in tmp.columns:
        if col != 'Class' and col != '** TOTAL **':
            rename_map[col] = f'{tag} {col}'
    tmp = tmp.rename(columns=rename_map)
    keep_cols = ['Class'] + [c for c in tmp.columns if tag in c]
    tmp = tmp[keep_cols]
    all_dfs.append(tmp)

df_compare = all_dfs[0]
for df_other in all_dfs[1:]:
    df_compare = df_compare.merge(df_other, on='Class', how='outer')
df_compare = df_compare.fillna(0)

print("\n" + "=" * 100)
print("  AREA COMPARISON — ALL SOURCES")
print("=" * 100)
print(df_compare.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 5. AREA COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
sources       = ['Training', 'Testing', 'MLC', 'Chi²']
source_colors = ['#1b9e77', '#d95f02', '#7570b3', '#66a61e']

# Find the area column for each source
area_cols = []
for s in sources:
    matches = [c for c in df_compare.columns if s in c and 'Total Area' in c]
    if matches:
        area_cols.append(matches[0])
    else:
        area_cols.append(None)

# Only plot real classes (exclude totals / unclassified)
skip_rows = ['** TOTAL **', 'Unclassified']
df_plot = df_compare[~df_compare['Class'].isin(skip_rows)].copy()
class_names_plot = df_plot['Class'].values

x     = np.arange(len(class_names_plot))
n_src = len(sources)
bar_w = 0.18

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

for i, (src, col, clr) in enumerate(zip(sources, area_cols, source_colors)):
    if col is not None and col in df_plot.columns:
        vals = df_plot[col].values
        ax.bar(x + i * bar_w, vals, width=bar_w, label=src,
               color=clr, edgecolor='white', linewidth=0.5)

ax.set_xticks(x + bar_w * (n_src - 1) / 2)
ax.set_xticklabels([n.replace('_', ' ').title() for n in class_names_plot],
                   fontsize=10)
ax.set_ylabel('Area (km²)', fontsize=11)
ax.set_title('Class Area Comparison Across Classification Methods',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

fig.tight_layout()
fig.savefig(PLOT_DIR / 'area_comparison_bar.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 6. PIXEL AREA STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
pa_km2 = pixel_areas / 1e6

print("\n" + "=" * 60)
print("  PIXEL AREA GEOMETRY STATISTICS")
print("=" * 60)
print(f"  Nadir pixel:   {NADIR_ALONG_TRACK * NADIR_CROSS_TRACK / 1e6:.6f} km²")
print(f"  Scene min:     {pa_km2.min():.6f} km²")
print(f"  Scene max:     {pa_km2.max():.6f} km²")
print(f"  Scene mean:    {pa_km2.mean():.6f} km²")
print(f"  Scene median:  {np.median(pa_km2):.6f} km²")
print(f"  Total scene:   {pa_km2.sum():.2f} km²")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL ACCURACY SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
df_acc_std = calculate_accuracy_metrics(y_test, y_pred, CLASS_LABELS)

# Handle both return signatures (some versions return df, some return df+cm)
if isinstance(df_acc_std, tuple):
    df_acc_std = df_acc_std[0]

n_total_test = len(y_test)
n_classified = int(np.sum(y_pred_chi2 < n_classes))
n_correct_std = int(np.sum(y_test == y_pred))
n_correct_chi2 = int(np.sum(
    y_test[chi2_mask] == y_pred_chi2[chi2_mask]
))

oa_std      = n_correct_std / n_total_test * 100
oa_chi2_all = n_correct_chi2 / n_total_test * 100
oa_chi2_cls = n_correct_chi2 / n_classified * 100 if n_classified > 0 else 0.0

summary_data = {
    'Method': [
        'Standard MLC',
        f'Chi² ({CHI2_CONFIDENCE:.0%}) — all pixels',
        f'Chi² ({CHI2_CONFIDENCE:.0%}) — classified only',
    ],
    'Overall Accuracy (%)': [oa_std, oa_chi2_all, oa_chi2_cls],
    'Pixels Evaluated':     [n_total_test, n_total_test, n_classified],
    'Pixels Rejected':      [0, n_total_test - n_classified, 0],
    'Rejection Rate (%)':   [
        0.0,
        (n_total_test - n_classified) / n_total_test * 100,
        0.0,
    ],
}
df_summary = pd.DataFrame(summary_data)

print("\n" + "=" * 70)
print("  FINAL ACCURACY COMPARISON")
print("=" * 70)
print(df_summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 8. PER-CLASS ACCURACY COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
# Get per-class Producer's and User's Accuracy for both methods
class_names = list(CLASS_LABELS.keys())

# Standard MLC
pa_std_list, ua_std_list = [], []
for cls_name, cls_val in CLASS_LABELS.items():
    true_mask = (y_test == cls_val)
    pred_mask = (y_pred == cls_val)
    tp = np.sum(true_mask & pred_mask)
    pa_std_list.append(tp / true_mask.sum() * 100 if true_mask.sum() > 0 else 0)
    ua_std_list.append(tp / pred_mask.sum() * 100 if pred_mask.sum() > 0 else 0)

# Chi-squared (classified only)
pa_chi2_list, ua_chi2_list = [], []
y_test_cls  = y_test[chi2_mask]
y_pred_cls  = y_pred_chi2[chi2_mask]
for cls_name, cls_val in CLASS_LABELS.items():
    true_mask = (y_test_cls == cls_val)
    pred_mask = (y_pred_cls == cls_val)
    tp = np.sum(true_mask & pred_mask)
    pa_chi2_list.append(tp / true_mask.sum() * 100 if true_mask.sum() > 0 else 0)
    ua_chi2_list.append(tp / pred_mask.sum() * 100 if pred_mask.sum() > 0 else 0)

pa_std  = np.array(pa_std_list)
ua_std  = np.array(ua_std_list)
pa_chi2 = np.array(pa_chi2_list)
ua_chi2 = np.array(ua_chi2_list)

x = np.arange(n_classes)
w = 0.2

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharey=True)

# Producer's Accuracy
axes[0].bar(x - w/2, pa_std,  width=w, label='Standard MLC',
            color='#4C72B0', edgecolor='white')
axes[0].bar(x + w/2, pa_chi2, width=w, label=f'Chi² ({CHI2_CONFIDENCE:.0%})',
            color='#DD5522', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels([n.replace('_',' ').title() for n in class_names],
                        rotation=30, ha='right', fontsize=9)
axes[0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0].set_title("Producer's Accuracy", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 105)
axes[0].grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

# User's Accuracy
axes[1].bar(x - w/2, ua_std,  width=w, label='Standard MLC',
            color='#4C72B0', edgecolor='white')
axes[1].bar(x + w/2, ua_chi2, width=w, label=f'Chi² ({CHI2_CONFIDENCE:.0%})',
            color='#DD5522', edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels([n.replace('_',' ').title() for n in class_names],
                        rotation=30, ha='right', fontsize=9)
axes[1].set_title("User's Accuracy", fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

fig.suptitle('Accuracy Comparison: Standard MLC vs Chi-squared Threshold',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'accuracy_comparison_bar.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 9. PIXEL AREA SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

im = axes[0].imshow(pixel_areas / 1e6, cmap='magma', origin='upper')
axes[0].set_title('Pixel Area (km²)', fontsize=12, fontweight='bold')
cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Area (km²)', fontsize=10)

axes[1].hist(pa_km2.ravel(), bins=80, color='#4C72B0',
             edgecolor='white', linewidth=0.3, alpha=0.85)
axes[1].axvline(np.median(pa_km2), color='#DD5522', linestyle='--',
                linewidth=1.5, label=f'Median: {np.median(pa_km2):.4f} km²')
axes[1].set_xlabel('Pixel Area (km²)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Pixel Area Distribution', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

fig.suptitle('VIIRS Scene — Pixel Area Geometry',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'pixel_area_summary.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 10. LaTeX — AREA COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
def _save_area_latex(df, filepath):
    lines = []
    lines.append(r"% Auto-generated by 06_accuracy_and_area.py")
    lines.append(r"% Requires: \usepackage{booktabs, adjustbox}")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Class Area Comparison Across Methods}")
    lines.append(r"  \label{tab:area_comparison}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    n_cols = len(df.columns)
    col_spec = "l" + " r" * (n_cols - 1)
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    headers = [r"\textbf{" + c.replace('%', r'\%').replace('²', r'$^2$') + "}"
               for c in df.columns]
    lines.append("    " + " & ".join(headers) + r" \\")
    lines.append(r"    \midrule")
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:,.2f}")
            elif isinstance(v, (int, np.integer)):
                vals.append(f"{int(v):,}")
            else:
                vals.append(str(v).replace('_', r'\_'))
        lines.append("    " + " & ".join(vals) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(r"\end{table}")
    with open(str(filepath), 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[LaTeX] Area comparison → {filepath}")

_save_area_latex(df_compare, TABLE_DIR / 'area_comparison.tex')

# ══════════════════════════════════════════════════════════════════════════════
# 11. LaTeX — ACCURACY SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def _save_accuracy_latex(df, filepath):
    lines = []
    lines.append(r"% Auto-generated by 06_accuracy_and_area.py")
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Final Classification Accuracy Summary}")
    lines.append(r"  \label{tab:accuracy_summary}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Method} & \textbf{OA (\%)} & "
                 r"\textbf{Pixels} & \textbf{Rejected} & "
                 r"\textbf{Rej. (\%)} \\")
    lines.append(r"    \midrule")
    for _, row in df.iterrows():
        method = str(row['Method']).replace('%', r'\%').replace('²', r'$^2$')
        oa     = f"{row['Overall Accuracy (%)']:.2f}"
        pix    = f"{int(row['Pixels Evaluated']):,}"
        rej    = f"{int(row['Pixels Rejected']):,}"
        rejp   = f"{row['Rejection Rate (%)']:.2f}"
        lines.append(f"    {method} & {oa} & {pix} & {rej} & {rejp}" + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    with open(str(filepath), 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[LaTeX] Accuracy summary → {filepath}")

_save_accuracy_latex(df_summary, TABLE_DIR / 'accuracy_summary.tex')

# ══════════════════════════════════════════════════════════════════════════════
# 12. FILE MANIFEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  OUTPUT FILE MANIFEST")
print("=" * 70)

print(f"\n  Plots ({PLOT_DIR}):")
for f in sorted(PLOT_DIR.glob('*.png')):
    print(f"    • {f.name}")

print(f"\n  Tables ({TABLE_DIR}):")
for f in sorted(TABLE_DIR.glob('*.*')):
    print(f"    • {f.name}")

print(f"\n  Models / Checkpoints ({MODEL_DIR}):")
for f in sorted(MODEL_DIR.glob('*.pkl')):
    print(f"    • {f.name}")

print("\n[06] Done — accuracy, area, and spectral statistics complete.")
print("=" * 70)