"""
06_accuracy_and_area.py
=======================
Final accuracy comparisons, pixel-area-based class area summaries
for training data, testing data, K-Means clusters, MLC full scene,
and chi-squared classification.  LaTeX table exports.

Inputs
------
checkpoint_01.pkl, checkpoint_02.pkl, checkpoint_03.pkl,
checkpoint_04.pkl, checkpoint_05.pkl

Outputs
-------
Area summary tables  → TABLE_DIR / area_*.tex
Comparison figures   → PLOT_DIR  / area_comparison_*.png
"""
# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import (
    DATA_DIR, PLOT_DIR, TABLE_DIR, MODEL_DIR, HEIGHT, WIDTH,
    ALL_BANDS, REFLECTIVE_BANDS, EMISSIVE_BANDS,
    CLASS_LABELS, CLASS_LABELS_EXT, CLASS_DISPLAY_NAMES,
    CLASS_COLORS_LIST, CLASS_COLORS_EXT,
    NADIR_ALONG_TRACK, NADIR_CROSS_TRACK, ORBITAL_HEIGHT,
    CHI2_CONFIDENCE,
    apply_plot_style, save_checkpoint, load_checkpoint,
)
from functions_project2 import (
    summarize_class_areas, calculate_accuracy_metrics,
    build_spectral_statistics_table,
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

n_classes      = len(CLASS_LABELS)
pixel_areas_flat = pixel_areas.ravel()

# ══════════════════════════════════════════════════════════════════════════════
# 2. AREA SUMMARIES — PER DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CLASS AREA SUMMARIES")
print("=" * 70)

# ── 2a. Training data ────────────────────────────────────────────────────────
df_train = summarize_class_areas(
    y_train, a_train, CLASS_LABELS,
    source_name="Training Data",
)
print("\n── Training Data ──")
print(df_train.to_string(index=False))

# ── 2b. Testing data ─────────────────────────────────────────────────────────
df_test = summarize_class_areas(
    y_test, a_test, CLASS_LABELS,
    source_name="Testing Data",
)
print("\n── Testing Data ──")
print(df_test.to_string(index=False))

# ── 2c. MLC full scene ───────────────────────────────────────────────────────
mlc_pred_flat = class_map.ravel()

df_mlc = summarize_class_areas(
    mlc_pred_flat, pixel_areas_flat, CLASS_LABELS,
    source_name="MLC Full Scene",
)
print("\n── MLC Full Scene ──")
print(df_mlc.to_string(index=False))

# ── 2d. K-Means clusters ─────────────────────────────────────────────────────
kmeans_flat   = cluster_map.ravel()
kmeans_labels = cluster_map.copy()

df_kmeans = summarize_class_areas(
    kmeans_flat, pixel_areas_flat, CLASS_LABELS,
    source_name="K-Means Clusters",
)
print("\n── K-Means Clusters ──")
print(df_kmeans.to_string(index=False))

# ── 2e. Chi-squared classification ───────────────────────────────────────────
class_labels_chi2 = {**CLASS_LABELS, "Unclassified": n_classes}

chi2_flat = class_map_chi2.ravel()

df_chi2 = summarize_class_areas(
    chi2_flat, pixel_areas_flat, class_labels_chi2,
    source_name=f"Chi² Classification ({CHI2_CONFIDENCE:.0%})",
)
print(f"\n── Chi² Classification ({CHI2_CONFIDENCE:.0%}) ──")
print(df_chi2.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMBINED AREA COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
#    Merge all source summaries into one wide-format comparison table.

all_dfs = []
for df, tag in [(df_train, 'Training'),
                (df_test,  'Testing'),
                (df_mlc,   'MLC'),
                (df_kmeans,'K-Means'),
                (df_chi2,  'Chi²')]:
    tmp = df.copy()
    tmp = tmp.rename(columns={
        'Area (km²)':     f'{tag} Area (km²)',
        'Area (%)':       f'{tag} Area (%)',
        'Pixel Count':    f'{tag} Pixels',
    })
    # Keep only class + the renamed columns
    keep_cols = ['Class'] + [c for c in tmp.columns if tag in c]
    tmp = tmp[keep_cols]
    all_dfs.append(tmp)

# Merge on Class name
df_compare = all_dfs[0]
for df_other in all_dfs[1:]:
    df_compare = df_compare.merge(df_other, on='Class', how='outer')

df_compare = df_compare.fillna(0)

print("\n" + "=" * 100)
print("  AREA COMPARISON — ALL SOURCES")
print("=" * 100)
print(df_compare.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. AREA COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
#    Group bar chart: one group per class, one bar per data source.

sources     = ['Training', 'Testing', 'MLC', 'K-Means', 'Chi²']
area_cols   = [f'{s} Area (km²)' for s in sources]
source_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

# Only use the real classes (exclude Unclassified row if present)
df_plot = df_compare[df_compare['Class'] != 'Unclassified'].copy()
class_names_plot = df_plot['Class'].values

x        = np.arange(len(class_names_plot))
n_src    = len(sources)
bar_w    = 0.15

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

for i, (src, col, clr) in enumerate(zip(sources, area_cols, source_colors)):
    if col in df_plot.columns:
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
# 5. PIXEL AREA STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PIXEL AREA GEOMETRY STATISTICS")
print("=" * 60)

pa_km2 = pixel_areas / 1e6

print(f"  Nadir pixel:   {NADIR_ALONG_TRACK * NADIR_CROSS_TRACK / 1e6:.6f} km²")
print(f"  Scene min:     {pa_km2.min():.6f} km²")
print(f"  Scene max:     {pa_km2.max():.6f} km²")
print(f"  Scene mean:    {pa_km2.mean():.6f} km²")
print(f"  Scene median:  {np.median(pa_km2):.6f} km²")
print(f"  Total scene:   {pa_km2.sum():.2f} km²")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL ACCURACY SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
#    Combine standard MLC and chi-squared metrics into one comparison.

df_acc_std, _ = calculate_accuracy_metrics(y_test, y_pred, CLASS_LABELS)
n_classified  = np.sum(y_pred_chi2 < n_classes)
n_total_test  = len(y_test)

# Build a compact comparison
print("\n" + "=" * 70)
print("  FINAL ACCURACY COMPARISON")
print("=" * 70)

# Standard MLC overall
std_oa = df_acc_std[df_acc_std['Class'] == '** OVERALL **']["Producer's Acc (%)"].values[0]

# Chi-squared overall (all pixels) and (classified only)
from functions_project2 import chi2_accuracy_report
df_chi2_full, _ = chi2_accuracy_report(
    y_test, y_pred_chi2, CLASS_LABELS,
    n_classes=n_classes,
    confidence=CHI2_CONFIDENCE,
    chi2_threshold=ckpt05['threshold_test'],
)
chi2_oa_all  = df_chi2_full.iloc[-1]["Producer's Acc (%)"]
chi2_oa_cls  = df_chi2_full.iloc[-1]["User's Acc (%)"]

summary_data = {
    'Method':                 ['Standard MLC',
                               f'Chi² ({CHI2_CONFIDENCE:.0%}) — all pixels',
                               f'Chi² ({CHI2_CONFIDENCE:.0%}) — classified only'],
    'Overall Accuracy (%)':   [std_oa, chi2_oa_all, chi2_oa_cls],
    'Pixels Evaluated':       [n_total_test, n_total_test, n_classified],
    'Pixels Rejected':        [0,
                               n_total_test - n_classified,
                               0],
    'Rejection Rate (%)':     [0.0,
                               (n_total_test - n_classified) / n_total_test * 100,
                               0.0],
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 7. PER-CLASS ACCURACY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
class_names = list(CLASS_LABELS.keys())

# Extract per-class Producer's and User's Accuracy for both methods
pa_std = df_acc_std[df_acc_std['Class'] != '** OVERALL **']["Producer's Acc (%)"].values
ua_std = df_acc_std[df_acc_std['Class'] != '** OVERALL **']["User's Acc (%)"].values

pa_chi2 = df_chi2_full.iloc[:n_classes]["Producer's Acc (%)"].values
ua_chi2 = df_chi2_full.iloc[:n_classes]["User's Acc (%)"].values

# Side-by-side bar chart
x = np.arange(n_classes)
w = 0.2

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharey=True)

# ── Producer's Accuracy ──────────────────────────────────────────────────────
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

# ── User's Accuracy ──────────────────────────────────────────────────────────
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
# 8. SPECTRAL STATISTICS — TRAINING & TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Spectral Statistics (Training) ──")
df_wide_train, df_long_train = build_spectral_statistics_table(
    X_raw=X_train_raw,
    y=y_train,
    all_band_names=ALL_BANDS,
    class_labels=CLASS_LABELS,
    save_dir=TABLE_DIR,
)

print("\n── Spectral Statistics (Testing) ──")
df_wide_test, df_long_test = build_spectral_statistics_table(
    X_raw=X_test_raw,
    y=y_test,
    all_band_names=ALL_BANDS,
    class_labels=CLASS_LABELS,
    save_dir=TABLE_DIR,
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. LATEX EXPORT — AREA COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def _save_area_comparison_latex(df, filepath):
    """Write area comparison DataFrame as a LaTeX table."""
    lines = []
    lines.append(r"% ── Auto-generated by 06_accuracy_and_area.py ──")
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Class Area Comparison Across Methods}")
    lines.append(r"  \label{tab:area_comparison}")
    lines.append(r"  \small")

    # Build column spec
    n_cols = len(df.columns)
    col_spec = "l" + " r" * (n_cols - 1)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")

    # Header
    headers = [r"\textbf{" + c.replace('%', r'\%').replace('²', r'$^2$') + "}"
               for c in df.columns]
    lines.append("    " + " & ".join(headers) + r" \\")
    lines.append(r"    \midrule")

    # Data rows
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
    lines.append(r"\end{table}")

    filepath = str(filepath)
    with open(filepath, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[LaTeX] Area comparison saved → {filepath}")


_save_area_comparison_latex(df_compare,
                            TABLE_DIR / 'area_comparison.tex')

# ══════════════════════════════════════════════════════════════════════════════
# 10. LATEX EXPORT — FINAL ACCURACY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def _save_accuracy_summary_latex(df, filepath):
    """Write the final accuracy comparison as LaTeX."""
    lines = []
    lines.append(r"% ── Auto-generated by 06_accuracy_and_area.py ──")
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"")
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

    filepath = str(filepath)
    with open(filepath, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[LaTeX] Accuracy summary saved → {filepath}")


_save_accuracy_summary_latex(df_summary,
                             TABLE_DIR / 'accuracy_summary.tex')

# ══════════════════════════════════════════════════════════════════════════════
# 11. FINAL SCENE AREA BREAKDOWN  (pixel area histogram)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# ── Left: Pixel area spatial map ──────────────────────────────────────────────
im = axes[0].imshow(pixel_areas / 1e6, cmap='magma', origin='upper')
axes[0].set_title('Pixel Area (km²)', fontsize=12, fontweight='bold')
cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Area (km²)', fontsize=10)

# ── Right: Histogram of pixel areas ──────────────────────────────────────────
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
# 12. PRINT FINAL FILE MANIFEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  OUTPUT FILE MANIFEST")
print("=" * 70)

print(f"\n  Plots ({PLOT_DIR}):")
for f in sorted(PLOT_DIR.glob('*.png')):
    print(f"    • {f.name}")

print(f"\n  Tables ({TABLE_DIR}):")
for f in sorted(TABLE_DIR.glob('*.tex')):
    print(f"    • {f.name}")

print(f"\n  Models / Checkpoints ({MODEL_DIR}):")
for f in sorted(MODEL_DIR.glob('*.pkl')):
    print(f"    • {f.name}")

print("\n[06] Done — accuracy and area analysis complete.")
print("=" * 70)