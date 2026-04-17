"""
functions_project2.py — All reusable functions for Project 2.
Plotting, analysis, classification, accuracy assessment.
"""
# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path

# ── Numerics ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, chi2, kstest

# ── Geospatial / Satellite ───────────────────────────────────────────────────
import xarray as xr
import rioxarray as rx
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from satpy.writers import get_enhanced_image
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_samples, silhouette_score)
import joblib

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Project config (single source of truth) ──────────────────────────────────
from p2_config import (
    BAND_WAVELENGTHS, ALL_BANDS, REFLECTIVE_BANDS, EMISSIVE_BANDS,
    CLASS_LABELS, CLASS_NAMES, CLASS_COLORS, CLASS_COLORS_LIST,
    CLASS_LABELS_EXT, CLASS_COLORS_EXT,
    HEIGHT, WIDTH, add_geo_ticks
)

# ══════════════════════════════════════════════════════════════════════════════
#                              FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ... (all your existing functions, cleaned of inline constant definitions)



#----------------------------------------------------------------------------------------------------------------------#
#                                         CLASSES, DEFINITIONS, ETC.                                                   #
#----------------------------------------------------------------------------------------------------------------------#
BAND_WAVELENGTHS = {
    'M01': 0.412,
    'M02': 0.445,
    'M03': 0.483,
    'M04': 0.555,
    'M05': 0.672,
    'M06': 0.746,
    'M07': 0.865,
    'M08': 1.240,
    'M09': 1.378,
    'M10': 1.610,
    'M11': 2.255,
    'M12': 3.700,
    'M13': 4.050,
    'M14': 8.550,
    'M15': 10.760,
    'M16': 12.015,
}

# ── CLASS COLORS (consistent with your classification map) ─────────
CLASS_COLORS = {
    'water':      '#2166ac',
    'cloud':      '#d1d1d1',
    'snow':       '#a6cee3',
    'smoke':      '#6a3d9a',
    'bare_soil':  '#d6a86b',
    'vegetation': '#4dac26',
    # 'urban':      '#e31a1c',
}

band_names = ['M01', 'M02', 'M03', 'M04',
              'M05', 'M06', 'M07', 'M08',
              'M09', 'M10', 'M11', 'M12',
              'M13', 'M14', 'M15', 'M16']

# VIIRS M-bands: reflective (solar) vs emissive (thermal)
REFLECTIVE_BANDS = ['M01', 'M02', 'M03',
                    'M04', 'M05', 'M06',
                    'M07', 'M08', 'M09',
                    'M10','M11']

EMISSIVE_BANDS   = ['M12', 'M13', 'M14',
                    'M15', 'M16']

# Adjust these lists to match exactly what's in your ds.data_vars
ALL_BANDS = REFLECTIVE_BANDS + EMISSIVE_BANDS

CLASS_LABELS = {
    'water':      0,
    'cloud':      1,
    'snow':       2,
    'smoke':      3,
    'bare_soil':  4,
    'vegetation': 5,
    # 'urban':      6,
}

class_names  = ['Water', 'Cloud',
                'Snow', 'Smoke',
                'Bare Soil', 'Vegetation',
                # 'Urban'
                ]

class_colors = ['#2166ac',   # Water      - blue
                '#f0f0f0',   # Cloud      - light grey
                '#a6cee3',   # Snow       - icy cyan      ← changed
                '#6a3d9a',   # Smoke      - dark purple   ← changed
                '#d6a86b',   # Bare Soil  - tan
                '#4dac26',   # Vegetation - green
                # '#e31a1c'
                ]   # Urban      - red


#----------------------------------------------------------------------------------------------------------------------#
#                                           FUNCTIONS                                                                  #
#----------------------------------------------------------------------------------------------------------------------#
def plot_rgb_subset(
    ds=None,
    y_slice=None,
    x_slice=None,
    rgb=None,
    custom_rgb=None,
    ax=None,
    overlay=None,
    overlay_alpha=0.4,
    make_transparent_zero=False,
    threshold=0,
    title=None,
    show=True
):
    """
    Flexible RGB plotting with optional overlays and axis reuse.

    You can either:
    - pass ds + slices (auto-generate RGB)
    - OR pass a precomputed rgb array

    Parameters
    ----------
    overlay : 2D array (optional)
        Mask or data to overlay
    """

    # --- Get RGB ---
    if rgb is None:
        img = get_enhanced_image(ds['true_color'])
        rgb = img.data.transpose('y', 'x', 'bands').values
        rgb = np.clip(rgb, 0, 1)

    # --- Subset ---
    if y_slice is not None and x_slice is not None:
        rgb = rgb[y_slice, x_slice, :]

    if custom_rgb is not None:
        rgb = custom_rgb

    # --- Transparency handling ---
    if make_transparent_zero:
        alpha = np.any(rgb > threshold, axis=2).astype(float)
        rgb = np.dstack((rgb, alpha))

    # --- Axis handling ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # --- Plot base image ---
    ax.imshow(rgb, origin='upper')

    # --- Overlay ---
    if overlay is not None:
        if y_slice is not None and x_slice is not None:
            overlay = overlay[y_slice, x_slice]

        ax.imshow(
            overlay,
            cmap='Reds',   # change as needed
            alpha=overlay_alpha,
            origin='upper'
        )

    # --- Formatting ---
    ax.set_title(title if title else "")
    ax.axis('off')

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_spectral_response(
    X_raw,
    y,
    all_band_names,
    reflective_bands,
    emissive_bands,
    class_labels,
    class_colors=CLASS_COLORS,
    band_wavelengths=BAND_WAVELENGTHS,
    mode='all_classes',
    show_individual=True,
    max_individual=300,
    figsize=(13, 5),
    dpi=150,
    title_prefix='VIIRS Spectral Response',
    save_dir=None,
    # ── NEW: y-axis limits ──────────────────────────────────────────
    ylim_reflective=(0, 100),   # (ymin, ymax) or None for auto-scale
    ylim_emissive=None,         # (ymin, ymax) or None for auto-scale
):
    """
    Plot spectral response curves split into Reflective and Emissive subplots.

    Modes
    -----
    'all_classes' : one figure per band type, all classes overlaid
    'per_class'   : one figure per class, reflective + emissive side by side

    Y-axis limits
    -------------
    ylim_reflective : tuple (ymin, ymax) or None
        Applied to all reflective-band axes. Default (0, 100) for
        reflectance percentage. Pass None to let matplotlib auto-scale.
    ylim_emissive : tuple (ymin, ymax) or None
        Applied to all emissive-band axes. Default None (auto-scale) since
        brightness temperature range varies with scene. Example: (200, 330).
    """

    sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)

    all_band_names = list(all_band_names)

    # Column indices for each band group
    ref_idx = [all_band_names.index(b) for b in reflective_bands
               if b in all_band_names]
    emi_idx = [all_band_names.index(b) for b in emissive_bands
               if b in all_band_names]

    ref_wl  = np.array([band_wavelengths[b] for b in reflective_bands
                        if b in all_band_names])
    emi_wl  = np.array([band_wavelengths[b] for b in emissive_bands
                        if b in all_band_names])

    ref_names = [b for b in reflective_bands if b in all_band_names]
    emi_names = [b for b in emissive_bands   if b in all_band_names]

    # ── Pre-compute per-class stats ───────────────────────────────
    class_stats = {}
    for class_name, label_int in class_labels.items():
        mask    = y == label_int
        X_class = X_raw[mask]
        if len(X_class) == 0:
            continue
        class_stats[class_name] = {
            'n':        len(X_class),
            'X_raw':    X_class,
            'ref_mean': np.nanmean(X_class[:, ref_idx], axis=0),
            'ref_std':  np.nanstd( X_class[:, ref_idx], axis=0),
            'emi_mean': np.nanmean(X_class[:, emi_idx], axis=0),
            'emi_std':  np.nanstd( X_class[:, emi_idx], axis=0),
        }

    # ── NEW: helper to apply ylim only when specified ─────────────
    def _apply_ylim(ax, ylim):
        """Set y-axis limits if ylim is provided; otherwise leave as auto."""
        if ylim is not None:
            ax.set_ylim(ylim)

    # ── Internal draw helper ──────────────────────────────────────
    def _draw_class_on_ax(ax, wl, band_names_grp, mean, std,
                          X_cls_cols, color, label, show_ind):
        """Draw individual traces + std envelope + mean line on ax."""
        n = len(X_cls_cols)

        if show_ind and n > 0:
            n_draw = min(n, max_individual)
            idx    = np.random.choice(n, n_draw, replace=False)
            for pixel in X_cls_cols[idx]:
                ax.plot(wl, pixel, color=color,
                        alpha=0.05, linewidth=0.6, zorder=1)

        ax.fill_between(wl, mean - std,   mean + std,
                        alpha=0.20, color=color, zorder=2)
        ax.fill_between(wl, mean - 2*std, mean + 2*std,
                        alpha=0.08, color=color, zorder=2)
        ax.plot(wl, mean, color=color, linewidth=2.5,
                marker='o', markersize=4, zorder=3, label=label)

        ax.set_xticks(wl)
        ax.set_xticklabels([f"{w:.3f}\n{b}" for w, b in
                            zip(wl, band_names_grp)],
                           rotation=45, linespacing=1.4, fontsize=8)
        ax.set_xlabel('Wavelength (µm)', fontsize=11)
        ax.margins(x=0.04)

    # ── MODE 1: All classes ───────────────────────────────────────
    if mode == 'all_classes':

        for band_type, wl, band_names_grp, mean_key, std_key, raw_idx, ylabel, ylim in [
            ('Reflective', ref_wl, ref_names,
             'ref_mean', 'ref_std', ref_idx,
             'Reflectance (%)',            ylim_reflective),   # ← NEW
            ('Emissive',   emi_wl, emi_names,
             'emi_mean', 'emi_std', emi_idx,
             'Brightness Temperature (K)', ylim_emissive),     # ← NEW
        ]:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            for class_name, stats in class_stats.items():
                color = class_colors.get(class_name, '#333333')
                n     = stats['n']
                _draw_class_on_ax(
                    ax, wl, band_names_grp,
                    mean       = stats[mean_key],
                    std        = stats[std_key],
                    X_cls_cols = stats['X_raw'][:, raw_idx],
                    color      = color,
                    label      = f"{class_name.replace('_',' ').title()} (n={n:,})",
                    show_ind   = False,
                )

            _apply_ylim(ax, ylim)                              # ← NEW

            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{title_prefix} — {band_type} Bands — All Classes',
                         fontsize=13, fontweight='bold', pad=12)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      borderaxespad=0., framealpha=0.9,
                      edgecolor='#cccccc', fontsize=9)
            sns.despine(ax=ax)
            fig.tight_layout()

            if save_dir is not None:
                fig.savefig(save_dir / f'spectral_all_{band_type.lower()}.png',
                            dpi=dpi, bbox_inches='tight')
            plt.show()

    # ── MODE 2: Per class ─────────────────────────────────────────
    elif mode == 'per_class':

        for class_name, stats in class_stats.items():
            color = class_colors.get(class_name, '#333333')
            n     = stats['n']
            label = f"Mean (n={n:,})"

            fig, (ax_ref, ax_emi) = plt.subplots(
                1, 2, figsize=(figsize[0]*1.6, figsize[1]), dpi=dpi
            )

            # Left: Reflective
            _draw_class_on_ax(
                ax_ref, ref_wl, ref_names,
                mean       = stats['ref_mean'],
                std        = stats['ref_std'],
                X_cls_cols = stats['X_raw'][:, ref_idx],
                color      = color,
                label      = label,
                show_ind   = show_individual,
            )
            _apply_ylim(ax_ref, ylim_reflective)               # ← NEW
            ax_ref.set_ylabel('Reflectance (%)', fontsize=11)
            ax_ref.set_title('Reflective Bands\n(Solar, 0.4–2.3 µm)',
                             fontsize=11, fontweight='bold')
            ax_ref.legend(loc='upper right', framealpha=0.9,
                          edgecolor='#cccccc', fontsize=9)

            # Right: Emissive
            _draw_class_on_ax(
                ax_emi, emi_wl, emi_names,
                mean       = stats['emi_mean'],
                std        = stats['emi_std'],
                X_cls_cols = stats['X_raw'][:, emi_idx],
                color      = color,
                label      = label,
                show_ind   = show_individual,
            )
            _apply_ylim(ax_emi, ylim_emissive)                 # ← NEW
            ax_emi.set_ylabel('Brightness Temperature (K)', fontsize=11)
            ax_emi.set_title('Emissive Bands\n(Thermal, 4–12 µm)',
                             fontsize=11, fontweight='bold')
            ax_emi.legend(loc='upper right', framealpha=0.9,
                          edgecolor='#cccccc', fontsize=9)

            fig.suptitle(
                f'{title_prefix} — '
                f'{class_name.replace("_"," ").title()}',
                fontsize=13, fontweight='bold', y=1.02
            )
            sns.despine()
            fig.tight_layout()

            if save_dir is not None:
                fig.savefig(save_dir / f'spectral_{class_name}.png',
                            dpi=dpi, bbox_inches='tight')
            plt.show()

    return class_stats


def plot_pca_rgb(
    X_pca, height, width,
    pc_indices=(0, 1, 2),
    stretch=2,
    figsize=(10, 8),
    dpi=150,
    # ── NEW: geo-labeling + title ─────────────────────────────────
    lon=None,
    lat=None,
    title=None,
    save_path=None,
    show=True,
    x=None,            # kept for backward compat (slice)
    y=None,            # kept for backward compat (slice)
):
    """
    Create a false-color RGB composite from three PCA components
    and display it with optional geographic axis labels.

    Parameters
    ----------
    X_pca      : np.ndarray (n_pixels, n_components)
    height     : int — image height in pixels
    width      : int — image width in pixels
    pc_indices : tuple of 3 ints — which PCs map to (R, G, B)
    stretch    : float — percentile stretch strength
                 (higher = more contrast; 2 → clip at 2nd/98th pctile)
    lon        : np.ndarray (height, width) or None — longitude grid
    lat        : np.ndarray (height, width) or None — latitude grid
    title      : str or None — figure title
    save_path  : str/Path or None — save figure to this path
    show       : bool — call plt.show()

    Returns
    -------
    rgb : np.ndarray (height, width, 3) — the stretched RGB array [0, 1]
    """

    # ── Build the 3-channel image ─────────────────────────────────────────────
    channels = []
    for pc_idx in pc_indices:
        ch = X_pca[:, pc_idx].reshape(height, width)
        channels.append(ch)
    rgb = np.stack(channels, axis=-1)

    # ── Percentile stretch per channel ────────────────────────────────────────
    def _stretch(ch, pct=stretch):
        lo = np.nanpercentile(ch, pct)
        hi = np.nanpercentile(ch, 100 - pct)
        ch = (ch - lo) / (hi - lo) if hi > lo else ch * 0
        return np.clip(ch, 0, 1)

    for i in range(3):
        rgb[:, :, i] = _stretch(rgb[:, :, i])

    # ── Optional subset (backward compat) ─────────────────────────────────────
    if x is not None and y is not None:
        rgb = rgb[y, x, :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(rgb, origin='upper', interpolation='nearest')

    # ── Geo ticks or pixel-only ───────────────────────────────────────────────
    if lon is not None and lat is not None:
        add_geo_ticks(ax, lon, lat, height=rgb.shape[0], width=rgb.shape[1])
    else:
        ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────────────
    pc_r, pc_g, pc_b = [f'PC{i+1}' for i in pc_indices]
    default_title = f'VIIRS PCA False-Color Composite  (R={pc_r}, G={pc_g}, B={pc_b})'
    ax.set_title(title if title else default_title,
                 fontsize=13, fontweight='bold', pad=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[plot_pca_rgb] Saved → {save_path}")

    if show:
        plt.show()

    return rgb


def plot_pca_loadings(pca_model, band_names, n_components=4):
    """
    Plot PCA loading scores for the first n_components principal components.

    Parameters
    ----------
    pca_model  : fitted sklearn PCA object
    band_names : list of str — VIIRS band labels (e.g. ['M01', 'M02', ...])
    n_components : int — number of PCs to display (default: 4)
    """

    # ── Build loadings DataFrame ───────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
        index=band_names
    )
    loadings_plot = loadings.iloc[:, :n_components]

    # ── Layout ────────────────────────────────────────────────────────────────
    # One subplot per PC so bars don't crowd each other
    fig, axes = plt.subplots(
        1, n_components,
        figsize=(4.5 * n_components, 5.5),
        sharey=True
    )

    # ── Color palette (one color per PC) ─────────────────────────────────────
    palette = ["#4C72B0", "#DD5522", "#2E8B57", "#8B5E9C"]

    n_bands = len(band_names)
    x      = np.arange(n_bands)

    for idx, (ax, pc) in enumerate(zip(axes, loadings_plot.columns)):

        values    = loadings_plot[pc].values
        color     = palette[idx % len(palette)]
        var_expl  = pca_model.explained_variance_ratio_[idx] * 100

        # ── Bars: color-coded positive / negative ────────────────────────────
        bar_colors = [color if v >= 0 else "#B0B0B0" for v in values]
        bars = ax.bar(x, values, color=bar_colors, width=0.65,
                      edgecolor="white", linewidth=0.6, zorder=2)

        # ── Zero reference line ───────────────────────────────────────────────
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--",
                   alpha=0.6, zorder=3)

        # ── Subtle threshold lines at ±0.3 (common significance rule) ────────
        for thresh in [0.3, -0.3]:
            ax.axhline(thresh, color=color, linewidth=0.8,
                       linestyle=":", alpha=0.5, zorder=1)

        # ── Value labels on bars ──────────────────────────────────────────────
        for bar, val in zip(bars, values):
            offset = 0.015 if val >= 0 else -0.015
            va     = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.2f}",
                ha="center", va=va,
                fontsize=6.5, color="dimgray"
            )

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, rotation=45, ha="right",
                           fontsize=9)
        ax.set_xlim(-0.6, n_bands - 0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        # ── Per-panel title ───────────────────────────────────────────────────
        ax.set_title(
            f"{pc}\n({var_expl:.1f}% variance)",
            fontsize=11, fontweight="bold", color=color, pad=8
        )

    # ── Shared y-axis label (leftmost panel only) ─────────────────────────────
    axes[0].set_ylabel("Loading Score", fontsize=11, labelpad=8)

    # ── Legend: positive / negative coding ───────────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette[0], label="Positive loading"),
        Patch(facecolor="#B0B0B0",  label="Negative loading"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=2,
        fontsize=9, framealpha=0.9,
        edgecolor="lightgray",
        bbox_to_anchor=(0.5, -0.02)
    )

    # ── Threshold annotation ──────────────────────────────────────────────────
    fig.text(
        0.99, 0.01,
        "Dotted lines: |loading| = 0.30 significance threshold",
        ha="right", va="bottom", fontsize=7.5, color="gray", style="italic"
    )

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "PCA Loadings — Contribution of Each VIIRS Band per Principal Component",
        fontsize=13, fontweight="bold", y=1.01
    )

    fig.tight_layout()
    plt.savefig("viirs_pca_loadings.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_pca_scree(explained_variance, cumulative_variance,
                   threshold=0.90, save_path="viirs_pca_scree.png"):
    """
    Plot a professional PCA scree plot with individual and cumulative
    explained variance for VIIRS data.

    Parameters
    ----------
    explained_variance  : array-like — per-component explained variance ratio
                          (e.g. pca.explained_variance_ratio_)
    cumulative_variance : array-like — cumulative explained variance ratio
                          (e.g. np.cumsum(pca.explained_variance_ratio_))
    threshold           : float — cumulative variance threshold to highlight
                          as a reference line (default: 0.90)
    save_path           : str or None — file path to save the figure;
                          pass None to skip saving (default: "viirs_pca_scree.png")
    """

    # ── Color palette ─────────────────────────────────────────────────────────
    bar_color = "#4C72B0"  # muted blue
    line_color = "#DD5522"  # burnt orange

    n_components = len(explained_variance)
    x = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ── Bar chart (individual variance) ───────────────────────────────────────
    bars = ax1.bar(
        x, explained_variance * 100,
        color=bar_color, alpha=0.85, width=0.6,
        zorder=2, label="Individual explained variance"
    )

    ax1.set_xlabel("Principal Component", fontsize=12, labelpad=8)
    ax1.set_ylabel("Individual Explained Variance (%)", fontsize=12, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.set_xlim(0.3, n_components + 0.7)
    ax1.set_ylim(0, max(explained_variance * 100) * 1.25)
    ax1.set_xticks(x)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    # ── Twin axis: cumulative variance line ────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(
        x, cumulative_variance * 100,
        color=line_color, linewidth=2.2, marker="o",
        markersize=5, zorder=3, label="Cumulative explained variance"
    )
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12, color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # ── Threshold line ─────────────────────────────────────────────────────────
    ax2.axhline(
        threshold * 100, color="gray", linestyle="--",
        linewidth=1.2, alpha=0.8, zorder=1
    )
    ax2.text(
        n_components + 0.55, threshold * 100 + 1.5,
        f"{int(threshold * 100)}% threshold",
        fontsize=9, color="gray", va="bottom", ha="right"
    )

    # ── Annotate the crossing point ────────────────────────────────────────────
    cross_idx = np.searchsorted(cumulative_variance, threshold)
    if cross_idx < n_components:
        ax2.axvline(
            cross_idx + 1, color="gray", linestyle=":",
            linewidth=1.2, alpha=0.8, zorder=1
        )
        ax2.annotate(
            f"PC {cross_idx + 1}",
            xy=(cross_idx + 1, cumulative_variance[cross_idx] * 100),
            xytext=(cross_idx + 2.5, cumulative_variance[cross_idx] * 100 - 8),
            fontsize=9, color=line_color,
            arrowprops=dict(arrowstyle="->", color=line_color, lw=1.2)
        )

    # ── Bar value labels ───────────────────────────────────────────────────────
    for bar, val in zip(bars, explained_variance * 100):
        if val > 1.5:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color=bar_color
            )

    # ── Combined legend ────────────────────────────────────────────────────────
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2, labels1 + labels2,
        loc="center right", fontsize=9,
        framealpha=0.9, edgecolor="lightgray"
    )

    # ── Title & layout ─────────────────────────────────────────────────────────
    plt.title(
        "VIIRS PCA — Explained Variance by Principal Component",
        fontsize=13, fontweight="bold", pad=14
    )
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_pca_scree] Figure saved → {save_path}")

    plt.show()


def plot_confusion_matrix(y_test, y_pred, class_labels,
                          save_path="mlc_confusion_matrix.png"):
    """
    Plot a professional side-by-side confusion matrix (raw counts + normalized)
    for a multi-class classifier, with row/column totals.

    Parameters
    ----------
    y_test       : array-like — true class labels (integer encoded)
    y_pred       : array-like — predicted class labels (integer encoded)
    class_labels : dict — mapping of class name → integer label
                   e.g. {"water": 0, "cloud": 1, ...}
    save_path    : str or None — file path to save the figure;
                   pass None to skip saving (default: "mlc_confusion_matrix.png")
    """

    # ── Data prep ──────────────────────────────────────────────────────────────
    class_names  = list(class_labels.keys())
    label_values = list(class_labels.values())
    n_classes    = len(class_names)

    cm      = confusion_matrix(y_test, y_pred, labels=label_values)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalize

    # ── Build extended matrices with totals row & column ───────────────────────
    #    Layout (n+1) × (n+1):
    #       rows 0..n-1, cols 0..n-1  = original CM
    #       col  n                     = row totals   (sum across predicted)
    #       row  n                     = col totals   (sum across true)
    #       cell (n, n)                = grand total

    # --- Raw counts extended ---
    row_totals = cm.sum(axis=1)                       # shape (n,)
    col_totals = cm.sum(axis=0)                       # shape (n,)
    grand_total = cm.sum()

    cm_ext = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    cm_ext[:n_classes, :n_classes] = cm
    cm_ext[:n_classes, n_classes]  = row_totals
    cm_ext[n_classes, :n_classes]  = col_totals
    cm_ext[n_classes, n_classes]   = grand_total

    # --- Normalized extended ---
    #     Row totals   → row-wise recall (mean or sum not meaningful; use NaN)
    #     Col totals   → per-class precision = diag / col_total
    #     Grand total  → overall accuracy
    cm_norm_ext = np.full((n_classes + 1, n_classes + 1), np.nan)
    cm_norm_ext[:n_classes, :n_classes] = cm_norm

    # Row totals column: show recall (diagonal / row total) — same as diag of cm_norm
    cm_norm_ext[:n_classes, n_classes] = np.diag(cm_norm)  # recall per class

    # Col totals row: show precision per class (diag / col_total)
    col_totals_safe = np.where(col_totals == 0, 1, col_totals)  # avoid /0
    precision = np.diag(cm).astype(float) / col_totals_safe
    cm_norm_ext[n_classes, :n_classes] = precision

    # Grand total cell: overall accuracy
    overall_acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    cm_norm_ext[n_classes, n_classes] = overall_acc

    # ── Extended label list ────────────────────────────────────────────────────
    extended_names = class_names + ["Total"]
    n_ext = n_classes + 1

    # ── Custom colormap (white → rich blue) ────────────────────────────────────
    cmap = LinearSegmentedColormap.from_list(
        "custom_blues", ["#FFFFFF", "#C6D9F0", "#4C72B0", "#1B3A6B"]
    )

    # ── Totals strip uses a neutral gray colormap ──────────────────────────────
    total_bg = "#E8E8E8"

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), dpi=150)
    fig.patch.set_facecolor("#F8F9FA")

    panels = [
        (cm_ext,                                "Raw Counts",           "raw",  None),
        (np.round(cm_norm_ext, 4),              "Normalized (Recall)",  "norm", [0, 1]),
    ]

    for ax, (matrix, subtitle, panel_type, vlim) in zip(axes, panels):

        # ── Draw the core n×n confusion matrix region ─────────────────────────
        core = matrix[:n_classes, :n_classes].astype(float)
        vmin, vmax = (core.min(), core.max()) if vlim is None else vlim
        im = ax.imshow(matrix.astype(float), interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # ── Gray-out the totals row & column ──────────────────────────────────
        for k in range(n_ext):
            # Last column (row totals)
            ax.add_patch(plt.Rectangle(
                (n_classes - 0.5, k - 0.5), 1, 1,
                fill=True, facecolor=total_bg, edgecolor="white",
                linewidth=1.0, zorder=2
            ))
            # Last row (col totals)
            ax.add_patch(plt.Rectangle(
                (k - 0.5, n_classes - 0.5), 1, 1,
                fill=True, facecolor=total_bg, edgecolor="white",
                linewidth=1.0, zorder=2
            ))

        # ── Colorbar ──────────────────────────────────────────────────────────
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        if vlim == [0, 1]:
            cbar.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
            )

        # ── Cell annotations ──────────────────────────────────────────────────
        thresh = (vmax + vmin) / 2
        for i in range(n_ext):
            for j in range(n_ext):
                val = matrix[i, j]
                is_total_cell = (i == n_classes or j == n_classes)
                is_diag       = (i == j) and (i < n_classes)
                is_grand      = (i == n_classes and j == n_classes)

                # --- Format the text ---
                if panel_type == "raw":
                    label = f"{int(val):,}"
                    txt_color = "black" if is_total_cell else (
                        "white" if float(val) > thresh else "black"
                    )

                else:  # normalized panel
                    if np.isnan(val):
                        continue

                    if is_grand:
                        # Grand total cell: overall accuracy
                        label = f"{val:.2%}"
                        txt_color = "#1B3A6B"
                    elif i == n_classes:
                        # Precision row
                        label = f"{val:.2%}"
                        txt_color = "#2E8B57" if val >= 0.90 else (
                            "#DD5522" if val >= 0.70 else "#B22222"
                        )
                    elif j == n_classes:
                        # Recall column
                        label = f"{val:.2%}"
                        txt_color = "#2E8B57" if val >= 0.90 else (
                            "#DD5522" if val >= 0.70 else "#B22222"
                        )
                    else:
                        # Core cells
                        txt_color = "white" if float(val) > thresh else "black"
                        if is_diag:
                            label = f"{val:.2f}\n({int(cm[i, j]):,})"
                        else:
                            label = f"{val:.2f}"

                # --- Weight ---
                if is_diag or is_grand:
                    weight = "bold"
                elif is_total_cell:
                    weight = "semibold"
                else:
                    weight = "normal"

                fontsize = 9 if not is_total_cell else 8.5

                ax.text(j, i, label,
                        ha="center", va="center",
                        fontsize=fontsize, color=txt_color,
                        fontweight=weight, zorder=4)

        # ── Axes ticks & labels ───────────────────────────────────────────────
        ax.set_xticks(range(n_ext))
        ax.set_yticks(range(n_ext))
        ax.set_xticklabels(extended_names, rotation=40, ha="right", fontsize=10)
        ax.set_yticklabels(extended_names, fontsize=10)

        ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
        ax.set_ylabel("True Label",      fontsize=11, labelpad=10)

        # ── Marginal labels for the totals ────────────────────────────────────
        if panel_type == "norm":
            # Label the totals margins
            ax.text(n_classes, -0.85, "Recall", ha="center", va="center",
                    fontsize=8, fontstyle="italic", color="gray")
            ax.text(-0.05, n_classes, "Precision", ha="right", va="center",
                    fontsize=8, fontstyle="italic", color="gray",
                    transform=ax.get_yaxis_transform())

        # ── Diagonal highlight box (core region only) ─────────────────────────
        for k in range(n_classes):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor="#DD5522",
                linewidth=1.5, zorder=5
            ))

        # ── Separator lines between core and totals ──────────────────────────
        ax.axhline(n_classes - 0.5, color="#888888", linewidth=1.5, zorder=5)
        ax.axvline(n_classes - 0.5, color="#888888", linewidth=1.5, zorder=5)

        # ── Per-panel title ───────────────────────────────────────────────────
        ax.set_title(f"Confusion Matrix — {subtitle}",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_facecolor("#F8F9FA")
        ax.spines[:].set_visible(False)

    # ── Per-class accuracy bar (bottom annotation strip) ──────────────────────
    per_class_acc = np.diag(cm_norm)
    for ax in axes:
        ax2 = ax.inset_axes([0, -0.45, 1, 0.14])   # strip below main plot
        colors = ["#2E8B57" if a >= 0.90 else
                  "#DD5522" if a >= 0.70 else
                  "#B22222" for a in per_class_acc]
        ax2.bar(range(n_classes), per_class_acc, color=colors,
                width=0.6, edgecolor="white", linewidth=0.6)
        ax2.set_xlim(-0.5, n_classes - 0.5)
        ax2.set_ylim(0, 1.15)
        ax2.set_xticks(range(n_classes))
        ax2.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
        ax2.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.set_ylabel("Recall", fontsize=8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.tick_params(labelsize=7)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_facecolor("#F8F9FA")
        for i, acc in enumerate(per_class_acc):
            ax2.text(i, acc + 0.03, f"{acc:.0%}",
                     ha="center", va="bottom", fontsize=7, color="dimgray")

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Maximum Likelihood Classifier — Confusion Matrix\n"
        f"Overall Accuracy: {overall_acc:.2%}  |  "
        f"N samples: {int(cm.sum()):,}",
        fontsize=13, fontweight="bold", y=1.01
    )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_confusion_matrix] Figure saved → {save_path}")

    plt.show()


def plot_per_class_confusion(y_test, y_pred, class_labels,
                             ncols=4, save_path="mlc_per_class_confusion.png"):
    """
    Plot a per-class binary confusion matrix (One-vs-Rest) for each class
    in a multi-class classifier, with precision, recall, and F1 annotations.

    Parameters
    ----------
    y_test       : array-like — true class labels (integer encoded)
    y_pred       : array-like — predicted class labels (integer encoded)
    class_labels : dict — mapping of class name → integer label
                   e.g. {"water": 0, "cloud": 1, ...}
    ncols        : int — number of columns in the subplot grid (default: 4)
    save_path    : str or None — file path to save the figure;
                   pass None to skip saving (default: "mlc_per_class_confusion.png")
    """

    # ── Layout ────────────────────────────────────────────────────────────────
    class_names  = list(class_labels.keys())
    label_values = list(class_labels.values())
    n_classes    = len(class_names)

    nrows  = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.5 * nrows), dpi=150)
    axes = axes.ravel()
    fig.patch.set_facecolor("#F8F9FA")

    # ── Custom colormap ───────────────────────────────────────────────────────
    cmap = LinearSegmentedColormap.from_list(
        "custom_blues", ["#FFFFFF", "#C6D9F0", "#4C72B0", "#1B3A6B"]
    )

    # ── Metric color thresholds ───────────────────────────────────────────────
    def metric_color(val):
        if val >= 0.90: return "#2E8B57"   # green
        if val >= 0.70: return "#DD5522"   # orange
        return "#B22222"                   # red

    # ── Per-class panels ──────────────────────────────────────────────────────
    for i, (label_int, class_name) in enumerate(zip(label_values, class_names)):

        # Binary conversion: this class (1) vs all others (0)
        y_true_bin = (y_test == label_int).astype(int)
        y_pred_bin = (y_pred == label_int).astype(int)

        cm_bin     = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        TN, FP     = cm_bin[0, 0], cm_bin[0, 1]
        FN, TP     = cm_bin[1, 0], cm_bin[1, 1]

        # Display matrix: rows = True, cols = Predicted
        display_cm = np.array([[TP, FN],
                                [FP, TN]])

        # ── Metrics ───────────────────────────────────────────────────────────
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        ax = axes[i]
        ax.set_facecolor("#F8F9FA")

        # ── Heatmap ───────────────────────────────────────────────────────────
        im = ax.imshow(display_cm, cmap=cmap, aspect="auto",
                       vmin=0, vmax=display_cm.max())

        # ── Cell labels ───────────────────────────────────────────────────────
        cell_tags = [["TP", "FN"],
                     ["FP", "TN"]]

        # Color corners: TP/TN = good (blue tones), FP/FN = bad (warm)
        cell_bg   = [["#4C72B0", "#DD5522"],
                     ["#DD5522", "#4C72B0"]]

        for row in range(2):
            for col in range(2):
                val        = display_cm[row, col]
                tag        = cell_tags[row][col]
                txt_color  = "white" if val > display_cm.max() * 0.45 else "black"

                # Tag badge
                ax.text(col, row - 0.22,
                        tag,
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=cell_bg[row][col],
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", alpha=0.6,
                                  edgecolor=cell_bg[row][col], linewidth=1.0))

                # Count value
                ax.text(col, row + 0.18,
                        f"{val:,}",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color=txt_color)

        # ── Diagonal highlight ────────────────────────────────────────────────
        for k in range(2):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor="#DD5522",
                linewidth=2.0, zorder=3
            ))

        # ── Axes ticks ────────────────────────────────────────────────────────
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nPositive", "Predicted\nNegative"],
                           fontsize=9)
        ax.set_yticklabels(["True\nPositive", "True\nNegative"],
                           fontsize=9, rotation=0)
        ax.spines[:].set_visible(False)

        # ── Class title ───────────────────────────────────────────────────────
        ax.set_title(class_name.upper(), fontsize=13,
                     fontweight="bold", pad=10, color="#1B3A6B")

        # ── Metric strip below each panel ─────────────────────────────────────
        metrics     = [("Precision", precision),
                       ("Recall",    recall),
                       ("F1",        f1),
                       ("Spec.",     specificity)]

        ax_strip = ax.inset_axes([0, -0.28, 1, 0.18])
        ax_strip.set_facecolor("#F8F9FA")
        ax_strip.set_xlim(0, len(metrics))
        ax_strip.set_ylim(0, 1)
        ax_strip.axis("off")

        for m_idx, (m_name, m_val) in enumerate(metrics):
            color = metric_color(m_val)
            ax_strip.text(m_idx + 0.5, 0.72, m_name,
                          ha="center", va="center",
                          fontsize=8, color="dimgray")
            ax_strip.text(m_idx + 0.5, 0.28, f"{m_val:.2f}",
                          ha="center", va="center",
                          fontsize=11, fontweight="bold", color=color)

        # Divider line
        ax_strip.axhline(0.5, color="lightgray", linewidth=0.8)

    # ── Hide unused subplots ──────────────────────────────────────────────────
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color="#2E8B57", label="≥ 0.90  Excellent"),
        mpatches.Patch(color="#DD5522", label="≥ 0.70  Acceptable"),
        mpatches.Patch(color="#B22222", label="< 0.70  Poor"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3,
        fontsize=9, framealpha=0.9, edgecolor="lightgray",
        title="Metric thresholds", title_fontsize=9,
        bbox_to_anchor=(0.5, -0.01)
    )

    # ── Main title ────────────────────────────────────────────────────────────
    overall_acc = np.sum(y_test == y_pred) / len(y_test)
    fig.suptitle(
        f"Per-Class Binary Confusion Matrix — Maximum Likelihood Classifier\n"
        f"Overall Accuracy: {overall_acc:.2%}  |  N samples: {len(y_test):,}",
        fontsize=13, fontweight="bold", y=1.01
    )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_per_class_confusion] Figure saved → {save_path}")

    plt.show()


def viirs_day_snow_fog_rgb(M10, M7, M5):
    """
    Create VIIRS Day Snow-Fog RGB

    Parameters
    ----------
    M10 : SWIR (1.61 µm)
    M7  : NIR  (0.865 µm)
    M5  : Red  (0.672 µm)
    """

    import numpy as np

    # Stack channels
    rgb = np.stack([M10, M7, M5], axis=-1)

    # Normalize each channel (simple stretch)
    def normalize(ch, vmin=None, vmax=None):
        if vmin is None or vmax is None:
            vmin, vmax = np.nanpercentile(ch, [2, 98])
        ch = (ch - vmin) / (vmax - vmin)
        return np.clip(ch, 0, 1)

    rgb_norm = np.zeros_like(rgb)
    for i in range(3):
        rgb_norm[:, :, i] = normalize(rgb[:, :, i])

    return rgb_norm

def build_spectral_statistics_table(
    X_raw,              # np.ndarray (n_pixels, n_bands) — RAW unscaled values
    y,                  # np.ndarray (n_samples,)        — integer class labels
    all_band_names,     # list of ALL band names matching columns of X_raw
    class_labels,       # dict {'water': 0, 'cloud': 1, ...}
    band_wavelengths=BAND_WAVELENGTHS,
    stats_to_compute=('mean', 'std', 'min', 'max', 'median',),
    save_dir=None,      # pathlib.Path or None
    float_fmt='.4f',
):
    """
    Build a multi-level statistics table per class per band.

    Returns
    -------
    df_wide   : pd.DataFrame  — wide format  (classes × [band × stat])
    df_long   : pd.DataFrame  — long format  (one row per class-band-stat)
    """

    all_band_names = list(all_band_names)
    records        = []

    for class_name, label_int in class_labels.items():
        mask    = (y == label_int)
        X_class = X_raw[mask]
        n       = len(X_class)

        if n == 0:
            continue

        for b_idx, band in enumerate(all_band_names):
            vals = X_class[:, b_idx].astype(float)
            wl   = band_wavelengths.get(band, np.nan)

            row = {
                'class'      : class_name,
                'band'       : band,
                'wavelength' : wl,
                'n_samples'  : n,
            }

            if 'mean'     in stats_to_compute:
                row['mean']     = np.nanmean(vals)
            if 'std'      in stats_to_compute:
                row['std']      = np.nanstd(vals, ddof=1)
            if 'min'      in stats_to_compute:
                row['min']      = np.nanmin(vals)
            if 'max'      in stats_to_compute:
                row['max']      = np.nanmax(vals)
            if 'median'   in stats_to_compute:
                row['median']   = np.nanmedian(vals)
            if 'cv'       in stats_to_compute:
                mu = np.nanmean(vals)
                row['cv']       = (np.nanstd(vals, ddof=1) / mu * 100
                                   if mu != 0 else np.nan)   # coefficient of variation (%)
            if 'skewness' in stats_to_compute:
                row['skewness'] = skew(vals, nan_policy='omit')
            if 'kurtosis' in stats_to_compute:
                row['kurtosis'] = kurtosis(vals, nan_policy='omit')  # excess kurtosis

            records.append(row)

    # ── Long format ───────────────────────────────────────────────────────
    df_long = pd.DataFrame(records)
    df_long = df_long.sort_values(['class', 'wavelength']).reset_index(drop=True)

    # ── Wide format  (rows = class, columns = MultiIndex [band, stat]) ────
    stat_cols  = [c for c in df_long.columns
                  if c not in ('class', 'band', 'wavelength', 'n_samples')]
    df_wide    = (df_long
                  .pivot_table(index='class',
                               columns='band',
                               values=stat_cols,
                               aggfunc='first')
                  .swaplevel(axis=1)            # band → outer, stat → inner
                  .sort_index(axis=1, level=0)) # sort by band name

    # ── Optional save ────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # CSV (long — easiest to query)
        df_long.to_csv(save_dir / 'spectral_stats_long.csv',
                       index=False, float_format=f'%{float_fmt}')

        # Excel (wide — human-readable pivot table)
        with pd.ExcelWriter(save_dir / 'spectral_stats_wide.xlsx',
                            engine='openpyxl') as writer:
            df_wide.to_excel(writer, sheet_name='Wide')
            df_long.to_excel(writer, sheet_name='Long', index=False)

        # LaTeX (long — for publications)
        latex_str = df_long.to_latex(
            index=False,
            float_format=f'{{:{float_fmt}}}'.format,
            caption='Per-class spectral statistics across VIIRS bands.',
            label='tab:spectral_stats',
        )
        (save_dir / 'spectral_stats.tex').write_text(latex_str)

        print(f"[Stats] Tables saved to: {save_dir}")

    return df_wide, df_long


def plot_pca_discrete(
    pc_data, height, width,
    n_levels=4,
    figsize=(10, 8),
    dpi=150,
    # ── NEW: geo-labeling + title ─────────────────────────────────
    lon=None,
    lat=None,
    title=None,
    save_path=None,
    show=True,
):
    """
    Visualize a single PCA component as a discretized (classified) map
    with optional geographic axis labels.

    Parameters
    ----------
    pc_data    : np.ndarray (n_pixels,) — one PCA component (flat)
    height     : int — image height
    width      : int — image width
    n_levels   : int — number of discrete bins (2..12 recommended)
    lon        : np.ndarray (height, width) or None — longitude grid
    lat        : np.ndarray (height, width) or None — latitude grid
    title      : str or None — figure title
    save_path  : str/Path or None — save figure to this path
    show       : bool — call plt.show()

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    pc_2d = pc_data.reshape(height, width)

    # ── Color logic ───────────────────────────────────────────────────────────
    if n_levels == 2:
        colors = ['black', 'white']
    else:
        cmap_base = plt.get_cmap('tab10')
        colors = [cmap_base(i) for i in range(n_levels)]

    custom_cmap = mcolors.ListedColormap(colors)

    # ── Boundaries ────────────────────────────────────────────────────────────
    bounds = np.linspace(pc_2d.min(), pc_2d.max(), n_levels + 1)
    norm   = mcolors.BoundaryNorm(bounds, n_levels)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(pc_2d, cmap=custom_cmap, norm=norm,
                   interpolation='nearest', origin='upper')

    # ── Discrete colorbar ─────────────────────────────────────────────────────
    tick_locs = (bounds[:-1] + bounds[1:]) / 2
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs,
                        fraction=0.046, pad=0.04)
    cbar.set_label('Variance Level')
    cbar.ax.set_yticklabels([f'Level {i}' for i in range(n_levels)])

    # ── Geo ticks or pixel-only ───────────────────────────────────────────────
    if lon is not None and lat is not None:
        add_geo_ticks(ax, lon, lat, height=height, width=width)
    else:
        ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────────────
    default_title = f'VIIRS PCA Discretized into {n_levels} Levels'
    ax.set_title(title if title else default_title,
                 fontsize=13, fontweight='bold', pad=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[plot_pca_discrete] Saved → {save_path}")

    if show:
        plt.show()

    return fig, ax


def calculate_accuracy_metrics(y_test, y_pred, class_labels, latex_path=None):
    """
    Calculate Producer's Accuracy (Recall) and User's Accuracy (Precision)
    per class from a confusion matrix, plus Overall Accuracy and Kappa.

    Parameters
    ----------
    y_test       : array-like — true class labels (integer encoded)
    y_pred       : array-like — predicted class labels (integer encoded)
    class_labels : dict — mapping of class name → integer label
                   e.g. {"water": 0, "cloud": 1, ...}

    Returns
    -------
    df_metrics : pd.DataFrame — per-class and overall accuracy metrics
    cm         : np.ndarray   — the raw confusion matrix (n × n)
    """

    class_names  = list(class_labels.keys())
    label_values = list(class_labels.values())
    n_classes    = len(class_names)

    # ── Build confusion matrix ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=label_values)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    diag       = np.diag(cm).astype(float)          # correctly classified
    row_totals = cm.sum(axis=1).astype(float)       # actual totals  (row sums)
    col_totals = cm.sum(axis=0).astype(float)       # predicted totals (col sums)

    # Producer's Accuracy = diagonal / row total  (Recall)
    #   "Of everything truly in this class, what fraction was correctly mapped?"
    producers_acc = np.where(row_totals > 0, diag / row_totals, 0.0)

    # User's Accuracy = diagonal / column total  (Precision)
    #   "Of everything predicted as this class, what fraction is actually correct?"
    users_acc = np.where(col_totals > 0, diag / col_totals, 0.0)

    # Errors of Omission  = 1 - Producer's Accuracy  (missed true positives)
    omission_error = 1.0 - producers_acc

    # Errors of Commission = 1 - User's Accuracy  (false positives included)
    commission_error = 1.0 - users_acc

    # F1 Score (harmonic mean of precision & recall)
    f1 = np.where(
        (producers_acc + users_acc) > 0,
        2 * (users_acc * producers_acc) / (users_acc + producers_acc),
        0.0
    )

    # ── Overall Accuracy ──────────────────────────────────────────────────────
    N = cm.sum()
    overall_acc = np.trace(cm) / N if N > 0 else 0.0

    # ── Cohen's Kappa ─────────────────────────────────────────────────────────
    #    Measures agreement beyond chance
    expected = (row_totals * col_totals) / N if N > 0 else np.zeros(n_classes)
    expected_acc = expected.sum() / N if N > 0 else 0.0
    kappa = (overall_acc - expected_acc) / (1 - expected_acc) if expected_acc < 1 else 0.0

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df_metrics = pd.DataFrame({
        "Class":               class_names,
        "True Count":          row_totals.astype(int),
        "Predicted Count":     col_totals.astype(int),
        "Correct":             diag.astype(int),
        "Producer's Acc (%)":  np.round(producers_acc * 100, 2),
        "User's Acc (%)":      np.round(users_acc * 100, 2),
        "Omission Err (%)":    np.round(omission_error * 100, 2),
        "Commission Err (%)":  np.round(commission_error * 100, 2),
        "F1 Score":            np.round(f1, 4),
    })

    # ── Summary row ───────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "Class":               ["** OVERALL **"],
        "True Count":          [int(N)],
        "Predicted Count":     [int(N)],
        "Correct":             [int(np.trace(cm))],
        "Producer's Acc (%)":  [round(overall_acc * 100, 2)],
        "User's Acc (%)":      [round(overall_acc * 100, 2)],
        "Omission Err (%)":    ["—"],
        "Commission Err (%)":  ["—"],
        "F1 Score":            ["—"],
    })

    df_metrics = pd.concat([df_metrics, summary], ignore_index=True)

    # ── Print report ──────────────────────────────────────────────────────────
    print("=" * 90)
    print("  ACCURACY ASSESSMENT REPORT")
    print("=" * 90)
    print(f"\n  Overall Accuracy : {overall_acc:.2%}")
    print(f"  Cohen's Kappa    : {kappa:.4f}")
    print(f"  Total Samples    : {int(N):,}\n")
    print("-" * 90)
    print(df_metrics.to_string(index=False))
    print("-" * 90)

    print("\n  KEY:")
    print("  • Producer's Accuracy (Recall)    — 'Did I find everything that was there?'")
    print("  • User's Accuracy     (Precision) — 'Can I trust what the map says?'")
    print("  • Omission Error      = 1 - Producer's Acc  (true class missed)")
    print("  • Commission Error    = 1 - User's Acc      (wrong class included)")
    print("  • Cohen's Kappa       — Agreement beyond chance (>0.80 = strong)\n")
    # ── LaTeX export ──────────────────────────────────────────────────────────
    if latex_path is not None:
        _save_accuracy_metrics_latex_table(df_metrics, overall_acc, kappa, int(N),
                          n_classes, latex_path)

    return df_metrics, cm



def _save_accuracy_metrics_latex_table(df_metrics, overall_acc, kappa, N,
                      n_classes, latex_path):
    """
    Write a publication-ready LaTeX table of the accuracy assessment.

    The output is a standalone table (not a full document) ready to be
    \\input{} into any .tex manuscript.
    """

    # ── Work on per-class rows only (exclude the OVERALL summary row) ─────────
    df_classes = df_metrics.iloc[:n_classes].copy()

    # ── Build LaTeX string ────────────────────────────────────────────────────
    lines = []

    # Preamble
    lines.append(r"% ── Auto-generated by calculate_accuracy_metrics() ──")
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Maximum Likelihood Classification — "
                 r"Accuracy Assessment}")
    lines.append(r"  \label{tab:mlc_accuracy}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l r r r r r r r r}")
    lines.append(r"    \toprule")

    # Header
    lines.append(
        r"    \textbf{Class} & \textbf{True} & \textbf{Pred.} & "
        r"\textbf{Correct} & \textbf{PA (\%)} & \textbf{UA (\%)} & "
        r"\textbf{Omis. (\%)} & \textbf{Comm. (\%)} & "
        r"\textbf{F1} \\"
    )
    lines.append(r"    \midrule")

    # Per-class rows
    for _, row in df_classes.iterrows():
        name   = row["Class"].replace("_", r"\_")
        true_c = int(row["True Count"])
        pred_c = int(row["Predicted Count"])
        corr   = int(row["Correct"])
        pa     = row["Producer's Acc (%)"]
        ua     = row["User's Acc (%)"]
        om     = row["Omission Err (%)"]
        co     = row["Commission Err (%)"]
        f1     = row["F1 Score"]

        lines.append(
            f"    {name} & {true_c:,} & {pred_c:,} & {corr:,} & "
            f"{pa:.2f} & {ua:.2f} & {om:.2f} & {co:.2f} & {f1:.4f} \\\\"
        )

    # Summary rows
    lines.append(r"    \midrule")
    lines.append(
        f"    \\textbf{{Overall}} & "
        f"\\multicolumn{{2}}{{c}}{{{N:,} samples}} & "
        f"{int(df_metrics.iloc[-1]['Correct']):,} & "
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{overall_acc * 100:.2f}\\%}}}} & "
        f"\\multicolumn{{3}}{{c}}"
        f"{{$\\kappa = {kappa:.4f}$}} \\\\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    # Footnote
    lines.append(r"  \vspace{4pt}")
    lines.append(r"  \begin{minipage}{0.92\textwidth}")
    lines.append(r"    \footnotesize")
    lines.append(r"    PA = Producer's Accuracy (Recall); "
                 r"UA = User's Accuracy (Precision); \\")
    lines.append(r"    Omis. = Omission Error ($1 - \text{PA}$); "
                 r"Comm. = Commission Error ($1 - \text{UA}$); \\")
    lines.append(r"    $\kappa$ = Cohen's Kappa "
                 r"($> 0.80$ indicates strong agreement).")
    lines.append(r"  \end{minipage}")

    lines.append(r"\end{table}")

    # ── Write file ────────────────────────────────────────────────────────────
    latex_str = "\n".join(lines) + "\n"

    with open(latex_path, "w") as f:
        f.write(latex_str)

    print(f"[calculate_accuracy_metrics] LaTeX table saved → {latex_path}")

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

def pca_eigen_table(pca, feature_names=None, latex_path=None, top_n=None):
    """
    Build a publication-ready table of PCA eigenvalues, explained variance,
    and eigenvectors (loadings) from a fitted sklearn PCA object.

    Parameters
    ----------
    pca           : sklearn.decomposition.PCA — a fitted PCA object
    feature_names : list of str or None — names of the original features
                    (e.g. band names). If None, defaults to
                    ["Feature_1", "Feature_2", ...].
    latex_path    : str or None — file path to save a LaTeX table;
                    pass None to skip (default: None)
    top_n         : int or None — if set, only include the first n
                    principal components (default: all)

    Returns
    -------
    df_eigen  : pd.DataFrame — eigenvalue / variance summary table
    df_load   : pd.DataFrame — eigenvector (loading) matrix
    """

    # ── Validate ──────────────────────────────────────────────────────────────
    if not hasattr(pca, "components_"):
        raise ValueError("PCA object has not been fitted yet. "
                         "Call pca.fit() or pca.fit_transform() first.")

    n_components, n_features = pca.components_.shape

    if feature_names is None:
        feature_names = [f"Feature_{i + 1}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length ({len(feature_names)}) "
                         f"!= number of features ({n_features})")

    if top_n is not None:
        top_n = min(top_n, n_components)
    else:
        top_n = n_components

    # ── Eigenvalue / variance summary ─────────────────────────────────────────
    #    sklearn stores explained_variance_ (eigenvalues of covariance matrix)
    #    and explained_variance_ratio_ (proportion of total variance)
    eigenvalues = pca.explained_variance_[:top_n]
    explained_ratio = pca.explained_variance_ratio_[:top_n]
    cumulative_ratio = np.cumsum(explained_ratio)

    pc_labels = [f"PC{i + 1}" for i in range(top_n)]

    df_eigen = pd.DataFrame({
        "Component": pc_labels,
        "Eigenvalue": np.round(eigenvalues, 6),
        "Explained Var. (%)": np.round(explained_ratio * 100, 4),
        "Cumulative Var. (%)": np.round(cumulative_ratio * 100, 4),
    })

    # ── Eigenvector (loading) matrix ──────────────────────────────────────────
    #    pca.components_ shape: (n_components, n_features)
    #    Each row is an eigenvector (loadings for one PC)
    loadings = pca.components_[:top_n, :]

    df_load = pd.DataFrame(
        np.round(loadings, 6),
        index=pc_labels,
        columns=feature_names,
    )
    df_load.index.name = "Component"

    # ── Print report ──────────────────────────────────────────────────────────
    print("=" * 80)
    print("  PCA EIGENVALUE / VARIANCE SUMMARY")
    print("=" * 80)
    print(df_eigen.to_string(index=False))
    print("-" * 80)

    print(f"\n  Total variance explained by {top_n} component(s): "
          f"{cumulative_ratio[-1] * 100:.2f}%")
    print(f"  Total features: {n_features}")
    print(f"  Components shown: {top_n} of {n_components}\n")

    print("=" * 80)
    print("  EIGENVECTOR (LOADING) MATRIX")
    print("  Rows = Principal Components  |  Columns = Original Features")
    print("=" * 80)
    print(df_load.to_string())
    print("-" * 80)

    # Highlight dominant loadings per PC
    print("\n  DOMINANT LOADINGS (|loading| ≥ 0.30) per component:\n")
    for i, pc in enumerate(pc_labels):
        row = loadings[i]
        dominant_idx = np.where(np.abs(row) >= 0.30)[0]
        if len(dominant_idx) > 0:
            parts = [f"{feature_names[j]} ({row[j]:+.4f})"
                     for j in dominant_idx[np.argsort(-np.abs(row[dominant_idx]))]]
            print(f"    {pc}: {', '.join(parts)}")
        else:
            print(f"    {pc}: (no loadings ≥ 0.30)")
    print()

    # ── LaTeX export ──────────────────────────────────────────────────────────
    if latex_path is not None:
        _save_pca_latex(df_eigen, df_load, pc_labels, feature_names,
                        cumulative_ratio, top_n, n_components,
                        n_features, latex_path)

    return df_eigen, df_load

def _save_pca_latex(df_eigen, df_load, pc_labels, feature_names,
                    cumulative_ratio, top_n, n_components,
                    n_features, latex_path):
    """
    Write two publication-ready LaTeX tables (eigenvalue summary +
    loading matrix) to a single .tex file.
    """

    lines = []

    # ── Preamble ──────────────────────────────────────────────────────────────
    lines.append(r"% ── Auto-generated by pca_eigen_table() ──")
    lines.append(r"% Requires: \usepackage{booktabs, adjustbox}")
    lines.append(r"")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 1 — Eigenvalue summary
    # ══════════════════════════════════════════════════════════════════════════
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{PCA Eigenvalue and Explained Variance Summary}")
    lines.append(r"  \label{tab:pca_eigenvalues}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Component} & \textbf{Eigenvalue} & "
                 r"\textbf{Explained (\%)} & \textbf{Cumulative (\%)} \\")
    lines.append(r"    \midrule")

    for _, row in df_eigen.iterrows():
        lines.append(
            f"    {row['Component']} & "
            f"{row['Eigenvalue']:.6f} & "
            f"{row['Explained Var. (%)']:.4f} & "
            f"{row['Cumulative Var. (%)']:.4f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 2 — Loading matrix
    # ══════════════════════════════════════════════════════════════════════════
    n_feat = len(feature_names)
    col_spec = "l " + " ".join(["r"] * n_feat)

    # Escape underscores in feature names
    feat_escaped = [f.replace("_", r"\_") for f in feature_names]

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{PCA Eigenvector (Loading) Matrix}")
    lines.append(r"  \label{tab:pca_loadings}")
    lines.append(r"  \small")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Header row
    header = r"    \textbf{Component}"
    for feat in feat_escaped:
        header += f" & \\textbf{{{feat}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    # Data rows
    for pc in pc_labels:
        row_vals = df_load.loc[pc].values
        row_str = f"    {pc}"
        for v in row_vals:
            # Bold dominant loadings (|v| >= 0.30)
            if abs(v) >= 0.30:
                row_str += f" & \\textbf{{{v:+.4f}}}"
            else:
                row_str += f" & {v:+.4f}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"    \bottomrule")
    lines.append(f"  \\end{{tabular}}")
    lines.append(r"  \end{adjustbox}")
    lines.append(r"  \vspace{4pt}")
    lines.append(r"  \begin{minipage}{0.92\textwidth}")
    lines.append(r"    \footnotesize")
    lines.append(r"    Bold values indicate dominant loadings "
                 r"($|\text{loading}| \geq 0.30$).")
    lines.append(r"  \end{minipage}")
    lines.append(r"\end{table}")

    # ── Write file ────────────────────────────────────────────────────────────
    latex_str = "\n".join(lines) + "\n"

    with open(latex_path, "w") as f:
        f.write(latex_str)

    print(f"[pca_eigen_table] LaTeX tables saved → {latex_path}")



def plot_zenith_angles(ds, save_path=None, figsize=(20, 8), dpi=150):
    """
    Plot solar zenith angle and satellite (viewing) zenith angle
    side-by-side from a VIIRS xarray Dataset using imshow.

    Parameters
    ----------
    ds        : xarray.Dataset — must contain 'solar_zenith_angle',
                'satellite_zenith_angle', 'latitude', 'longitude'
    save_path : str or None — file path to save figure (default: None)
    figsize   : tuple — figure size (default: (20, 8))
    dpi       : int — resolution (default: 150)

    """

    # ── Extract data ──────────────────────────────────────────────────────────
    sza = ds["solar_zenith_angle"].values       # Solar Zenith Angle
    vza = ds["satellite_zenith_angle"].values   # Viewing (Satellite) Zenith Angle
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # ── Geographic extent for axis labels ─────────────────────────────────────
    extent = [np.nanmin(lon), np.nanmax(lon),
              np.nanmin(lat), np.nanmax(lat)]

    # ── Colormaps ─────────────────────────────────────────────────────────────
    sza_cmap = LinearSegmentedColormap.from_list(
        "sza_cmap", ["#FFF8DC", "#FFD700", "#FF8C00", "#B22222", "#4B0082"]
    )
    vza_cmap = LinearSegmentedColormap.from_list(
        "vza_cmap", ["#F0F8FF", "#87CEEB", "#4682B4", "#1B3A6B", "#0D1B2A"]
    )

    # ── Panel configuration ───────────────────────────────────────────────────
    panels = [
        {
            "data":  sza,
            "title": "Solar Zenith Angle (SZA)",
            "cmap":  sza_cmap,
            "label": "Solar Zenith Angle (°)",
        },
        {
            "data":  vza,
            "title": "Satellite Zenith Angle (VZA)",
            "cmap":  vza_cmap,
            "label": "Satellite Zenith Angle (°)",
        },
    ]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, panel in zip(axes, panels):
        data = panel["data"]

        im = ax.imshow(
            data,
            cmap=panel["cmap"],
            vmin=np.nanmin(data),
            vmax=np.nanmax(data),
            aspect="auto",
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )

        # ── Colorbar ─────────────────────────────────────────────────────────
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(panel["label"], fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.1f}°")
        )

        # ── Axis labels ──────────────────────────────────────────────────────
        ax.set_xlabel("Longitude (°)", fontsize=10, labelpad=8)
        ax.set_ylabel("Latitude (°)",  fontsize=10, labelpad=8)
        ax.tick_params(labelsize=9)

        # ── Title ────────────────────────────────────────────────────────────
        ax.set_title(panel["title"], fontsize=12, fontweight="bold", pad=12)

        # ── Stats annotation ─────────────────────────────────────────────────
        stats_str = (f"Min: {np.nanmin(data):.2f}°  |  "
                     f"Max: {np.nanmax(data):.2f}°  |  "
                     f"Mean: {np.nanmean(data):.2f}°")
        ax.text(0.5, -0.10, stats_str,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="dimgray", fontstyle="italic")

        ax.set_facecolor("#F8F9FA")

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "VIIRS — Zenith Angle Geometry",
        fontsize=14, fontweight="bold", y=1.02
    )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[plot_zenith_angles] Figure saved → {save_path}")

    plt.show()


def chi2_accuracy_report(y_test, y_pred_chi2, class_labels,
                         n_classes, confidence=None, chi2_threshold=None,
                         latex_path=None):
    """
    Build a confusion matrix and accuracy report that includes
    the Unclassified column from chi-squared thresholding.

    The matrix has:
      - Rows    = true classes (n_classes)
      - Columns = predicted classes (n_classes) + Unclassified column

    Parameters
    ----------
    y_test         : array-like — true labels (0 to n_classes-1)
    y_pred_chi2    : array-like — predicted labels (0 to n_classes-1,
                     plus n_classes = Unclassified)
    class_labels   : dict — {"Water": 0, ...}
    n_classes      : int — number of real classes (excluding Unclassified)
    confidence     : float or None — for annotation
    chi2_threshold : float or None — for annotation
    latex_path     : str or None — save LaTeX table

    Returns
    -------
    df_report : pd.DataFrame — per-class accuracy with rejection info
    cm_ext    : np.ndarray — extended confusion matrix (n_classes × n_classes+1)
    """

    class_names  = list(class_labels.keys())
    label_values = list(class_labels.values())

    # ── Extended confusion matrix: rows=true, cols=pred+unclassified ──────────
    all_pred_labels = label_values + [n_classes]  # add unclassified label
    cm_ext = confusion_matrix(y_test, y_pred_chi2,
                               labels=label_values + [n_classes])

    # cm_ext shape: (n_classes, n_classes+1) if no true "Unclassified"
    # But confusion_matrix makes it square, so we get (n+1, n+1).
    # The last ROW (true=Unclassified) will be all zeros — remove it.
    # Keep only the first n_classes rows.
    # Actually, if y_test has no label=n_classes, the last row is all zeros.
    # Let's just build it manually for clarity:

    cm_core = np.zeros((n_classes, n_classes + 1), dtype=int)
    for i, true_label in enumerate(label_values):
        for j, pred_label in enumerate(label_values + [n_classes]):
            cm_core[i, j] = np.sum((y_test == true_label) &
                                    (y_pred_chi2 == pred_label))

    # ── Metrics ───────────────────────────────────────────────────────────────
    diag        = np.array([cm_core[i, i] for i in range(n_classes)], dtype=float)
    row_totals  = cm_core.sum(axis=1).astype(float)          # all true samples
    col_class   = cm_core[:, :n_classes].sum(axis=0).astype(float)  # classified pred totals
    unclass_col = cm_core[:, n_classes].astype(float)         # unclassified per true class

    # Producer's Accuracy = correct / row total (includes unclassified losses)
    producers_acc = np.where(row_totals > 0, diag / row_totals, 0.0)

    # User's Accuracy = correct / column total (classified predictions only)
    users_acc = np.where(col_class > 0, diag / col_class, 0.0)

    # Rejection rate per class
    rejection_rate = np.where(row_totals > 0, unclass_col / row_totals, 0.0)

    # F1
    f1 = np.where(
        (producers_acc + users_acc) > 0,
        2 * (users_acc * producers_acc) / (users_acc + producers_acc),
        0.0
    )

    # ── Overall stats ─────────────────────────────────────────────────────────
    total_samples   = int(row_totals.sum())
    total_correct   = int(diag.sum())
    total_unclass   = int(unclass_col.sum())
    total_classified = total_samples - total_unclass

    overall_acc_all       = total_correct / total_samples if total_samples > 0 else 0
    overall_acc_classified = total_correct / total_classified if total_classified > 0 else 0

    # ── DataFrame ─────────────────────────────────────────────────────────────
    df_report = pd.DataFrame({
        "Class":               class_names,
        "True Count":          row_totals.astype(int),
        "Classified":          (row_totals - unclass_col).astype(int),
        "Unclassified":        unclass_col.astype(int),
        "Rejected (%)":        np.round(rejection_rate * 100, 2),
        "Correct":             diag.astype(int),
        "Producer's Acc (%)":  np.round(producers_acc * 100, 2),
        "User's Acc (%)":      np.round(users_acc * 100, 2),
        "F1 Score":            np.round(f1, 4),
    })

    # Summary
    summary = pd.DataFrame({
        "Class":               ["** OVERALL **"],
        "True Count":          [total_samples],
        "Classified":          [total_classified],
        "Unclassified":        [total_unclass],
        "Rejected (%)":        [round(total_unclass / total_samples * 100, 2)],
        "Correct":             [total_correct],
        "Producer's Acc (%)":  [round(overall_acc_all * 100, 2)],
        "User's Acc (%)":      [round(overall_acc_classified * 100, 2)],
        "F1 Score":            ["—"],
    })

    df_report = pd.concat([df_report, summary], ignore_index=True)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("=" * 95)
    print("  CHI-SQUARED MLC ACCURACY REPORT")
    print("=" * 95)
    if confidence is not None:
        print(f"  Confidence     : {confidence:.1%}")
        print(f"  χ² threshold   : {chi2_threshold:.4f}")
    print(f"\n  Overall Accuracy (all pixels)       : {overall_acc_all:.2%}")
    print(f"  Overall Accuracy (classified only)  : {overall_acc_classified:.2%}")
    print(f"  Total Rejected                      : {total_unclass:,} / "
          f"{total_samples:,} ({total_unclass/total_samples:.2%})")
    print()
    print("-" * 95)
    print(df_report.to_string(index=False))
    print("-" * 95)

    print("\n  KEY:")
    print("  • Producer's Acc — based on ALL true samples (penalized by rejection)")
    print("  • User's Acc     — based on CLASSIFIED predictions only")
    print("  • Rejected (%)   — fraction of true class sent to Unclassified")
    print(f"  • Overall (all)  = {total_correct}/{total_samples} — "
          "treats rejected as incorrect")
    print(f"  • Overall (classified) = {total_correct}/{total_classified} — "
          "ignores rejected pixels\n")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    if latex_path is not None:
        _save_chi2_latex(df_report, overall_acc_all, overall_acc_classified,
                         total_samples, total_classified, total_unclass,
                         n_classes, confidence, chi2_threshold, latex_path)

    return df_report, cm_core


def _save_chi2_latex(df_report, oa_all, oa_classified,
                     total, classified, unclass,
                     n_classes, confidence, chi2_thresh, latex_path):
    """Write chi-squared accuracy report as LaTeX table."""

    df_classes = df_report.iloc[:n_classes].copy()
    lines = []

    lines.append(r"% ── Auto-generated by chi2_accuracy_report() ──")
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{MLC Accuracy Assessment with "
                 r"$\chi^2$ Rejection Threshold}")
    lines.append(r"  \label{tab:chi2_accuracy}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l r r r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Class} & \textbf{True} & \textbf{Classif.} & "
        r"\textbf{Unclass.} & \textbf{Rej. (\%)} & \textbf{Correct} & "
        r"\textbf{PA (\%)} & \textbf{UA (\%)} & \textbf{F1} \\"
    )
    lines.append(r"    \midrule")

    for _, row in df_classes.iterrows():
        name    = row["Class"].replace("_", r"\_")
        true_c  = int(row["True Count"])
        classif = int(row["Classified"])
        unclass_c = int(row["Unclassified"])
        rej_pct = row["Rejected (%)"]
        correct = int(row["Correct"])
        pa      = row["Producer's Acc (%)"]
        ua      = row["User's Acc (%)"]
        f1      = row["F1 Score"]

        lines.append(
            f"    {name} & {true_c:,} & "
            f"{classif:,} & {unclass_c:,} & "
            f"{rej_pct:.2f} & {correct:,} & "
            f"{pa:.2f} & {ua:.2f} & {f1:.4f} \\\\"
        )

    lines.append(r"    \midrule")
    conf_str = f"{confidence:.1%}" if confidence else "—"
    chi2_str = f"{chi2_thresh:.2f}" if chi2_thresh else "—"
    rej_pct_total = unclass / total * 100
    lines.append(
        f"    \\textbf{{Overall}} & {total:,} & {classified:,} & "
        f"{unclass:,} & {rej_pct_total:.2f} & "
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{oa_all*100:.2f}\\%}}}} & "
        f"\\multicolumn{{2}}{{c}}"
        f"{{Classif. only: \\textbf{{{oa_classified*100:.2f}\\%}}}} \\\\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \vspace{4pt}")
    lines.append(r"  \begin{minipage}{0.92\textwidth}")
    lines.append(r"    \footnotesize")
    lines.append(f"    $\\chi^2$ confidence: {conf_str}, "
                 f"threshold: {chi2_str}, df = number of bands. \\\\")
    lines.append(r"    PA = Producer's Accuracy; UA = User's Accuracy; "
                 r"Rej. = Rejected to Unclassified.")
    lines.append(r"  \end{minipage}")
    lines.append(r"\end{table}")

    with open(latex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[chi2_accuracy_report] LaTeX saved → {latex_path}")

def plot_chi2_classification(ds, class_map_chi2, class_labels,
                              chi2_threshold=None, confidence=None,
                              n_unclassified=None,
                              save_path=None, figsize=(18, 7), dpi=150):
    """
    Plot True Color | Chi-squared Classification side-by-side.

    Parameters
    ----------
    ds              : xarray.Dataset — must contain 'true_color', 'longitude', 'latitude'
    class_map_chi2  : np.ndarray (height, width) — classified map with unclassified
    class_labels    : dict — {"water": 0, ...}
    chi2_threshold  : float or None — threshold value (for annotation)
    confidence      : float or None — confidence level (for annotation)
    n_unclassified  : int or None — count of unclassified pixels
    save_path       : str or None — file path to save
    figsize         : tuple — figure size (default: (18, 7))
    dpi             : int — resolution (default: 150)
    """

    class_names  = list(class_labels.keys())
    n_classes    = len(class_names)
    height, width = class_map_chi2.shape

    # ── Colors: original classes + unclassified (black) ───────────────────────
    class_colors = [
        '#2166ac',   # Water      - blue
        '#f0f0f0',   # Cloud      - light grey
        '#a6cee3',   # Snow       - icy cyan
        '#6a3d9a',   # Smoke      - dark purple
        '#d6a86b',   # Bare Soil  - tan
        '#4dac26',   # Vegetation - green
        '#e31a1c',   # Unclassified    - red
    ]
    all_names = class_names + ['Unclassified']
    n_total   = n_classes + 1

    custom_cmap = mcolors.ListedColormap(class_colors[:n_total])
    norm        = mcolors.BoundaryNorm(np.arange(n_total + 1) - 0.5, n_total)

    # ── True color ────────────────────────────────────────────────────────────
    img_full = get_enhanced_image(ds['true_color'])
    rgb_full = img_full.data.transpose('y', 'x', 'bands').values
    rgb_full = np.clip(rgb_full, 0, 1)

    # ── Lon / Lat ─────────────────────────────────────────────────────────────
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    def add_geo_ticks(ax, n_xticks=6, n_yticks=6):
        x_pixel_pos = np.linspace(0, width  - 1, n_xticks, dtype=int)
        y_pixel_pos = np.linspace(0, height - 1, n_yticks, dtype=int)
        x_lon_vals  = lon[height // 2, x_pixel_pos]
        y_lat_vals  = lat[y_pixel_pos, width  // 2]
        ax.set_xticks(x_pixel_pos)
        ax.set_xticklabels([f"{v:.1f}°E" for v in x_lon_vals], fontsize=8)
        ax.set_yticks(y_pixel_pos)
        ax.set_yticklabels([f"{v:.1f}°N" for v in y_lat_vals], fontsize=8)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")

    # ── Panel 1: True Color ───────────────────────────────────────────────────
    axes[0].imshow(rgb_full, origin='upper')
    add_geo_ticks(axes[0])
    axes[0].set_title('VIIRS True Color', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude',  fontsize=10)

    # ── Panel 2: Chi-squared Classification ───────────────────────────────────
    im = axes[1].imshow(class_map_chi2, cmap=custom_cmap, norm=norm,
                        interpolation='nearest', origin='upper')
    add_geo_ticks(axes[1])
    axes[1].set_title('MLC with χ² Threshold', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('Latitude',  fontsize=10)

    # ── Discrete colorbar ─────────────────────────────────────────────────────
    tick_locs = np.arange(n_total)
    cbar = fig.colorbar(im, ax=axes[1], ticks=tick_locs,
                        fraction=0.046, pad=0.04)
    cbar.set_label('Land Cover Class', fontsize=10)
    cbar.ax.set_yticklabels(all_names, fontsize=9)

    # ── Annotation ────────────────────────────────────────────────────────────
    if chi2_threshold is not None and confidence is not None:
        total = height * width
        unc   = n_unclassified if n_unclassified is not None else 0
        info = (f"χ² threshold: {chi2_threshold:.2f}  "
                f"(α = {1 - confidence:.3f})\n"
                f"Unclassified: {unc:,} / {total:,} "
                f"({unc / total:.2%})")
        axes[1].text(0.5, -0.12, info,
                     transform=axes[1].transAxes, ha='center', va='top',
                     fontsize=8, color='dimgray', fontstyle='italic')

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        'VIIRS Surface Classification — Maximum Likelihood with χ² Rejection',
        fontsize=14, fontweight='bold', y=1.02
    )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[plot_chi2_classification] Figure saved → {save_path}")

    plt.show()


def chi2_classify(X_scaled, mlc, class_labels, height, width,
                  confidence=0.95):
    """
    Apply a chi-squared threshold to an MLC classification to identify
    pixels that do not belong to any trained class ("Unclassified").

    Parameters
    ----------
    X_scaled     : np.ndarray, shape (n_pixels, n_bands) — scaled feature array
    mlc          : fitted sklearn classifier — must expose .theta_ (means)
                   and .var_ (variances) or be a GaussianNB / custom MLC
                   with class statistics
    class_labels : dict — {"water": 0, "cloud": 1, ...}
    height       : int — image height in pixels
    width        : int — image width in pixels
    confidence   : float — confidence level for chi-squared threshold
                   (default: 0.95)

    Returns
    -------
    class_map_chi2 : np.ndarray, shape (height, width) — classified map
                     with unclassified pixels = n_classes
    chi2_threshold : float — the critical chi-squared value used
    min_distances  : np.ndarray, shape (height, width) — minimum
                     Mahalanobis distance per pixel
    n_unclassified : int — count of unclassified pixels
    """

    class_names  = list(class_labels.keys())
    label_values = list(class_labels.values())
    n_classes    = len(class_names)
    n_pixels, n_bands = X_scaled.shape

    # ── Compute class statistics from training data ───────────────────────────
    # Get means and covariance matrices per class
    class_means = []
    class_cov_inv = []

    for k in label_values:
        # Extract training samples for class k
        mask_k = (mlc.predict(X_scaled) == k)  # or use stored training data
        X_k = X_scaled[mask_k]

        mean_k = np.mean(X_k, axis=0)
        cov_k  = np.cov(X_k, rowvar=False)

        # Regularize covariance to avoid singularity
        cov_k += np.eye(n_bands) * 1e-6

        class_means.append(mean_k)
        class_cov_inv.append(np.linalg.inv(cov_k))

    # ── Chi-squared critical value ────────────────────────────────────────────
    #    Degrees of freedom = number of bands
    chi2_threshold = chi2.ppf(confidence, df=n_bands)

    print(f"[chi2_classify] Chi-squared threshold:")
    print(f"  Confidence : {confidence:.1%}")
    print(f"  Bands (df) : {n_bands}")
    print(f"  χ² critical: {chi2_threshold:.4f}")

    # ── Compute Mahalanobis distance to every class for every pixel ───────────
    #    Shape: (n_pixels, n_classes)
    distances = np.zeros((n_pixels, n_classes))

    for k_idx in range(n_classes):
        diff = X_scaled - class_means[k_idx]                 # (n_pixels, n_bands)
        left = diff @ class_cov_inv[k_idx]                   # (n_pixels, n_bands)
        distances[:, k_idx] = np.sum(left * diff, axis=1)    # (n_pixels,)

    # ── Classify with threshold ───────────────────────────────────────────────
    min_distances   = np.min(distances, axis=1)          # closest class distance
    best_class      = np.argmin(distances, axis=1)       # MLC assignment

    # Pixels where even the closest class exceeds threshold → unclassified
    unclassified_mask = min_distances > chi2_threshold
    n_unclassified    = np.sum(unclassified_mask)

    # Unclassified label = n_classes (one beyond the last class)
    class_map_chi2 = best_class.copy()
    class_map_chi2[unclassified_mask] = n_classes

    class_pred_chi2 = best_class.copy()
    class_pred_chi2[unclassified_mask] = n_classes

    # Reshape to image
    class_map_chi2 = class_map_chi2.reshape(height, width)
    min_distances   = min_distances.reshape(height, width)

    # ── Report ────────────────────────────────────────────────────────────────
    total = height * width
    print(f"\n[chi2_classify] Results:")
    print(f"  Total pixels    : {total:,}")
    print(f"  Classified      : {total - n_unclassified:,} "
          f"({(total - n_unclassified) / total:.2%})")
    print(f"  Unclassified    : {n_unclassified:,} "
          f"({n_unclassified / total:.2%})")
    print(f"\n  Per-class counts:")
    for k_idx, name in enumerate(class_names):
        count = np.sum(class_map_chi2 == k_idx)
        print(f"    {name:<14s}: {count:>8,}  ({count / total:>6.2%})")
    unc = np.sum(class_map_chi2 == n_classes)
    print(f"    {'Unclassified':<14s}: {unc:>8,}  ({unc / total:>6.2%})")

    return class_map_chi2, chi2_threshold, min_distances, n_unclassified


def merge_and_relabel(cluster_map, merge_dict):
    """
    Merge clusters and relabel sequentially so there are no gaps.

    Parameters
    ----------
    cluster_map : np.ndarray (H, W) — original cluster labels (0-indexed)
    merge_dict  : dict — {source_cluster: target_cluster}
                  e.g. {7: 2, 3: 1} means merge cluster 7 into 2,
                  and cluster 3 into 1 (0-indexed)

    Returns
    -------
    remapped    : np.ndarray (H, W) — relabeled cluster map (0 to n_new-1)
    label_map   : dict — {old_label: new_label} for reference
    """

    merged = cluster_map.copy()

    # ── Step 1: Apply merges ──────────────────────────────────────────────────
    for source, target in merge_dict.items():
        merged[merged == source] = target

    # ── Step 2: Relabel sequentially (no gaps) ────────────────────────────────
    old_labels = np.sort(np.unique(merged))
    label_map  = {old: new for new, old in enumerate(old_labels)}

    remapped = np.zeros_like(merged)
    for old, new in label_map.items():
        remapped[merged == old] = new

    print(f"  Merged {len(merge_dict)} cluster(s)")
    print(f"  Old unique labels: {np.sort(np.unique(cluster_map))}")
    print(f"  After merge:       {old_labels}")
    print(f"  Relabeled to:      {np.sort(np.unique(remapped))}")
    print(f"  Label mapping:     {label_map}")

    return remapped, label_map


def plot_kmeans_spectral_response(
        X_raw, cluster_map, k,
        all_band_names, reflective_bands, emissive_bands,
        band_wavelengths=BAND_WAVELENGTHS,
        merge_dict=None,
        mode='all_classes',
        show_individual=True,
        max_individual=300,
        figsize=(13, 5),
        dpi=150,
        save_dir=None,
        ylim_reflective=(0, 100),
        ylim_emissive=None,
):
    """
    Plot spectral response curves for K-Means cluster groups.

    Parameters
    ----------
    X_raw            : np.ndarray (n_pixels, n_bands) — raw (unscaled) feature array
    cluster_map      : np.ndarray (height, width) — cluster labels (already merged
                       and relabeled if applicable)
    k                : int — original number of clusters (for title)
    all_band_names   : list of str — band names matching columns of X_raw
    reflective_bands : list of str — reflective band names
    emissive_bands   : list of str — emissive band names
    band_wavelengths : dict — {band_name: wavelength_um}
    merge_dict       : dict or None — {source: target} merges applied (for title)
    mode             : str — 'all_classes' or 'per_class'
    show_individual  : bool — show individual pixel traces in per_class mode
    max_individual   : int — max individual traces to draw
    figsize          : tuple — figure size
    dpi              : int — resolution
    save_dir         : Path or None — directory to save figures
    ylim_reflective  : tuple or None — y-axis limits for reflective panels
    ylim_emissive    : tuple or None — y-axis limits for emissive panels
    """

    # ── Flatten cluster map to match X_raw ────────────────────────────────────
    cluster_labels = cluster_map.ravel()  # shape: (n_pixels,)

    # ── Build class_labels dict from unique clusters ──────────────────────────
    unique_clusters = np.sort(np.unique(cluster_labels))
    n_clusters = len(unique_clusters)

    # Label dict: {"Cluster 1": 0, "Cluster 2": 1, ...}
    cluster_class_labels = {
        f"Cluster {c + 1}": int(c) for c in unique_clusters
    }

    # ── Generate distinct colors per cluster ──────────────────────────────────
    cmap_base = plt.get_cmap('tab20')
    cluster_colors = {
        f"Cluster {c + 1}": mcolors.to_hex(cmap_base(i))
        for i, c in enumerate(unique_clusters)
    }

    # ── Title ─────────────────────────────────────────────────────────────────
    if merge_dict:
        merge_str = ", ".join([f"{s + 1}→{t + 1}" for s, t in merge_dict.items()])
        title = (f"K-Means Spectral Response (k={k}, "
                 f"merged: {merge_str}, "
                 f"{n_clusters} final clusters)")
    else:
        title = f"K-Means Spectral Response (k={k}, {n_clusters} clusters)"

    # ── Call existing plot function ───────────────────────────────────────────
    class_stats = plot_spectral_response(
        X_raw=X_raw,
        y=cluster_labels,
        all_band_names=all_band_names,
        reflective_bands=reflective_bands,
        emissive_bands=emissive_bands,
        class_labels=cluster_class_labels,
        class_colors=cluster_colors,
        band_wavelengths=band_wavelengths,
        mode=mode,
        show_individual=show_individual,
        max_individual=max_individual,
        figsize=figsize,
        dpi=dpi,
        title_prefix=title,
        save_dir=save_dir,
        ylim_reflective=ylim_reflective,
        ylim_emissive=ylim_emissive,
    )

    return class_stats


def compute_pixel_area_grid(
    sat_data,
    nadir_along_track_resolution,
    nadir_cross_track_resolution,
    sat_orb_height,
    correct_for_earth_curvature=True,
):
    """
    Compute the physical area (m²) of every pixel in the scene
    based on satellite viewing geometry.

    Parameters
    ----------
    sat_data                     : xr.Dataset with 'satellite_zenith_angle'
    nadir_along_track_resolution : float — nadir ground sample distance (m)
    nadir_cross_track_resolution : float — nadir ground sample distance (m)
    sat_orb_height               : float — orbital altitude (m)
    correct_for_earth_curvature  : bool

    Returns
    -------
    pixel_areas  : np.ndarray (y, x) — area of each pixel in m²
    pixel_cross  : np.ndarray (y, x) — cross-track size in m
    pixel_along  : np.ndarray (y, x) — along-track size in m
    """

    earth_radius = 6_378_000.0  # metres

    ifov_cross = 2 * np.arctan((nadir_cross_track_resolution / 2) / sat_orb_height)
    ifov_along = 2 * np.arctan((nadir_along_track_resolution / 2) / sat_orb_height)

    vza   = sat_data["satellite_zenith_angle"].values   # (y, x) degrees
    theta = np.deg2rad(vza)

    if correct_for_earth_curvature:
        phi = np.arcsin(
            (earth_radius + sat_orb_height) / earth_radius * np.sin(theta)
        )
        pixel_along = ifov_along * sat_orb_height * (1.0 / np.cos(theta))
        pixel_cross = (
            ifov_cross
            * (sat_orb_height + earth_radius * (1.0 - np.cos(phi)))
            * (1.0 / np.cos(theta))
        )
    else:
        pixel_along = ifov_along * sat_orb_height * (1.0 / np.cos(theta))
        pixel_cross = ifov_cross * sat_orb_height * (1.0 / np.cos(theta)) ** 2

    pixel_areas = pixel_cross * pixel_along  # m²

    return pixel_areas, pixel_cross, pixel_along


def summarize_class_areas(y_labels, pixel_areas_m2, class_labels,
                          source_name="", area_unit="km2"):
    """
    Summarize total physical area per class from label + area arrays.

    Parameters
    ----------
    y_labels       : np.ndarray (n_samples,) — class labels
    pixel_areas_m2 : np.ndarray (n_samples,) — area of each pixel in m²
    class_labels   : dict — class name → integer label
    source_name    : str — label for printout (e.g. "Training", "MLC")
    area_unit      : str — "m2", "km2", or "ha"

    Returns
    -------
    df : pd.DataFrame — per-class area summary
    """
    unit_factors = {"m2": 1.0, "km2": 1e-6, "ha": 1e-4}
    uf = unit_factors[area_unit]

    records = []
    for class_name, class_val in class_labels.items():
        mask     = (y_labels == class_val)
        n_pixels = mask.sum()
        areas    = pixel_areas_m2[mask]

        records.append({
            "Class":                        class_name,
            "N Pixels":                     int(n_pixels),
            f"Total Area ({area_unit})":    round(areas.sum() * uf, 4),
            f"Mean Pixel ({area_unit})":    round(areas.mean() * uf, 8) if n_pixels > 0 else 0,
            f"Min Pixel ({area_unit})":     round(areas.min() * uf, 8) if n_pixels > 0 else 0,
            f"Max Pixel ({area_unit})":     round(areas.max() * uf, 8) if n_pixels > 0 else 0,
        })

    # Grand total
    total_px   = sum(r["N Pixels"] for r in records)
    total_area = sum(r[f"Total Area ({area_unit})"] for r in records)
    records.append({
        "Class":                        "** TOTAL **",
        "N Pixels":                     total_px,
        f"Total Area ({area_unit})":    round(total_area, 4),
        f"Mean Pixel ({area_unit})":    "—",
        f"Min Pixel ({area_unit})":     "—",
        f"Max Pixel ({area_unit})":     "—",
    })

    df = pd.DataFrame(records)

    if source_name:
        print(f"\n{'=' * 75}")
        print(f"  AREA SUMMARY — {source_name.upper()}")
        print(f"{'=' * 75}")
    print(df.to_string(index=False))
    print()

    return df



















def extract_spectral_stats(
    X_raw,
    y,
    all_band_names,
    class_labels,
    band_wavelengths=BAND_WAVELENGTHS,
    reflective_bands=REFLECTIVE_BANDS,
    emissive_bands=EMISSIVE_BANDS,
    save_dir=None,
    latex_path=None,
    dec_ref=3,
    dec_em=3,
    caption_prefix="NOAA-20 VIIRS",
    table_label_prefix="spectral",
):
    """
    Compute mean & std per band per class — works with ANY label array:
    class labels, predictions, or cluster IDs.

    Parameters
    ----------
    X_raw          : (n, n_bands) raw values
    y              : (n,) integer labels  — class labels, y_pred, OR cluster_map.ravel()
    all_band_names : list of band names matching columns of X_raw
    class_labels   : dict  {'water':0, ...} OR {'Cluster 1':0, 'Cluster 2':1, ...}
    band_wavelengths, reflective_bands, emissive_bands : from your config
    save_dir       : Path — save CSV
    latex_path     : Path — save .tex (mean, std, mean±std)

    Returns
    -------
    df_mean, df_std : pd.DataFrame  — rows = bands, columns = classes
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    all_band_names = list(all_band_names)
    ref_names = [b for b in reflective_bands if b in all_band_names]
    emi_names = [b for b in emissive_bands   if b in all_band_names]

    class_display = list(class_labels.keys())

    mean_rows, std_rows = [], []

    for band_idx, band in enumerate(all_band_names):
        wl = band_wavelengths.get(band, 0.0)
        m_row = {'Band': band, 'Wavelength (µm)': wl}
        s_row = {'Band': band, 'Wavelength (µm)': wl}

        for cls_name, cls_val in class_labels.items():
            vals = X_raw[y == cls_val, band_idx]
            m_row[cls_name] = np.nanmean(vals) if len(vals) > 0 else np.nan
            s_row[cls_name] = np.nanstd(vals)  if len(vals) > 0 else np.nan

        mean_rows.append(m_row)
        std_rows.append(s_row)

    df_mean = pd.DataFrame(mean_rows)
    df_std  = pd.DataFrame(std_rows)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}\n  MEAN\n{'='*80}")
    print(df_mean.to_string(index=False, float_format='%.3f'))
    print(f"\n  STD:")
    print(df_std.to_string(index=False, float_format='%.3f'))

    # ── CSV ───────────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(save_dir / f'{table_label_prefix}_mean.csv', index=False)
        df_std.to_csv(save_dir / f'{table_label_prefix}_std.csv',   index=False)
        print(f"[CSV] → {save_dir}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    if latex_path is not None:
        _write_spectral_latex(
            df_mean, df_std, class_display,
            all_band_names, ref_names, emi_names,
            band_wavelengths, dec_ref, dec_em,
            caption_prefix, table_label_prefix, str(latex_path),
        )

    return df_mean, df_std


def _write_spectral_latex(
    df_mean, df_std, class_display,
    all_band_names, ref_names, emi_names,
    band_wavelengths, dec_ref, dec_em,
    caption_prefix, label_prefix, latex_path,
):
    """Write mean, std, mean±std LaTeX tables to one .tex file."""
    n_cls    = len(class_display)
    col_spec = "l r " + " ".join(["r"] * n_cls)
    lines    = []
    lines.append(r"% Auto-generated by extract_spectral_stats()")
    lines.append(r"% Requires: \usepackage{booktabs, adjustbox}")
    lines.append("")

    header = r"    \textbf{Band} & \textbf{$\lambda$ ($\mu$m)}"
    for cls in class_display:
        header += f" & \\textbf{{{cls.replace('_', ' ').title()}}}"
    header += r" \\"

    def _dec(band):
        return dec_em if band in emi_names else dec_ref

    def _table(caption, label, cell_fn):
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{{label}}}")
        lines.append(r"  \scriptsize")
        lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \midrule")

        prev_em = False
        for idx, band in enumerate(all_band_names):
            is_em = band in emi_names
            if not prev_em and is_em:
                lines.append(r"    \midrule")
            prev_em = is_em
            wl  = band_wavelengths.get(band, 0.0)
            row = f"    {band} & {wl:.3f}"
            for cls in class_display:
                row += f" & {cell_fn(idx, cls)}"
            row += r" \\"
            lines.append(row)

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"  \end{adjustbox}")
        lines.append(r"  \begin{flushleft}")
        lines.append(r"    \footnotesize Reflective in \%; emissive in K.")
        lines.append(r"  \end{flushleft}")
        lines.append(r"\end{table}")
        lines.append("")

    # Mean
    _table(f"{caption_prefix} mean spectral response per class.",
           f"tab:{label_prefix}_mean",
           lambda i, c: f"{df_mean.iloc[i][c]:.{_dec(all_band_names[i])}f}")
    # Std
    _table(f"{caption_prefix} std spectral response per class.",
           f"tab:{label_prefix}_std",
           lambda i, c: f"{df_std.iloc[i][c]:.{_dec(all_band_names[i])}f}")
    # Mean ± Std
    _table(f"{caption_prefix} mean $\\pm$ std spectral response per class.",
           f"tab:{label_prefix}_meanstd",
           lambda i, c: (f"${df_mean.iloc[i][c]:.{_dec(all_band_names[i])}f}"
                         f" \\pm {df_std.iloc[i][c]:.{_dec(all_band_names[i])}f}$"))

    with open(latex_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[LaTeX] → {latex_path}")

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    def viirs_dust_rgb(ds, m12_var='M12', m15_var='M15', m16_var='M16',
                       red_range=(-4, 2), green_range=(0, 15), blue_range=(261, 289),
                       gamma_red=1.0, gamma_green=2.5, gamma_blue=1.0,
                       title="VIIRS Dust RGB", figsize=(12, 10),
                       save_path=None, dpi=200):
        """
        Generate a Dust RGB composite image from a VIIRS xarray Dataset.

        Dust RGB Recipe (adapted from EUMETSAT):
            Red:   BT(M16, 12.0µm) - BT(M15, 10.8µm)   [-4, +2 K],   gamma=1.0
            Green: BT(M15, 10.8µm) - BT(M12, 3.7µm)     [0, +15 K],   gamma=2.5
            Blue:  BT(M15, 10.8µm)                        [261, 289 K], gamma=1.0

        Parameters
        ----------
        ds : xr.Dataset
            xarray Dataset containing VIIRS brightness temperature variables.
        m12_var : str, optional
            Variable name for band M12 (3.7 µm) in the Dataset. Default is 'M12'.
        m15_var : str, optional
            Variable name for band M15 (10.8 µm) in the Dataset. Default is 'M15'.
        m16_var : str, optional
            Variable name for band M16 (12.0 µm) in the Dataset. Default is 'M16'.
        red_range : tuple, optional
            (min, max) range for the Red channel (M16-M15 difference) in K.
            Default is (-4, 2).
        green_range : tuple, optional
            (min, max) range for the Green channel (M15-M12 difference) in K.
            Default is (0, 15).
        blue_range : tuple, optional
            (min, max) range for the Blue channel (M15 BT) in K.
            Default is (261, 289).
        gamma_red : float, optional
            Gamma correction for Red channel. Default is 1.0.
        gamma_green : float, optional
            Gamma correction for Green channel. Default is 2.5.
        gamma_blue : float, optional
            Gamma correction for Blue channel. Default is 1.0.
        title : str, optional
            Title for the plot. Default is "VIIRS Dust RGB".
        figsize : tuple, optional
            Figure size. Default is (12, 10).
        save_path : str or None, optional
            If provided, saves the figure to this path.
        dpi : int, optional
            DPI for saved figure. Default is 200.

        Returns
        -------
        dust_rgb : xr.DataArray
            3-channel (H, W, 3) Dust RGB DataArray with values in [0, 1].
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The matplotlib axes object.
        """

        # ---- Step 1: Extract BT arrays from xarray Dataset ----
        bt_m12 = ds[m12_var].astype(np.float64)
        bt_m15 = ds[m15_var].astype(np.float64)
        bt_m16 = ds[m16_var].astype(np.float64)

        # ---- Step 2: Compute channel differences ----
        red_data = bt_m16 - bt_m15  # BT(12.0µm) - BT(10.8µm)
        green_data = bt_m15 - bt_m12  # BT(10.8µm) - BT(3.7µm)
        blue_data = bt_m15  # BT(10.8µm)

        # ---- Step 3: Normalize each channel to [0, 1] ----
        def normalize(data, vmin, vmax):
            """Clip and scale xarray DataArray to [0, 1]."""
            return ((data - vmin) / (vmax - vmin)).clip(0, 1)

        red = normalize(red_data, *red_range)
        green = normalize(green_data, *green_range)
        blue = normalize(blue_data, *blue_range)

        # ---- Step 4: Apply gamma correction ----
        red = red ** (1.0 / gamma_red)
        green = green ** (1.0 / gamma_green)
        blue = blue ** (1.0 / gamma_blue)

        # ---- Step 5: Stack into RGB DataArray ----
        dust_rgb = xr.concat([red, green, blue], dim='rgb')
        dust_rgb = dust_rgb.transpose(..., 'rgb')  # move RGB to last axis (H, W, 3)

        # Convert to numpy for plotting, fill NaNs with 0 (black)
        rgb_np = dust_rgb.values.copy()
        rgb_np = np.nan_to_num(rgb_np, nan=0.0)
        rgb_np = np.clip(rgb_np, 0, 1)

        # ---- Step 6: Plot ----
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(rgb_np, interpolation='nearest')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Pixel Column')
        ax.set_ylabel('Pixel Row')
        ax.grid(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Dust RGB saved to: {save_path}")

        plt.show()

        return dust_rgb, fig, ax