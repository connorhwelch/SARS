"""
config.py
=========
Project-wide constants, paths, and plot styling.
Every script imports from here — nothing is defined twice.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR  = Path('~/Downloads/sars_data').expanduser()
PLOT_DIR  = DATA_DIR / 'p2_plots'
TABLE_DIR = DATA_DIR / 'p2_tables'
MODEL_DIR = DATA_DIR / 'p2_models'

for d in [PLOT_DIR, TABLE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
HEIGHT = 512
WIDTH  = 640

# Subsetting slices applied to the raw VIIRS granule
Y_SLICE = slice(2935 - 256, 2935 + 256)
X_SLICE = slice(1448 - 320, 1448 + 320)

# ══════════════════════════════════════════════════════════════════════════════
# VIIRS BAND DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
BAND_WAVELENGTHS = {
    'M01': 0.412, 'M02': 0.445, 'M03': 0.483, 'M04': 0.555,
    'M05': 0.672, 'M06': 0.746, 'M07': 0.865, 'M08': 1.240,
    'M09': 1.378, 'M10': 1.610, 'M11': 2.255, 'M12': 3.700,
    'M13': 4.050, 'M14': 8.550, 'M15': 10.760, 'M16': 12.015,
}

ALL_BANDS        = [f'M{str(i).zfill(2)}' for i in range(1, 17)]
REFLECTIVE_BANDS = ['M01','M02','M03','M04','M05','M06',
                    'M07','M08','M09','M10','M11']
EMISSIVE_BANDS   = ['M12','M13','M14','M15','M16']

# ══════════════════════════════════════════════════════════════════════════════
# CLASS DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
CLASS_LABELS = {
    'water':      0,
    'cloud':      1,
    'snow':       2,
    'smoke':      3,
    'bare_soil':  4,
    'vegetation': 5,
}

CLASS_NAMES = {v: k for k, v in CLASS_LABELS.items()}

CLASS_COLORS = {
    'water':      '#2166ac',
    'cloud':      '#d1d1d1',
    'snow':       '#a6cee3',
    'smoke':      '#6a3d9a',
    'bare_soil':  '#d6a86b',
    'vegetation': '#4dac26',
}

# Ordered list for colormaps — keeps same order as CLASS_LABELS
CLASS_COLORS_LIST = [CLASS_COLORS[k] for k in CLASS_LABELS]

# Extended (with Unclassified) for chi-squared plots
CLASS_LABELS_EXT  = {**CLASS_LABELS, 'Unclassified': len(CLASS_LABELS)}
CLASS_COLORS_EXT  = CLASS_COLORS_LIST + ['#e31a1c']

# Display names (for plot legends, colorbars, tables)
CLASS_DISPLAY_NAMES = ['Water', 'Cloud', 'Snow', 'Smoke',
                       'Bare Soil', 'Vegetation']

# ══════════════════════════════════════════════════════════════════════════════
# SATELLITE PARAMETERS  (for pixel area calculation)
# ══════════════════════════════════════════════════════════════════════════════
NADIR_ALONG_TRACK = 750.0      # metres  (VIIRS M-bands)
NADIR_CROSS_TRACK = 750.0      # metres
ORBITAL_HEIGHT    = 824_000.0   # metres  (Suomi NPP)

# ══════════════════════════════════════════════════════════════════════════════
# ML PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
CHI2_CONFIDENCE = 0.95

# ══════════════════════════════════════════════════════════════════════════════
# PLOT STYLE — call apply_plot_style() at the top of every script
# ══════════════════════════════════════════════════════════════════════════════
def apply_plot_style():
    """Apply a consistent matplotlib + seaborn style across all scripts."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('default')
    sns.set_style("white")

    mpl.rcParams.update({
        # Font
        'font.family':        'sans-serif',
        'font.size':           11,
        'axes.titlesize':      13,
        'axes.labelsize':      11,
        'xtick.labelsize':     9,
        'ytick.labelsize':     9,
        'legend.fontsize':     9,

        # Figure
        'figure.dpi':          150,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'figure.facecolor':    '#F8F9FA',
        'axes.facecolor':      'white',

        # Lines
        'lines.linewidth':     1.8,
        'lines.markersize':    5,

        # Spines
        'axes.spines.top':     False,
        'axes.spines.right':   False,

        # Image
        'image.cmap':          'viridis',
    })
    print("[config] Plot style applied.")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np

def add_geo_ticks(ax, lon, lat, height=HEIGHT, width=WIDTH,
                  n_xticks=6, n_yticks=6):
    """Add lon/lat tick labels to a pixel-space image axis."""
    x_pixel_pos = np.linspace(0, width  - 1, n_xticks, dtype=int)
    y_pixel_pos = np.linspace(0, height - 1, n_yticks, dtype=int)
    x_lon_vals  = lon[height // 2, x_pixel_pos]
    y_lat_vals  = lat[y_pixel_pos, width  // 2]
    ax.set_xticks(x_pixel_pos)
    ax.set_xticklabels([f"{v:.1f}°E" for v in x_lon_vals], fontsize=9)
    ax.set_yticks(y_pixel_pos)
    ax.set_yticklabels([f"{v:.1f}°N" for v in y_lat_vals], fontsize=9)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude',  fontsize=10)


def save_checkpoint(data_dict, filename, directory=MODEL_DIR):
    """Pickle a dictionary to the model directory."""
    import pickle
    path = directory / filename
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(filename, directory=MODEL_DIR):
    """Load a pickled checkpoint dictionary."""
    import pickle
    path = directory / filename
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"[checkpoint] Loaded ← {path}")
    return data
