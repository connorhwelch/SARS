import numpy as np
import pandas as pd
from pyproj import Geod
from datetime import timedelta
from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file
from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime


geod = Geod(ellps="WGS84")


########################################################################################################################
class HistoricalOrbitAnalyzer:
    def __init__(self, satellites, sat_definitions):
        """
        satellites: list of EarthSatellite
        sat_definitions: dict {name: norad_id}
        """
        self.ts = load.timescale()
        self.sat_definitions = sat_definitions

        self.sats_by_norad = defaultdict(list)
        for sat in satellites:
            self.sats_by_norad[sat.model.satnum].append(sat)

        # sort TLEs by epoch
        for norad in self.sats_by_norad:
            self.sats_by_norad[norad].sort(key=lambda s: s.epoch.tt)

    def get_satellite_history(self, name):
        """Get all TLEs for a satellite by name"""
        norad = self.sat_definitions[name]
        return self.sats_by_norad[norad]

    def ground_track(self, name, points_per_tle=300, max_days_last=7.0):
        """Calculate satellite ground track propagation"""
        sats = self.get_satellite_history(name)

        lat, lon, heights, time = [], [], [], []

        for i, sat in enumerate(sats):
            t0 = sat.epoch

            if i < len(sats) - 1:
                t1 = sats[i + 1].epoch
                t_end = self.ts.tt_jd(0.5 * (t0.tt + t1.tt))
            else:
                t_end = self.ts.tt_jd(t0.tt + max_days_last)

            t = self.ts.tt_jd(
                np.linspace(t0.tt, t_end.tt, points_per_tle)
            )

            sp = wgs84.subpoint_of(sat.at(t))
            lat.extend(sp.latitude.degrees)
            lon.extend(sp.longitude.degrees)
            # heights.extend(sp.elevation.m)
            time.extend(t.utc_datetime())

        return {'lat': np.array(lat), 'lon': np.array(lon), 'time': np.array(time)}

    def get_tle_time_differences(self, name, unit='hours'):
        """
        Calculate time differences between consecutive TLEs

        Args:
            name: Satellite name
            unit: Time unit - 'hours', 'days', or 'seconds'

        Returns:
            numpy array of time differences in specified unit
        """
        sat_history = self.get_satellite_history(name)

        if len(sat_history) < 2:
            return np.array([])

        # Extract epochs as numpy datetime64
        epochs = [tle.epoch.utc_datetime().replace(tzinfo=None) for tle in sat_history]
        epochs_np = np.array(epochs, dtype='datetime64[us]')

        # Compute full difference matrix
        diff_matrix = np.abs(epochs_np[:, np.newaxis] - epochs_np[np.newaxis, :])

        # Extract sequential differences (diagonal k=1)
        sequential_diffs = np.diag(diff_matrix, k=1)

        # Convert to desired unit
        if unit == 'hours':
            # Convert timedelta64 to hours
            sequential_diffs = sequential_diffs / np.timedelta64(1, 'h')
        elif unit == 'days':
            sequential_diffs = sequential_diffs / np.timedelta64(1, 'D')
        elif unit == 'seconds':
            sequential_diffs = sequential_diffs / np.timedelta64(1, 's')
        else:
            raise ValueError(f"Unknown unit: {unit}. Use 'hours', 'days', or 'seconds'")

        return sequential_diffs

    def plot_tle_update_frequency(self, name, unit='hours', bins=None, figsize=(8, 4)):
        """
        Plot histogram of time differences between consecutive TLEs

        Args:
            name: Satellite name
            unit: Time unit for x-axis ('hours', 'days', 'seconds')
            bins: Bin specification (None = auto, int = number of bins, array = bin edges)
            figsize: Figure size tuple

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        tle_diffs = self.get_tle_time_differences(name, unit=unit)

        if len(tle_diffs) == 0:
            print(f"Warning: No TLE differences found for {name}")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)

        # Auto-generate bins centered on whole numbers if not specified
        if bins is None:
            min_val = np.floor(tle_diffs.min())
            max_val = np.ceil(tle_diffs.max())
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

        # Create histogram
        ax.hist(tle_diffs,
                color='grey',
                edgecolor='k',
                hatch='///',
                bins=bins,
                alpha=0.8)

        # Set x-ticks at whole numbers
        if isinstance(bins, np.ndarray):
            tick_vals = np.arange(np.ceil(bins.min()), np.floor(bins.max()) + 1, 4)
            ax.set_xticks(tick_vals)
            #ax.set_xticklabels([int(x) for x in tick_vals], rotation=90, ha='left')

        # Labels and title
        unit_label = unit.capitalize()
        ax.set_xlabel(f'Time Between Consecutive TLEs ({unit})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name.capitalize()} TLE Update Frequency')

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add statistics text
        stats_text = (
            f'Mean: {tle_diffs.mean():.2f} {unit}\n'
            f'Median: {np.median(tle_diffs):.2f} {unit}\n'
            f'Min: {tle_diffs.min():.2f} {unit}\n'
            f'Max: {tle_diffs.max():.2f} {unit}\n'
            f'N TLEs: {len(tle_diffs) + 1}'
        )
        ax.text(0.98, 0.97, stats_text,
                color='white',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        plt.tight_layout()

        return fig, ax

    def get_tle_statistics(self, name):
        """
        Get summary statistics for TLE update frequency

        Returns:
            Dictionary with statistics
        """
        sat_history = self.get_satellite_history(name)
        tle_diffs_hours = self.get_tle_time_differences(name, unit='hours')
        tle_diffs_days = self.get_tle_time_differences(name, unit='days')

        if len(tle_diffs_hours) == 0:
            return None

        epochs = [tle.epoch.utc_datetime() for tle in sat_history]

        return {
            'satellite_name': name,
            'norad_id': self.sat_definitions[name],
            'n_tles': len(sat_history),
            'first_epoch': min(epochs),
            'last_epoch': max(epochs),
            'time_span_days': (max(epochs) - min(epochs)).total_seconds() / 86400,
            'mean_update_hours': tle_diffs_hours.mean(),
            'median_update_hours': np.median(tle_diffs_hours),
            'min_update_hours': tle_diffs_hours.min(),
            'max_update_hours': tle_diffs_hours.max(),
            'std_update_hours': tle_diffs_hours.std(),
            'mean_update_days': tle_diffs_days.mean(),
            'median_update_days': np.median(tle_diffs_days),
        }

    def compare_tle_frequencies(self, names=None, unit='hours', figsize=(12, 6)):
        """
        Compare TLE update frequencies across multiple satellites

        Args:
            names: List of satellite names (None = all satellites)
            unit: Time unit
            figsize: Figure size

        Returns:
            fig, ax
        """
        if names is None:
            names = list(self.sat_definitions.keys())

        fig, ax = plt.subplots(figsize=figsize)

        positions = []
        labels = []

        for i, name in enumerate(names):
            diffs = self.get_tle_time_differences(name, unit=unit)
            if len(diffs) > 0:
                bp = ax.boxplot([diffs], positions=[i], widths=0.6,
                                patch_artist=True,
                                boxprops=dict(facecolor='lightblue', alpha=0.7),
                                medianprops=dict(color='red', linewidth=2))
                positions.append(i)
                labels.append(name)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f'Time Between TLEs ({unit})')
        ax.set_title('TLE Update Frequency Comparison')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        return fig, ax

########################################################################################################################
def groundtrack_intersections(track1, track2, max_km=200, max_dt_sec=7200):
    intersections = []

    for i, t1 in enumerate(track1["time"]):
        dt = abs(track2["time"] - t1)
        idx = np.where(dt < timedelta(seconds=max_dt_sec))[0]

        for j in idx:
            # Check if both points are within 50N and 60S latitude range
            lat1 = track1["lat"][i]
            lat2 = track2["lat"][j]

            if not (-60 <= lat1 <= 60 and -60 <= lat2 <= 60):
                continue

            d = ground_distance_km(
                lat1, track1["lon"][i],
                lat2, track2["lon"][j]
            )
            if d <= max_km:
                t2 = track2["time"][j]
                intersections.append((t1, t2, i, j, d))

    return intersections

########################################################################################################################
def ground_distance_km(lat1, lon1, lat2, lon2):
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


########################################################################################################################
# def triple_groundtrack_intersections(intersections_ab, intersections_ac):
#         time_ab = set(t for t, _, _ in intersections_ab)
#         time_ac = set(t for t, _, _ in intersections_ac)
#
#         return sorted(time_ab & time_ac)
#
def triple_groundtrack_intersections(intersections_ab, intersections_ac, time_buffer=timedelta(hours=2)):
    """
    Efficient version using sorted arrays
    """
    if not intersections_ab or not intersections_ac:
        return []

    # Sort by time
    ab_sorted = sorted(intersections_ab, key=lambda x: x[0])
    ac_sorted = sorted(intersections_ac, key=lambda x: x[0])

    matches = []
    ac_time = np.array([t for t, _, _, _ in ac_sorted])

    for t_ab_a, t_ab_b, idx_a_ab, idx_b, d_ab in ab_sorted:
        # Use binary search to find AC time within buffer
        lower_bound = t_ab_a - time_buffer
        upper_bound = t_ab_a + time_buffer

        idx_start = np.searchsorted(ac_time, lower_bound, side='left')
        idx_end = np.searchsorted(ac_time, upper_bound, side='right')

        # Add all matches in this time range
        for i in range(idx_start, idx_end):
            t_ac_a, t_ac_c, idx_a_ac, idx_c, d_ac = ac_sorted[i]
            time_diff = abs(t_ab_a - t_ac_a)
            matches.append((t_ab_a, t_ab_b, t_ac_a, t_ac_c, time_diff))

    return sorted(matches, key=lambda x: x[0])


########################################################################################################################
def load_all_tle(path_to_tle, glob_file_pattern='sat00*'):
    satellites = []
    ts = load.timescale()

    tle_dir = Path(path_to_tle).expanduser()

    for tle_path in tle_dir.glob(glob_file_pattern):
        with load.open(str(tle_path)) as f:
            satellites.extend(
                parse_tle_file(f, ts)
            )

    return satellites


########################################################################################################################
def save_all_tle(satellites, path_to_tle):
    # save as datafile? what kind? text?
    None


########################################################################################################################
def save_groundtrack_matches_csv(matches, filepath, column_labels=None):
    """
    Save matches as CSV file

    Args:
        matches: List of tuples (t_ab_a, t_ab_b, t_ac_a, t_ac_b, time_diff) or detailed dicts
        filepath: Path to save the CSV file
        column_labels: Optional custom column labels
    """
    if column_labels is None:
        column_labels = ['t_ab_a', 't_ab_b', 't_ac_a', 't_ac_b', 'time_diff']

    if not matches:
        print("No matches to save")
        return

    # Check if matches are tuples or dicts
    if isinstance(matches[0], tuple):
        # Convert tuples to DataFrame
        df = pd.DataFrame(matches, columns=column_labels)

        # Convert timedelta to seconds for CSV compatibility
        if 'time_diff' in df.columns:
            if isinstance(df['time_diff'].iloc[0], timedelta):
                df['time_diff_seconds'] = df['time_diff'].apply(lambda x: x.total_seconds())
                df = df.drop('time_diff', axis=1)
    else:
        # Already dicts
        df = pd.DataFrame(matches)

        # Convert timedelta columns to seconds
        if 'time_diff' in df.columns:
            if isinstance(df['time_diff'].iloc[0], timedelta):
                df['time_diff_seconds'] = df['time_diff'].apply(lambda x: x.total_seconds())
                df = df.drop('time_diff', axis=1)

    # Convert datetime columns to readable format (yyyy-mm-ddTHH:MM:SS)
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or isinstance(df[col].iloc[0], (pd.Timestamp, datetime)):
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M'))

    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} matches to {filepath}")


########################################################################################################################
def load_groundtrack_matches_csv(filepath):
    """Load matches from CSV file"""
    df = pd.read_csv(filepath, parse_dates=['time_ab', 'time_ac'])

    # Convert seconds back to timedelta if present
    if 'time_diff_seconds' in df.columns:
        df['time_diff'] = pd.to_timedelta(df['time_diff_seconds'], unit='s')
        df = df.drop('time_diff_seconds', axis=1)

    print(f"Loaded {len(df)} matches from {filepath}")
    return df
