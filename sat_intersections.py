from collections import defaultdict
from pathlib import Path
from sat_orbits import *
from sat_info import *
import argparse
from datetime import datetime

def get_month_bounds(start_date, month_index):
    """
    Calculate the start and end dates for a given month index

    Args:
        start_date: datetime object for the start of your 3-year period
        month_index: 0-35 for 36 months

    Returns:
        (start_datetime, end_datetime) tuple
    """
    # Calculate which month this is
    year_offset = month_index // 12
    month_offset = month_index % 12

    month_start = datetime(start_date.year + year_offset,
                           start_date.month + month_offset,
                           1)

    # Get first day of next month, then subtract one day
    if month_start.month == 12:
        month_end = datetime(month_start.year + 1, 1, 1)
    else:
        month_end = datetime(month_start.year, month_start.month + 1, 1)

    month_end = month_end - timedelta(seconds=1)

    return month_start, month_end


def filter_data_by_month(data, month_start, month_end):
    """
    Filter the full dataset to only include data from the specified month

    Args:
        data: Dictionary with 'lat', 'lon', 'time' keys (numpy arrays or lists)
        month_start: datetime for start of month
        month_end: datetime for end of month

    Returns:
        Filtered dictionary for this month only
    """
    print(f"Filtering data for {month_start.strftime('%Y-%m')}...")

    # Convert to numpy arrays if not already
    time_array = np.array(data['time'])
    lat_array = np.array(data['lat'])
    lon_array = np.array(data['lon'])

    # Handle different time formats
    if isinstance(time_array[0], datetime):
        # Already datetime objects
        mask = (time_array >= month_start) & (time_array <= month_end)
    elif isinstance(time_array[0], (int, float)):
        # Assume Unix timestamp
        month_start_ts = month_start.timestamp()
        month_end_ts = month_end.timestamp()
        mask = (time_array >= month_start_ts) & (time_array <= month_end_ts)
    elif isinstance(time_array[0], np.datetime64):
        # NumPy datetime64
        month_start_np = np.datetime64(month_start)
        month_end_np = np.datetime64(month_end)
        mask = (time_array >= month_start_np) & (time_array <= month_end_np)
    else:
        raise ValueError(f"Unsupported time format: {type(time_array[0])}")

    # Apply mask
    filtered_data = {
        'lat': lat_array[mask],
        'lon': lon_array[mask],
        'time': time_array[mask]
    }

    print(f"Filtered to {len(filtered_data['time']):,} points "
          f"({100 * len(filtered_data['time']) / len(time_array):.1f}% of total)")

    return filtered_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process satellite intersections for one month')
    parser.add_argument('--month-index', type=int, required=True,
                        help='Month index (0-35 for 3 years)')
    parser.add_argument('--data-path', type=str,
                        default='/path/to/satellite_data.nc',
                        help='Path to satellite ground track data')
    parser.add_argument('--start-date', type=str,
                        default='2020-01-01',
                        help='Start date of 3-year period (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str,
                        default='results',
                        help='Directory to save results')

    args = parser.parse_args()

    satellites = load_all_tle("")


    orbit_analyzer = HistoricalOrbitAnalyzer(satellites, satellite_names)

    # based on qualitative anlaysis of nasa worldview use AQUA / NOAA20 / SENTINEL-2C
    aqua_track = orbit_analyzer.ground_track('aqua')
    noaa20_track = orbit_analyzer.ground_track('noaa20')
    sentinel2c_track = orbit_analyzer.ground_track('sentinel2c')

    # filter data by month
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    month_start, month_end = get_month_bounds(start_date, args.month_index)

    monthly_data = filter_data_by_month(full_data, month_start, month_end)

    # multiprocess intersection through slurm allocation
