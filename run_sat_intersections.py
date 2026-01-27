from collections import defaultdict
from pathlib import Path
from sat_orbits import *
from sat_info import *
import argparse
from datetime import datetime
from sat_datafilter import *

########################################################################################################################
def args_for_hpc():
    parser = argparse.ArgumentParser(description='Process satellite intersections for one month')
    parser.add_argument('--month-index', type=int, required=True,
                        help='Month index (0-35 for 3 years)')
    parser.add_argument('--data-path', type=str,
                        default='/path/to/satellite_data.nc',
                        help='Path to satellite ground track data')
    parser.add_argument('--start-date', type=str,
                        default='2023-01-01',
                        help='Start date of 3-year period (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str,
                        default='sat_intersect_results',
                        help='Directory to save results')

    args = parser.parse_args()

########################################################################################################################
def main():
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


if __name__ == '__main__':

    args_for_hpc()
    main()
