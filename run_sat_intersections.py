from collections import defaultdict
from pathlib import Path
from sat_orbits import *
from sat_info import *
import argparse
from datetime import datetime
from sat_datafilter import *


########################################################################################################################
def args_for_batching():



    parser = argparse.ArgumentParser(description='Process satellite intersections for one month')
    parser.add_argument('--month-index', type=int, required=True,
                        help='Month index (0-35 for 3 years)')
    parser.add_argument('--data-dir', type=str,
                        default='~/Downloads',
                        help='Path to satellite ground track data directory')
    parser.add_argument('--start-date', type=str,
                        default='2023-01-01',
                        help='Start date of 3-year period (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str,
                        default='./.',
                        help='Directory to save results')

    args = parser.parse_args()

    return args

########################################################################################################################
def main(args):

    satellites = load_all_tle(args.data_dir)

    orbit_analyzer = HistoricalOrbitAnalyzer(satellites, satellite_names)

    # AQUA / NOAA20 / SENTINEL-2C selection is based on qualitative analysis of nasa worldview.
    # data is satellite position in lat, lon, datetime determined from historical TLE list from https://celestrak.org
    # ground track uses the TLE position and time and the sgp4 algorithm to propagate the satellite both backwards
    # and forwards in time halfway between the surrounding TLE datapoints.
    # 300 points per TLE file is used selected based on ~ 5 minute points if the TLE is daily updating
    # this was done to limit number of datapoints used for calculations and later intersection determination
    #
    aqua_track = orbit_analyzer.ground_track('aqua')
    noaa20_track = orbit_analyzer.ground_track('noaa20')
    sentinel2c_track = orbit_analyzer.ground_track('sentinel2c')

    # filter data by month
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    # end date determined by function and depends on the amount of processors used
    # in this case 36 processors -> 36 months or 3 years [2023-01-01 through 2026-01-01]
    month_start, month_end = get_month_bounds_flexible(start_date, args.month_index)

    # filter by the monthly data... each processor will be filtering by a different month
    aqua_month_data = filter_data_by_month(aqua_track, month_start, month_end)
    noaa20_month_data = filter_data_by_month(noaa20_track, month_start, month_end)
    sentinel2c_month_data = filter_data_by_month(sentinel2c_track, month_start, month_end)

    # obtains groundtrack intersections for sentinel intersecting aqua and noaa20
    # if sentinel intersects with these satellites it is likely that noaa20 and aqua swaths intersect within the time
    # bound of 4 hours.
    sentinel2c_inter_aqua = groundtrack_intersections(sentinel2c_month_data, aqua_month_data)
    sentinel2c_inter_noaa20 = groundtrack_intersections(sentinel2c_month_data, noaa20_month_data)
    # aqua_inter_noaa20 = groundtrack_intersections(aqua_month_data, noaa20_month_data) # not necessary

    # add triple intersection time
    overpass_times = triple_groundtrack_intersections(sentinel2c_inter_aqua, sentinel2c_inter_noaa20)
    print(overpass_times)
    # save intersections
    save_groundtrack_matches_csv(overpass_times, args.output_dir, column_labels=['sentinel2c-aqua', 'sentinel2c-noaa20', 'diff'])

########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    args = args_for_batching()
    main(args)
