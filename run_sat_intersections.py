from collections import defaultdict
from pathlib import Path
from sat_orbits import *
from sat_info import *
import argparse
from datetime import datetime, timedelta, timezone
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
                        default='./',
                        help='Directory to save results')

    args = parser.parse_args()

    return args

########################################################################################################################
def main(args):

    sat_tracks = {}
    monthly_filtered = {}
    intersections_ab = {}
    intersections_ac = {}
    triple_intersections = {}

    satellites = load_all_tle(args.data_dir)

    orbit_analyzer = HistoricalOrbitAnalyzer(satellites, satellite_names)

    # AQUA / NOAA20 / SENTINEL-2C selection is based on qualitative analysis of nasa worldview.
    # data is satellite position in lat, lon, datetime determined from historical TLE list from https://celestrak.org
    # ground track uses the TLE position and time and the sgp4 algorithm to propagate the satellite both backwards
    # and forwards in time halfway between the surrounding TLE datapoints.
    # 300 points per TLE file is used selected based on ~ 5 minute points if the TLE is daily updating
    # this was done to limit number of datapoints used for calculations and later intersection determination
    #
    for sats in satellite_names.keys():
        sat_tracks[sats] = orbit_analyzer.ground_track(sats)

    # filter data by month
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    # end date determined by function and depends on the amount of processors used
    # in this case 36 processors -> 36 months or 3 years [2023-01-01 through 2026-01-01]
    month_start, month_end = get_month_bounds_flexible(start_date, args.month_index)

    # filter by the monthly data... each processor will be filtering by a different month
    for sats in satellite_names.keys():
        monthly_filtered[sats] = filter_data_by_month(sat_tracks[sats],
                                                      month_start,
                                                      month_end)

    # obtains groundtrack intersections for sentinel intersecting aqua and noaa20
    # if sentinel intersects with these satellites it is likely that noaa20 and aqua swaths intersect within the time
    # bound of 4 hours.

    for msi in ['sentinel2a', 'sentinel2b', 'sentinel2c']:
        for modis in ['aqua', 'terra']:
            for viirs in ['noaa20', 'noaa21']:
                # Safely retrieve data using dictionary
                msi_data = monthly_filtered.get(msi)
                modis_data = monthly_filtered.get(modis)
                viirs_data = monthly_filtered.get(viirs)

                # Check if all data exists
                if msi_data and modis_data and viirs_data:

                    key_ab = f"{msi}_{modis}"
                    key_ac = f"{msi}_{viirs}"
                    key_abc = f"{msi}_{modis}_{viirs}"

                    intersections_ab[key_ab] = groundtrack_intersections(msi_data, modis_data,
                                                                      max_dt_sec=3600,
                                                                      lat_bounds=(-45,45))

                    intersections_ac[key_ac] = groundtrack_intersections(msi_data, viirs_data,
                                                                      max_dt_sec=3600,
                                                                      lat_bounds=(-45,45))

                    try:
                        triple_intersections[key_abc] = triple_groundtrack_intersections(
                            intersections_ab,
                            intersections_ac,
                            time_buffer=timedelta(hours=1),
                        )
                    except Exception as e:
                        print(
                            f"[WARN] Triple intersection failed for {key_abc} "
                            f"Intersections for A B:    {len(intersections_ab[key_ab])}"
                            f"Intersections for A C:    {len(intersections_ac[key_ac])}"
                            f"(AB={key_ab}, AC={key_ac}): {e}"
                        )
                        continue

                else:
                    raise ValueError('No data for msi {} or modis {} or virrs {}'.format(msi_data, modis_data, viirs_data))


    for key, overpass_times in triple_intersections.items():
        msi_sat_name, modis_sat_name, viirs_sat_name = key.split('_')
        df = pd.DataFrame(overpass_times,
                          columns=[f't_{msi_sat_name}_{modis_sat_name}',
                                   f't_{modis_sat_name}_{msi_sat_name}',
                                   f't_{msi_sat_name}_{viirs_sat_name}',
                                   f't_{viirs_sat_name}_{msi_sat_name}',
                                   'time_diff']
                          )
        for col in df.columns[0:-1]:
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M'))

        df.to_csv(Path(args.output_dir) / f'{key}-gtmatch_{args.month_index}.csv')


########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    args = args_for_batching()
    main(args)
