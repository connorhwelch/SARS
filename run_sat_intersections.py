from collections import defaultdict
from pathlib import Path
from sat_orbits import *
from sat_info import *
import argparse
from datetime import datetime, timedelta, timezone
from sat_datawrangle import *


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
def main(args, timebuffer_hours=2, sep_dist_km=100.0):
    timediff_hour = timedelta(hours=timebuffer_hours)
    timediff_seconds = timediff_hour.total_seconds()
    sat_tracks = {}
    monthly_filtered = {}
    intersections_ab = {}
    intersections_ac = {}
    triple_intersections = {}

    satellites = load_all_tle(args.data_dir)
    orbit_analyzer = HistoricalOrbitAnalyzer(satellites, satellite_names)

    # Generate ground tracks
    for sats in satellite_names.keys():
        print(f'Processing satellite track - {sats}')
        sat_tracks[sats] = orbit_analyzer.ground_track(sats, daytime_only=True)

    # Filter data by month
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    month_start, month_end = get_month_bounds_flexible(start_date, args.month_index)

    for sats in satellite_names.keys():
        print(f'--- Filtering satellite data for - {sats} ---')
        monthly_filtered[sats] = filter_data_by_month(sat_tracks[sats],
                                                      month_start,
                                                      month_end)

    # Find triple intersections
    for msi in ['sentinel2a', 'sentinel2b', 'sentinel2c']:
        for modis in ['aqua', 'terra']:
            for viirs in ['noaa20', 'noaa21']:
                print(f"--- STEP --- {msi}, {modis}, {viirs} ----")
                msi_data = monthly_filtered.get(msi)
                modis_data = monthly_filtered.get(modis)
                viirs_data = monthly_filtered.get(viirs)

                if msi_data and modis_data and viirs_data:
                    key_ab = f"{msi}_{modis}"
                    key_ac = f"{msi}_{viirs}"
                    key_abc = f"{msi}_{modis}_{viirs}"

                    intersections_ab[key_ab] = groundtrack_intersections(
                        msi_data, modis_data,
                        max_dt_sec=timediff_seconds,
                        max_km=sep_dist_km,
                        lat_bounds=(-45, 45)
                    )

                    intersections_ac[key_ac] = groundtrack_intersections(
                        msi_data, viirs_data,
                        max_dt_sec=timediff_seconds,
                        max_km=sep_dist_km,
                        lat_bounds=(-45, 45)
                    )

                    try:
                        triple_intersections[key_abc] = triple_groundtrack_intersections(
                            intersections_ab[key_ab],
                            intersections_ac[key_ac],
                            max_time_window=timediff_hour,
                            max_distance_km=sep_dist_km
                        )
                        print(f"[Success] Triple Intersection Completed For {key_abc}")
                    except Exception as e:
                        print(
                            f"[WARN] Triple intersection failed for {key_abc}\n"
                            f"       Intersections for AB: {len(intersections_ab[key_ab])}\n"
                            f"       Intersections for AC: {len(intersections_ac[key_ac])}\n"
                            f"       Error: {e}"
                        )
                        continue
                else:
                    raise ValueError(
                        f'[ERROR] Missing data - msi: {len(msi_data) if msi_data else 0}, '
                        f'modis: {len(modis_data) if modis_data else 0}, '
                        f'viirs: {len(viirs_data) if viirs_data else 0}'
                    )

    # Save results to CSV
    for key, matches in triple_intersections.items():
        if not matches:
            print(f"[WARN] No matches found for {key}, skipping CSV creation")
            continue

        msi_sat_name, modis_sat_name, viirs_sat_name = key.split('_')

        # Convert dataclass objects to dictionaries
        data_dicts = [
            match.to_dict(msi_sat_name, modis_sat_name, viirs_sat_name)
            for match in matches
        ]

        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(data_dicts)

        # Format datetime columns
        time_columns = [col for col in df.columns if col.startswith('t_')]
        for col in time_columns:
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))

        # Save to CSV
        output_path = (Path(args.output_dir) /
                       f'{key}-gtmatch_{args.month_index}_tbuff_{timebuffer_hours}hr_{sep_dist_km}km.csv')
        df.to_csv(output_path, index=False)
        print(f"[Success] Data saved to: {output_path}")
        print(f"          Rows: {len(df)}, Columns: {len(df.columns)}")
########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    args = args_for_batching()
    main(args,
         timebuffer_hours=2,
         sep_dist_km=145)
