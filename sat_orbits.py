import numpy as np
from pyproj import Geod
from datetime import timedelta
from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file
from collections import defaultdict
from pathlib import Path

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
        norad = self.sat_definitions[name]
        return self.sats_by_norad[norad]

    # calculates a satellite propogation
    def ground_track(self, name, points_per_tle=300, max_days_last=7.0):
        sats = self.get_satellite_history(name)

        lats, lons, heights, times = [], [], [], []

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
            lats.extend(sp.latitude.degrees)
            lons.extend(sp.longitude.degrees)
            heights.extend(sp.elevation.m)
            times.extend(t.utc_datetime())

        return {'lats':np.array(lats), 'lons':np.array(lons), 'times':np.array(times)}


########################################################################################################################
def groundtrack_intersections(track1, track2, max_km=100, max_dt_sec=10800):

    intersections = []

    for i, t1 in enumerate(track1["times"]):
        dt = abs(track2["times"] - t1)
        idx = np.where(dt < timedelta(seconds=max_dt_sec))[0]

        for j in idx:
            d = ground_distance_km(
                track1["lats"][i], track1["lons"][i],
                track2["lats"][j], track2["lons"][j]
            )
            if d <= max_km:
                intersections.append((t1, i, j, d))

    return intersections


########################################################################################################################
def ground_distance_km(lat1, lon1, lat2, lon2):
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


########################################################################################################################
def triple_intersections(intersections_ab, intersections_ac):
        times_ab = set(t for t, _, _ in intersections_ab)
        times_ac = set(t for t, _, _ in intersections_ac)

        return sorted(times_ab & times_ac)


########################################################################################################################
def load_all_tle(path_to_tle, glob_file_pattern='sat00*'):
    satellites = []
    sats_by_norad = defaultdict(list)
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
