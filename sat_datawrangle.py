import numpy as np
from datetime import datetime, timedelta, timezone

########################################################################################################################
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class GroundTrackIntersection:
    """Intersection between two satellite ground tracks"""
    time_sat_a: datetime
    time_sat_b: datetime
    lat_sat_a: float
    lon_sat_a: float
    lat_sat_b: float
    lon_sat_b: float
    distance_km: float

    @property
    def time_difference(self) -> timedelta:
        """Time difference between the two satellite passes"""
        return abs(self.time_sat_a - self.time_sat_b)

    def __str__(self):
        return (f"Intersection: Sat A at {self.time_sat_a}, "
                f"Sat B at {self.time_sat_b}, "
                f"distance={self.distance_km:.1f}km")


@dataclass
class TripleGroundTrackIntersection:
    """Triple intersection where three satellite tracks come close together"""
    intersection_xy: GroundTrackIntersection  # Intersection between satellites A and B
    intersection_xz: GroundTrackIntersection  # Intersection between satellites A and C

    # Simple property accessors - return single values, not dicts
    @property
    def sat_x_time_xy(self) -> datetime:
        """Time of satellite A during AB intersection"""
        return self.intersection_xy.time_sat_a

    @property
    def sat_x_time_xz(self) -> datetime:
        """Time of satellite A during AC intersection"""
        return self.intersection_xz.time_sat_a

    @property
    def sat_y_time_xy(self) -> datetime:
        """Time of satellite B"""
        return self.intersection_xy.time_sat_b

    @property
    def sat_z_time_xz(self) -> datetime:
        """Time of satellite C"""
        return self.intersection_xz.time_sat_b  # Note: sat_b_time in pair_ac is actually sat C

    @property
    def sat_x_position_xy(self) -> tuple:
        """Latitude and Longitude of satellite X for satellite pair X and Y """
        return (self.intersection_xy.lat_sat_a, self.intersection_xy.lon_sat_b)

    @property
    def sat_y_position_xy(self) -> tuple:
        """Latitude and Longitude of satellite Y for satellite pair X and Y """
        return (self.intersection_xy.lat_sat_b, self.intersection_xy.lon_sat_b)

    @property
    def sat_x_position_xz(self) -> tuple:
        """Latitude and Longitude of satellite X for satellite pair X and Z """
        return self.intersection_xz.lat_sat_a, self.intersection_xz.lon_sat_a

    @property
    def sat_z_position_xz(self) -> tuple:
        """Latitude and Longitude of satellite Z for satellite pair X and Z """
        return self.intersection_xz.lat_sat_b, self.intersection_xz.lon_sat_b

    @property
    def distance_km_xy(self) -> float:
        """Average distance of both intersections"""
        return self.intersection_xy.distance_km

    @property
    def distance_km_xz(self) -> float:
        """Average distance of both intersections"""
        return self.intersection_xz.distance_km

    @property
    def time_diff_xy(self) -> timedelta:
        """Time difference between the two satellite passes X and Y """
        return self.intersection_xy.time_difference

    @property
    def time_diff_xz(self) -> timedelta:
        """Time difference between the two satellite passes X and Y """
        return self.intersection_xz.time_difference
    
    def to_dict(self, sat_x: str, sat_y: str, sat_z: str) -> dict:
        """Convert to dictionary with custom column names"""
        time_diff = abs(self.sat_x_time_xy - self.sat_x_time_xz)
        return {
            f't_{sat_x}_{sat_y}': self.sat_x_time_xy,
            f't_{sat_y}_{sat_x}': self.sat_y_time_xy,
            f't_{sat_x}_{sat_z}': self.sat_x_time_xz,
            f't_{sat_z}_{sat_x}': self.sat_z_time_xz,
            'time_diff': time_diff,
            f'distance_{sat_x}{sat_y}_km': self.distance_km_xy,
            f'distance_{sat_x}{sat_z}_km': self.distance_km_xz,
            f'lat_{sat_x}_{sat_x}{sat_y}': self.sat_x_position_xy[0],
            f'lon_{sat_x}_{sat_x}{sat_y}': self.sat_x_position_xy[1],
            f'lat_{sat_y}_{sat_x}{sat_y}': self.sat_y_position_xy[0],
            f'lon_{sat_y}_{sat_x}{sat_y}': self.sat_y_position_xy[1],
            f'lat_{sat_x}_{sat_x}{sat_z}': self.sat_x_position_xz[0],
            f'lon_{sat_x}_{sat_x}{sat_z}': self.sat_x_position_xz[1],
            f'lat_{sat_z}_{sat_x}{sat_z}': self.sat_z_position_xz[0],
            f'lon_{sat_z}_{sat_x}{sat_z}': self.sat_z_position_xz[1],
        }

    def __str__(self):
        return (f"Triple Intersection Time at XY:{self.time_diff_xy}, XZ {self.time_diff_xz}, | "
                f"XY={self.intersection_xy.distance_km:.1f}km, "
                f"XZ={self.intersection_xz.distance_km:.1f}km, "
                f"dt_XY={self.time_diff_xy}"
                f"dt_XZ={self.time_diff_xz}")

########################################################################################################################
def get_month_bounds(start_date, month_index, months_per_job=1):
    """
    Calculate the start and end dates for a given month range

    Args:
        start_date: datetime object for the start of your 3-year period
        month_index: 0-17 for 18 processors (each handling months_per_job months)
        months_per_job: Number of months each processor handles (default: 2)

    Returns:
        (start_datetime, end_datetime) tuple
    """
    # Calculate actual starting month (0-35 for 36 months)
    actual_month_start = month_index * months_per_job

    # Calculate starting month offset
    year_offset_start = actual_month_start // 12
    month_offset_start = actual_month_start % 12

    # First day of the first month in the range
    month_start = datetime(start_date.year + year_offset_start,
                           start_date.month + month_offset_start,
                           1,
                           tzinfo=timezone.utc)

    # Calculate ending month (last month in the range)
    actual_month_end = actual_month_start + months_per_job - 1
    year_offset_end = actual_month_end // 12
    month_offset_end = actual_month_end % 12

    # First day of the month AFTER the last month
    temp_month = start_date.month + month_offset_end
    temp_year = start_date.year + year_offset_end

    # Handle month overflow
    while temp_month > 12:
        temp_month -= 12
        temp_year += 1

    if temp_month == 12:
        month_end = datetime(temp_year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        month_end = datetime(temp_year, temp_month + 1, 1, tzinfo=timezone.utc)

    # Subtract one second to get last moment of the last month
    month_end = month_end - timedelta(seconds=1)

    return month_start, month_end


########################################################################################################################
def get_month_bounds_flexible(start_date, job_index, total_months=36, total_jobs=18):
    """
    Flexible version that auto-calculates months per job

    Args:
        start_date: datetime object for the start period
        job_index: 0 to (total_jobs-1)
        total_months: Total months to process (default: 36)
        total_jobs: Total number of jobs/processors (default: 18)

    Returns:
        (start_datetime, end_datetime) tuple
    """
    months_per_job = total_months // total_jobs

    if total_months % total_jobs != 0:
        raise ValueError(f"total_months ({total_months}) must be divisible by total_jobs ({total_jobs})")

    return get_month_bounds(start_date, job_index, months_per_job=months_per_job)


########################################################################################################################
def filter_data_by_month(data, month_start, month_end):
    """
    Filter the full dataset to only include data from the specified month range

    Args:
        data: Dictionary with 'lat', 'lon', 'time' keys (numpy arrays or lists)
        month_start: datetime for start of period
        month_end: datetime for end of period

    Returns:
        Filtered dictionary for this period only
    """
    # Format the date range for display
    if month_start.year == month_end.year and month_start.month == month_end.month:
        date_range = month_start.strftime('%Y-%m')
    else:
        date_range = f"{month_start.strftime('%Y-%m')} to {month_end.strftime('%Y-%m')}"

    print(f"Filtering data for {date_range}...")

    # Convert to numpy arrays if not already
    time_array = np.array(data['time'])
    lat_array = np.array(data['lat'])
    lon_array = np.array(data['lon'])

    # Handle different time formats
    if isinstance(time_array[0], datetime):
        # Strip timezone info from time_array if present
        time_array_naive = np.array([t.replace(tzinfo=None) if t.tzinfo is not None else t
                                     for t in time_array])

        # Strip timezone from bounds if present
        month_start_naive = month_start.replace(tzinfo=None) if month_start.tzinfo else month_start
        month_end_naive = month_end.replace(tzinfo=None) if month_end.tzinfo else month_end

        mask = (time_array_naive >= month_start_naive) & (time_array_naive <= month_end_naive)

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