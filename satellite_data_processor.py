
from pathlib import Path
from datetime import datetime

from fiona.crs import defaultdict
from pyresample.resampler import AreaDefinition
from satpy import Scene, find_files_and_readers
import xarray as xr
from xarray import Dataset, DataArray
from typing import Dict

from sat_info import *

def process_satellite_data(
        scene: Scene,
        # satpy_reader: str,
        # start_time: str,
        # end_time: str,
        satellite_name: str,
        satellite_instrument: str,
        identifier: str = None,
        load_recipes: list = None,
        load_composites_recipe: list = None,
        auto_correction: bool = True,
        save_path: Path = None,
        correction_type: str = "both",
        satpy_resample_option: str = "native",
        target_area=None,
        ) -> Dict[str, xr.Dataset]:
    """Process satellite data using SatPy and return xarray datasets.

    Args:
        scene: Base directory containing satellite data files
        start_time: Start time for file search in format 'YYYYMMDDTHHMM'
        end_time: End time for file search in format 'YYYYMMDDTHHMM'
        satellite_name: Name of the satellite (e.g., 'Aqua', 'NOAA-20', 'Sentinel-2b')
        satellite_instrument: Name of the instrument (e.g., 'ABI', 'VIIRS', 'MSI')
        load_recipes: List of tuples specifying (channels, modifiers) for scene loading.
            Each tuple contains:
            - channels: List of channel/band names or variable names (e.g., ['1', '3', '4'])
            - modifiers: List of modifier names to apply (e.g., ['rayleigh_corrected', 'sunz_corrected'])
                        Use empty list [] for no modifiers
            Example:
                VISIBLE = ["1", "3", "4"]
                GEOMETRY  = ["solar_zenith_angle"]
                load_recipes = [
                    (VISIBLE, ["rayleigh_corrected", "sunz_corrected"]),
                    (GEOMETRY, None),
                    (["5"], ["sunz_corrected"]),
                ]
        save_path: Optional directory path where NetCDF files will be saved.
            If None, files are not saved to disk. Default: None
        correction_type: Type of atmospheric correction to apply. Options:
            - "uncorrected": Process only uncorrected data
            - "corrected": Process only corrected data
            - "both": Process both corrected and uncorrected data
            Default: "both"
        satpy_resample_option: Resampling method to use. See Satpy documentation for resampling details.
            Default: "native"
        target_area: Target area definition for resampling. Required when
            satpy_resample_option is not "native". Can be an AreaDefinition object
            or area name string. Default: None

    Returns:
        dict: Dictionary mapping dataset names to xarray Dataset objects.
            Keys are formatted as '{satellite_name}_{satellite_instrument}_{correction_type}'
            Example: {'Terra_MODIS_uncorrected': <xarray.Dataset>,
                     'Terra_MODIS_corrected': <xarray.Dataset>}

    Raises:
        ValueError: If correction_type is not one of the valid options
        ValueError: If no files are found in the specified directory and time range
        ValueError: If target_area is None when using non-native resampling

    Example:
        >>> from pathlib import Path
        >>> data_dir = Path('/data/modis_data')
        >>> load_recipes = [(['1', '3', '4'], ['sunz_corrected', 'rayleigh_corrected']),
                            (['true_color'], [])]
        >>> myfiles = file_processor()
        >>> satpy_scene = Scene(filenames=myfiles)
        >>> datasets = process_satellite_data(
        ...     scene=satpy_scene),
        ...     satpy_reader='modis_l1b',
        # ...     start_time='20260101T1230',
        # ...     end_time='20260101T1230',
        ...     satellite_name='Aqua',
        ...     satellite_instrument='MODIS',
        ...     load_recipes=load_recipes,
        ...     save_path=Path('/output'),
        ...     correction_type='both'
        ... )
        :param identifier:
    """
    # Validate correction_type
    valid_corrections = {"uncorrected", "corrected", "both"}
    if correction_type not in valid_corrections:
        raise ValueError(f"correction_type must be one of {valid_corrections}")

    # # Find files
    # myfiles = find_files_and_readers(
    #     base_dir=str(data_dir),
    #     start_time=datetime.strptime(start_time, "%Y%m%dT%H%M"),
    #     end_time=datetime.strptime(end_time, "%Y%m%dT%H%M"),
    #     reader=satpy_reader
    # )
    #
    # if not myfiles:
    #     raise ValueError(f"No files found in {data_dir} for the specified time range")

    # Determine which correction types to process
    correction_types = []
    if correction_type in ("uncorrected", "both"):
        correction_types.append("uncorrected")
    if correction_type in ("corrected", "both"):
        correction_types.append("corrected")

    # Store results for all correction types
    all_data = {}

    for label in correction_types:

        # if load_recipes is None and auto_correction:
        #     load_recipes = _generate_load_recipes(scene=scene)
        # Load datasets
        for bands, modifiers in load_recipes:
            scene.load(bands, modifiers=modifiers)

        coarsest_area = scene.coarsest_area()
        available = set(scene.available_composite_names())

        composite_list = load_composites_recipe
        use_resolution = (
                isinstance(coarsest_area, AreaDefinition)
                and all(comp in available for comp in composite_list)
        )

        load_kwargs = {}
        if use_resolution:
            print("#### meets criteria for composites sentinel2 ####")
            load_kwargs["resolution"] = coarsest_area.pixel_size_x

        scene.load(composite_list, **load_kwargs)

        # Resample if not using native resolution
        if satpy_resample_option == "native":
            # Native resampling doesn't require target_area
            resampled_scene = scene.resample(scene.coarsest_area(), resampler="native")
        else:
            # Non-native resampling requires target_area
            if target_area is None:
                raise ValueError(f"target_area must be provided when using '{satpy_resample_option}' resampling")
            resampled_scene = scene.resample(destination=target_area, resampler=satpy_resample_option)



        # Convert to xarray
        xr_dataset = resampled_scene.to_xarray(include_lonlats=True)
        xr_dataset = xr_dataset.compute()

        # Store dataset
        dataset_name = f'{satellite_name}_{satellite_instrument}_{label}'
        all_data[dataset_name] = xr_dataset

        # Save if requested
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f'{dataset_name}_{identifier}.nc'
            xr_dataset.to_netcdf(output_file, mode="w")

    return all_data


# def scene_processor(file_list: list, satpy_reader: str, start_time: datetime, end_time: datetime,):
#
#     sat_scenes = defaultdict()
#          myfiles = file_list
#




        # if  satpy_reader == 'msi_safe':
        #     groups = defaultdict(list)
        #     for f in myfiles[data_info['reader']]:
        #         safe_name = Path(f).parts[6]  # grabs SAFE directory name
        #         groups[safe_name].append(f)
        #     msi_scenes = {}
        #     for safe, safe_files in groups.items():
        #         msi_scene = Scene(filenames=safe_files, reader='msi_safe')
        #         process_satellite_data(msi_scene,
        #                                )
        # else:
        #     sat_scene = Scene(myfiles)



def extract_identifier(file_list):
    fname = Path(file_list[0]).name

    if fname.startswith("S2"):
        return fname.split('_')[-2]  # Sentinel-2

    elif ".A" in fname:
        parts = fname.split('.')
        return parts[1] + "_" + parts[2]  # MODIS / VIIRS

    else:
        raise ValueError("Unknown file format")

import xarray as xr
import numpy as np

def pixel_selector(data: xr.Dataset, lat_lon_point: tuple,
                   lat_key: str = 'lat', lon_key: str = 'lon',
                   radius: int = 0) -> xr.Dataset:
    """
    Select the pixel nearest to a given (lat, lon) point and optionally grab surrounding pixels.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing 2D latitude and longitude coordinates with dimensions (y, x).
    lat_lon_point : tuple
        (latitude, longitude) of the target location.
    lat_key : str, optional
        Name of the latitude coordinate in the dataset (default 'lat').
    lon_key : str, optional
        Name of the longitude coordinate in the dataset (default 'lon').
    radius : int, optional
        Number of pixels around the central pixel to include (default is 0).

    Returns
    -------
    xarray.Dataset
        Subset of the dataset around the selected pixel.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> ny, nx = 10, 10
    >>> lats = np.linspace(30, 40, ny).reshape(ny, 1) + np.zeros((ny, nx))
    >>> lons = np.linspace(-110, -100, nx).reshape(1, nx) + np.zeros((ny, nx))
    >>> data_values = np.random.rand(ny, nx)
    >>> dataset = xr.Dataset(
    ...     {"reflectance": (("y", "x"), data_values)},
    ...     coords={"lat": (("y", "x"), lats), "lon": (("y", "x"), lons)}
    ... )
    >>> # Select nearest pixel to (35, -106) with radius 1 (3x3 window)
    >>> subset = pixel_selector(dataset, lat_lon_point=(35, -106), radius=1)
    >>> print(subset)
    <xarray.Dataset>
    Dimensions:      (y: 3, x: 3)
    Coordinates:
      * lat          (y, x) float64 ...
      * lon          (y, x) float64 ...
    Data variables:
        reflectance  (y, x) float64 ...
    """
    lat_target, lon_target = lat_lon_point

    # Use the coordinate keys
    distance_sq = (data[lat_key] - lat_target)**2 + (data[lon_key] - lon_target)**2

    iy, ix = np.unravel_index(distance_sq.argmin().values, distance_sq.shape)

    # Handle edges
    y_start = max(0, iy - radius)
    y_end   = min(data[lat_key].shape[0], iy + radius + 1)
    x_start = max(0, ix - radius)
    x_end   = min(data[lon_key].shape[1], ix + radius + 1)

    return data.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))


