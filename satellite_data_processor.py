
from pathlib import Path
from datetime import datetime
from satpy import Scene, find_files_and_readers
import xarray as xr
from typing import Dict

def process_satellite_data(
        data_dir: Path,
        satpy_reader: str,
        start_time: str,
        end_time: str,
        satellite_name: str,
        satellite_instrument: str,
        load_recipes: list,
        save_path: Path = None,
        correction_type: str = "both",
        satpy_resample_option: str = "native",
        target_area=None,
        ) -> Dict[str, xr.Dataset]:
    """Process satellite data using SatPy and return xarray datasets.

    Args:
        data_dir: Base directory containing satellite data files
        satpy_reader: Name of the SatPy reader to use (e.g., 'modis_l1b', 'viirs_l1b', 'msi_safe')
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
                    (GEOMETRY, ["solar_zenith_angle"]),
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
        >>> datasets = process_satellite_data(
        ...     data_dir=data_dir,
        ...     satpy_reader='modis_l1b',
        ...     start_time='20260101T1230',
        ...     end_time='20260101T1230',
        ...     satellite_name='Aqua',
        ...     satellite_instrument='MODIS',
        ...     load_recipes=load_recipes,
        ...     save_path=Path('/output'),
        ...     correction_type='both'
        ... )
    """
    # Validate correction_type
    valid_corrections = {"uncorrected", "corrected", "both"}
    if correction_type not in valid_corrections:
        raise ValueError(f"correction_type must be one of {valid_corrections}")

    # Find files
    myfiles = find_files_and_readers(
        base_dir=str(data_dir),
        start_time=datetime.strptime(start_time, "%Y%m%dT%H%M"),
        end_time=datetime.strptime(end_time, "%Y%m%dT%H%M"),
        reader=satpy_reader
    )

    if not myfiles:
        raise ValueError(f"No files found in {data_dir} for the specified time range")

    # Determine which correction types to process
    correction_types = []
    if correction_type in ("uncorrected", "both"):
        correction_types.append("uncorrected")
    if correction_type in ("corrected", "both"):
        correction_types.append("corrected")

    # Store results for all correction types
    all_data = {}

    for label in correction_types:
        scene = Scene(filenames=myfiles)

        # Load datasets
        for bands, modifiers in load_recipes:
            scene.load(bands, modifiers=modifiers)

        # Resample if not using native resolution
        if satpy_resample_option == "native":
            # Native resampling doesn't require target_area
            resampled_scene = scene.resample(resampler="native")
        else:
            # Non-native resampling requires target_area
            if target_area is None:
                raise ValueError(f"target_area must be provided when using '{satpy_resample_option}' resampling")
            resampled_scene = scene.resample(destination=target_area, resampler=satpy_resample_option)

        # Convert to xarray
        xr_dataset = resampled_scene.to_xarray()

        # Store dataset
        dataset_name = f'{satellite_name}_{satellite_instrument}_{label}'
        all_data[dataset_name] = xr_dataset

        # Save if requested
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f'{dataset_name}_{start_time}-{end_time}.nc'
            xr_dataset.to_netcdf(output_file)

    return all_data