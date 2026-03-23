import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import rioxarray  # noqa: F401  -- registers .rio accessor
import xarray as xr
from pyresample.resampler import AreaDefinition
from rasterio.transform import from_bounds
from satpy import Scene


def process_satellite_data(
        files: Dict[str, Path],
        satpy_reader: str,
        satellite_name: str,
        satellite_instrument: str,
        identifier: str = None,
        load_recipes: list = None,
        load_composites_recipe: list = None,
        gamma: dict = None,
        auto_correction: bool = True,
        save_path: Path = None,
        save_geotiff: bool = False,
        geotiff_bands: dict = None,   # {"bands": [...], "composites": [...]} or None for auto
        correction_type: str = "both",
        satpy_resample_option: str = "native",
        target_area=None,
        ) -> Dict[str, xr.Dataset]:
    """Process satellite data using SatPy and return xarray datasets.

    Args:
        files: Dictionary mapping reader names to file paths
        satellite_name: Name of the satellite (e.g., 'Aqua', 'NOAA-20', 'Sentinel-2b')
        satellite_instrument: Name of the instrument (e.g., 'ABI', 'VIIRS', 'MSI')
        identifier: Optional scene identifier string (e.g. timestamp)
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
        load_composites_recipe: List of composite names to load (e.g. ['true_color'])
        save_path: Optional directory path where output files will be saved.
            If None, files are not saved to disk. Default: None
        save_geotiff: If True, save each band as a GeoTIFF. Default: False
        geotiff_bands: Optional dict {"bands": [...], "composites": [...]} controlling
            which bands are written to GeoTIFF. If None, auto-derived from load_recipes
            and load_composites_recipe. Default: None
        correction_type: Type of atmospheric correction to apply. Options:
            - "uncorrected": Process only uncorrected data
            - "corrected": Process only corrected data
            - "both": Process both corrected and uncorrected data
            Default: "both"
        satpy_resample_option: Resampling method to use. Default: "native"
        target_area: Target area definition for resampling. Required when
            satpy_resample_option is not "native". Default: None

    Returns:
        dict: Dictionary mapping dataset names to xarray Dataset objects.
            Keys are formatted as '{satellite_name}_{satellite_instrument}_{correction_type}'
            Example: {'Terra_MODIS_uncorrected': <xarray.Dataset>,
                     'Terra_MODIS_corrected': <xarray.Dataset>}

    Raises:
        ValueError: If correction_type is not one of the valid options
        ValueError: If target_area is None when using non-native resampling
    """

    # Validate correction_type
    valid_corrections = {"uncorrected", "corrected", "both"}
    if correction_type not in valid_corrections:
        raise ValueError(f"correction_type must be one of {valid_corrections}")

    # Determine which correction types to process
    correction_types = []
    if correction_type in ("uncorrected", "both"):
        correction_types.append("uncorrected")
    if correction_type in ("corrected", "both"):
        correction_types.append("corrected")

    # Store results for all correction types
    all_data = {}

    for label in correction_types:
        dataset_name = f'{satellite_name}_{satellite_instrument}_{label}'

        if satellite_name == 'terra':
            scene = Scene(
                filenames=files,
                reader=satpy_reader,
                reader_kwargs=dict(mask_saturated=False),
            )
        else:
            scene = Scene(
                filenames=files,
                reader=satpy_reader,
            )

        # Load datasets from recipes
        for bands, modifiers in load_recipes:
            if label == 'uncorrected':
                modifiers = ()
            scene.load(bands, modifiers=modifiers)

        # Load composites
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

        # Resample
        if satpy_resample_option == "native":
            resampled_scene = scene.resample(scene.coarsest_area(), resampler="native")
        else:
            if target_area is None:
                raise ValueError(f"target_area must be provided when using '{satpy_resample_option}' resampling")
            resampled_scene = scene.resample(destination=target_area, resampler=satpy_resample_option)

        # Apply emissive correction to the scene in-place (lazy — no compute yet)
        if label == "corrected":
            apply_emissive_correction_to_scene(resampled_scene)

        # Save GeoTIFF directly from scene (before to_xarray to avoid double compute)
        if save_geotiff and save_path is not None:
            band_selection = geotiff_bands or select_loaded_science_bands(load_recipes, load_composites_recipe)

            save_scene_as_geotiff(
                scene=resampled_scene,
                save_path=save_path,
                dataset_name=dataset_name,
                identifier=identifier,
                bands=band_selection["bands"],
                composites=band_selection["composites"],

            )

        # Convert to xarray for analysis / NetCDF
        xr_dataset = resampled_scene.to_xarray(include_lonlats=True).compute()

        # Store dataset
        all_data[dataset_name] = xr_dataset

        # Save NetCDF if requested
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f'{dataset_name}_{identifier}.nc'
            xr_dataset.to_netcdf(output_file, mode="w")

    return all_data


# ── Band selection ─────────────────────────────────────────────────────────────

def select_loaded_science_bands(load_recipes: list, load_composites_recipe: list) -> dict:
    """
    Return band names split by type, derived directly from what was loaded.

    Geometry/ancillary variables are excluded from the scalar bands list.
    Composites are passed through as-is since they are explicitly requested.

    Returns
    -------
    dict with keys:
        'bands'      : list of scalar band name strings from load_recipes
        'composites' : list of composite name strings from load_composites_recipe
    """
    GEOMETRY_BANDS = {
        "solar_zenith_angle", "solar_azimuth_angle",
        "satellite_zenith_angle", "satellite_azimuth_angle",
        "sensor_zenith_angle", "sensor_azimuth_angle",
        "latitude", "longitude",
    }

    science_bands = [
        band
        for bands, _ in load_recipes
        for band in bands
        if band not in GEOMETRY_BANDS
    ]

    composites = list(load_composites_recipe) if load_composites_recipe else []

    return {"bands": science_bands, "composites": composites}


# ── GeoTIFF export ─────────────────────────────────────────────────────────────

def save_scene_as_geotiff(
    scene,
    save_path: Path,
    dataset_name: str,
    identifier: str,
    bands: list = None,
    composites: list = None,
):
    """
    Save bands from a resampled satpy Scene to GeoTIFF using rioxarray.
    Scalar bands and RGB composites are both handled correctly.

    Parameters
    ----------
    scene      : resampled satpy Scene
    save_path  : output directory
    dataset_name : e.g. 'Terra_MODIS_corrected'
    identifier : granule timestamp / scene ID
    bands      : list of scalar band names to save
    composites : list of composite names to save (e.g. ['true_color'])
    """
    save_path.mkdir(parents=True, exist_ok=True)

    loaded_names = {k['name'] for k in scene.keys()}


    all_names  = list(bands or [])
    all_names += list(composites or [])

    for name in all_names:
        if name not in loaded_names:
            print(f"[warn] '{name}' not found in scene, skipping")
            continue

        da = scene[name].compute()          # materialise dask → numpy
        area = da.attrs.get("area")

        # Normalise dims: composites are (bands, y, x); scalars are (y, x)
        if "bands" in da.dims:
            da = da.transpose("bands", "y", "x")
            if "A" in da.coords["bands"].values:
                da = da.sel(bands=["R", "G", "B"])
        else:
            da = da.transpose("y", "x")

        # Attach CRS from satpy AreaDefinition
        crs = area.crs.to_wkt
        # print(crs)
        da = da.rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

        out_file = save_path / f"{dataset_name}_{identifier}_{name}.tif"
        da.rio.to_raster(out_file, dtype="float32", compress="LZW", tiled=True)
        print(f"[GeoTIFF] saved → {out_file}")


# ── Emissive correction ────────────────────────────────────────────────────────

def apply_emissive_correction_to_scene(scene):
    """
    Apply emissive band correction directly to a satpy Scene in-place.
    Operates lazily — no compute triggered until save or to_xarray.
    """
    for dataset_id in scene.keys():
        da = scene[dataset_id]
        band_type = da.attrs.get("standard_name", "")
        if band_type == "toa_brightness_temperature":
            scene[dataset_id] = emissive_band_corrector(da)


def emissive_band_corrector(band_data: xr.DataArray, emissivity: float = 0.95) -> xr.DataArray:
    """
    Apply emissivity correction to a thermal infrared brightness temperature band.

    Uses the Planck approximation:
        T_corrected = T / (1 + (λT/a) * ln(ε))
    where a = hc/k = 1.438e-2 m·K

    Stays dask-lazy if input is dask-backed.
    """
    a = 1.438e-2  # m·K  (second radiation constant hc/k)
    wavelength_m = np.float32(parse_wavelength(band_data.attrs['wavelength'])['center']) * 1e-6

    corrected = band_data / (1 + (wavelength_m * band_data / a * np.log(emissivity)))

    return xr.DataArray(
        corrected.data,
        dims=band_data.dims,
        coords=band_data.coords,
        attrs=band_data.attrs,
    )


# ── Wavelength parsing ─────────────────────────────────────────────────────────

def parse_wavelength(wavelength, output_unit: str = "") -> dict:
    """
    Parse wavelength from either:
    - satpy WavelengthRange namedtuple: WavelengthRange(min, central, max, unit)
    - string like: '0.443 µm (0.415-0.47 µm)'

    Returns
    -------
    dict with keys 'center', 'lower', 'upper' in µm (default) or nm
    """
    if hasattr(wavelength, 'central'):          # satpy WavelengthRange
        center = wavelength.central
        lower  = wavelength.min
        upper  = wavelength.max

    elif isinstance(wavelength, str):
        center = float(re.findall(r"\d+\.\d+", wavelength)[0])
        range_match = re.search(r"\((.*?)\)", wavelength)
        lower, upper = map(float, re.findall(r"\d+\.\d+", range_match.group(1)))

    else:
        raise TypeError(f"Unsupported wavelength type: {type(wavelength)}")

    if output_unit == "nm":
        return {'center': center * 1000, 'lower': lower * 1000, 'upper': upper * 1000}
    return {'center': center, 'lower': lower, 'upper': upper}


# ── Utilities ──────────────────────────────────────────────────────────────────

def extract_identifier(file_list) -> str:
    """Extract a scene identifier string from the first filename in a list."""
    fname = Path(file_list[0]).name

    if fname.startswith("S2"):
        return fname.split('_')[-2]             # Sentinel-2

    elif ".A" in fname:
        parts = fname.split('.')
        return parts[1] + "_" + parts[2]        # MODIS / VIIRS

    else:
        raise ValueError(f"Unknown file format: {fname}")


def pixel_selector(
    data: xr.Dataset,
    lat_lon_point: tuple,
    lat_key: str = 'latitude',
    lon_key: str = 'longitude',
    radius: int = 0,
) -> tuple[xr.Dataset, tuple]:
    """
    Select the pixel nearest to a given (lat, lon) point and optionally grab
    surrounding pixels.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset with 2D latitude/longitude coordinates and dims (y, x).
    lat_lon_point : tuple
        (latitude, longitude) of the target location.
    lat_key : str
        Name of the latitude coordinate (default 'latitude').
    lon_key : str
        Name of the longitude coordinate (default 'longitude').
    radius : int
        Number of pixels around the central pixel to include (default 0).

    Returns
    -------
    xarray.Dataset
        Subset of the dataset around the selected pixel.
    tuple
        (y_start, y_end, x_start, x_end) indices of the selected window.
    """
    lat_target, lon_target = lat_lon_point

    distance_sq = (data[lat_key] - lat_target) ** 2 + (data[lon_key] - lon_target) ** 2
    iy, ix = np.unravel_index(distance_sq.argmin().values, distance_sq.shape)

    y_start = max(0, iy - radius)
    y_end   = min(data[lat_key].shape[0], iy + radius + 1)
    x_start = max(0, ix - radius)
    x_end   = min(data[lon_key].shape[1], ix + radius + 1)

    return data.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)), (y_start, y_end, x_start, x_end)


def get_pixel_info(
    sat_data: xr.DataArray,
    nadir_along_track_resolution,
    nadir_cross_track_resoltuion,
    sat_orb_height,
    correct_for_earth_curvature: bool,
    lat_lon_point: tuple,
    radius: int = 1,
) -> dict:

    earth_radius = 6378 * 1000  # metres

    natr = nadir_along_track_resolution
    nctr = nadir_cross_track_resoltuion

    pixel_data, (ys, ye, xs, xe) = pixel_selector(sat_data, lat_lon_point=lat_lon_point, radius=radius)

    saa = sat_data['solar_azimuth_angle'][ys:ye, xs:xe].values
    sza = sat_data['solar_zenith_angle'][ys:ye, xs:xe].values
    vaa = sat_data['satellite_azimuth_angle'][ys:ye, xs:xe].values
    vza = sat_data['satellite_zenith_angle'][ys:ye, xs:xe].values

    ifov_cross_track = 2 * np.arctan((nctr / 2) / sat_orb_height)
    ifov_along_track = 2 * np.arctan((natr / 2) / sat_orb_height)

    if correct_for_earth_curvature:
        p_along = ifov_along_track * sat_orb_height * (1 / np.cos(np.deg2rad(vza)))

        theta = np.deg2rad(vza)
        phi   = np.arcsin((earth_radius + sat_orb_height) / earth_radius * np.sin(theta))

        p_cross = (
            ifov_cross_track
            * (sat_orb_height + earth_radius * (1 - np.cos(phi)))
            * (1 / np.cos(theta))
        )
        p_cross_nadir = ifov_cross_track * (sat_orb_height + earth_radius * (1 - np.cos(phi)))
        p_along_nadir = ifov_along_track * sat_orb_height

        eps = (
            ifov_cross_track
            * (sat_orb_height + earth_radius * (1 - np.cos(phi)))
            * (1 / np.cos(theta))
            * (1 / np.cos(theta - phi))
        )

        return {
            "pixel_size_crosstrack":        p_cross,
            "pixel_size_along_track":       p_along,
            "pixel_size_along_track_nadir": p_along_nadir,
            "pixel_size_crosstrack_nadir":  p_cross_nadir,
            "selected_area_size":           (p_cross.mean(axis=0).sum(), p_along.mean(axis=1).sum()),
            "avg_pixel_size_crosstrack":    eps.mean(),
            "stdv_pixel_size_crosstrack":   eps.std(),
            "avg_pixel_size_alongtrack":    p_along.mean(),
            "stdv_pixel_size_alongtrack":   p_along.std(),
            "effective_pixel_size":         eps,
            "mean_effective_pixel_size":    eps.mean(),
            "stdv_effective_pixel_size":    eps.std(),
        }

    else:
        p_cross = ifov_cross_track * sat_orb_height * (1 / np.cos(np.deg2rad(vza))) ** 2
        p_along = ifov_along_track * sat_orb_height * (1 / np.cos(np.deg2rad(vza)))

        p_cross_nadir = ifov_cross_track * sat_orb_height
        p_along_nadir = ifov_along_track * sat_orb_height

        return {
            "pixel_size_crosstrack":        p_cross,
            "pixel_size_along_track":       p_along,
            "pixel_size_along_track_nadir": p_along_nadir,
            "pixel_size_crosstrack_nadir":  p_cross_nadir,
            "selected_area_size":           (p_cross.mean(axis=0).sum(), p_along.mean(axis=1).sum()),
            "avg_pixel_size_crosstrack":    p_cross.mean(),
            "stdv_pixel_size_crosstrack":   p_cross.std(),
            "avg_pixel_size_alongtrack":    p_along.mean(),
            "stdv_pixel_size_alongtrack":   p_along.std(),
            "effective_pixel_size":         60,
            "mean_effective_pixel_size":    60,
            "stdv_effective_pixel_size":    0,
        }


def gamma_correction(pixel_val, min, max, gamma):
    return 255 * ((pixel_val - min) / (max - min)) ** (1 / gamma)