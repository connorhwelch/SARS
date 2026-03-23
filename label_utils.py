import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pathlib import Path


def geojson_to_raster(
    geojson_path: str | Path,
    reference_raster_path: str | Path,
    output_path: str | Path,
    label_field: str = None,
    default_value: int = 1,
    background_value: int = 0,
    dtype: str = "uint8",
    all_touched: bool = False,
) -> Path:
    """
    Burn GeoJSON features into a raster aligned to a reference GeoTIFF.
    Designed to convert leafmap-drawn training labels into a mask raster.

    Parameters
    ----------
    geojson_path : path to the GeoJSON file (e.g. from m.save_draw_features())
    reference_raster_path : path to the satellite GeoTIFF to align to
    output_path : where to write the label raster
    label_field : attribute field in the GeoJSON to use as pixel value.
                  If None, all features are burned with default_value.
                  Useful if you labelled features with class IDs in leafmap.
    default_value : burn value when label_field is None (default: 1)
    background_value : value for pixels with no feature (default: 0)
    all_touched : if True, burn all pixels touched by geometry, not just
                  those whose centre falls inside. Better for thin features.
    dtype : output raster dtype. 'uint8' supports 0-255 class labels.

    Returns
    -------
    Path to the written label raster.

    Example
    -------
    >>> geojson_to_raster(
    ...     geojson_path="labels.geojson",
    ...     reference_raster_path="Terra_MODIS_corrected_20260101_1230_1.tif",
    ...     output_path="labels_raster.tif",
    ...     label_field="class_id",   # or None for binary mask
    ... )
    """
    geojson_path = Path(geojson_path)
    reference_raster_path = Path(reference_raster_path)
    output_path = Path(output_path)

    # ── Load GeoJSON ──────────────────────────────────────────────────────
    gdf = gpd.read_file(geojson_path)

    if gdf.empty:
        raise ValueError(f"No features found in {geojson_path}")

    # ── Open reference raster to get spatial metadata ────────────────────
    with rasterio.open(reference_raster_path) as ref:
        transform = ref.transform
        crs       = ref.crs
        width     = ref.width
        height    = ref.height
        out_meta  = ref.meta.copy()

    # ── Reproject GeoJSON to match raster CRS if needed ──────────────────
    if gdf.crs is None:
        print("[warn] GeoJSON has no CRS — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")

    if gdf.crs != crs:
        print(f"[info] Reprojecting labels from {gdf.crs} → {crs}")
        gdf = gdf.to_crs(crs)

    # ── Build (geometry, value) pairs for rasterize ───────────────────────
    if label_field and label_field in gdf.columns:
        shapes = [
            (geom, int(val))
            for geom, val in zip(gdf.geometry, gdf[label_field])
            if geom is not None and not geom.is_empty
        ]
    else:
        if label_field:
            print(f"[warn] Field '{label_field}' not found — using default_value={default_value}")
        shapes = [
            (geom, default_value)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]

    if not shapes:
        raise ValueError("No valid geometries to rasterize.")

    # ── Burn features into raster ─────────────────────────────────────────
    label_array = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=background_value,
        dtype=dtype,
        all_touched=all_touched,
    )

    # ── Write output ──────────────────────────────────────────────────────
    out_meta.update({
        "count": 1,
        "dtype": dtype,
        "compress": "LZW",
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(label_array, 1)
        dst.update_tags(
            source_geojson=str(geojson_path),
            label_field=label_field or "default",
        )

    print(f"[label raster] saved → {output_path}  shape=({height},{width})")
    return output_path