# name, noradid
satellite_names = {
    "terra": 25994,
    "aqua": 27424,
    "landsat8": 39084,
    "sentinel2a": 40697,
    "noaa20": 43013,
    "landsat9": 49260,
    "noaa21": 54234,
    "sentinel2c":60989,
    "sentinel2b":42063,
    "snpp":37849,
}

instrument_swath_widths = {
    'MODIS': 2330 * 1000,
    'VIIRS': 3060 * 1000,
    'Landsat': 185 * 1000,
    'Sentinel2': 290 * 1000
}

terra_visible = [
    "1",   # 0.645 µm (red, 250m)
    "3",   # 0.469 µm (blue, 500m)
    "4",   # 0.555 µm (green, 500m)
    "8",   # 0.41 µm
    "9",   # 0.44 µm
    "10",  # 0.49 µm
    "11",  # 0.53 µm
    "12",  # 0.55 µm
    "13hi",  # 0.67 µm
    "13lo",
    "14hi",  # 0.68 µm
    "14lo",

]

terra_swir_nir = [
    "2",  # 0.859 µm (NIR, 250m)
    "5",   # 1.24 µm
    "6",   # 1.64 µm
    "7",   # 2.13 µm
    "15",  # 0.75 µm
    "16",  # 0.86 µm
    "17",  # 0.91 µm
    "18",  # 0.94 µm
    "19",  # 0.94 µm (water vapor)
]

terra_ir = [
    "20",  # 3.75 µm
    "21",  # 3.96 µm
    "22",  # 3.96 µm
    "23",  # 4.05 µm
    "24",  # 4.47 µm
    "25",  # 4.52 µm
    "27",  # 6.72 µm (water vapor)
    "28",  # 7.33 µm
    "29",  # 8.55 µm
    "30",  # 9.73 µm
    "31",  # 11.03 µm
    "32",  # 12.02 µm
    "33",  # 13.33 µm
    "34",  # 13.63 µm
    "35",  # 13.93 µm
    "36",  # 14.24 µm
]

terra_geometry = [
    'height',
    # 'latitude',
    # 'longitude',
    'range',
    'satellite_azimuth_angle',
    'satellite_zenith_angle',
    'solar_azimuth_angle',
    'solar_zenith_angle',
]

terra_composites = [
    'true_color',
    'natural_color',
    'day_microphysics',
    'airmass',
    'dust',
    'ocean_color',
    'snow',
]

noaa20_iband_visible = [
    "I01",  # 0.64 µm (red)
]

noaa20_iband_swir_nir = [
    "I02",  # 0.86 µm (NIR)
    "I03",  # 1.61 µm (SWIR, reflective)
]
noaa20_mband_visible = [
    "M01",  # 0.41 µm
    "M02",  # 0.45 µm
    "M03",  # 0.49 µm
    "M04",  # 0.55 µm
    "M05",  # 0.67 µm
]

noaa20_iband_ir = [
    "I04",  # 3.74 µm
    "I05",  # 11.45 µm
]

noaa20_mband_ir = [
    "M12",  # 3.7 µm
    "M13",  # 4.05 µm
    "M14",  # 8.55 µm
    "M15",  # 10.76 µm
    "M16",  # 12.01 µm
]

noaa20_mband_swir_nir = [
    "M06",  # 0.75 µm (NIR edge)
    "M07",  # 0.86 µm (NIR)
    "M08",  # 1.24 µm (NIR)
    "M09",  # 1.38 µm (water vapor, SWIR)
    "M10",  # 1.61 µm (SWIR)
    "M11",  # 2.25 µm (SWIR)
]

noaa20_geometry = [
    # 'i_lat',
    # 'i_lon',
    # 'm_lat',
    # 'm_lon',
    'satellite_azimuth_angle',
    'satellite_zenith_angle',
    'solar_azimuth_angle',
    'solar_zenith_angle'
]

noaa20_composites = [
    'true_color',
    'natural_color',
    'false_color',
    # 'day_microphysics',
    'dust',
    # 'ash',
    # 'cloudtop_daytime',
    # 'snow',
    # 'cloud_phase',
    # 'cimss_cloud_type',
]

sentinel2b_geometry = [
    'satellite_azimuth_angle',
    'satellite_zenith_angle',
    'solar_azimuth_angle',
    'solar_zenith_angle'
]

sentinel2b_bands_visible = [
    "B01",  # 0.443 µm      Coastal aerosol (60m)
    "B02",  # 0.490 µm      Blue (10m)
    "B03",  # 0.560 µm      Green (10m)
    "B04",  # 0.665 µm      Red (10m)
]

sentinel2b_swir_nir =[
    "B05",  # 0.705 µm      Red edge (20m)
    "B06",  # 0.740 µm      Red edge (20m)
    "B07",  # 0.783 µm      Red edge (20m)
    "B08",  # 0.842 µm      NIR (10m)
    "B8A",  # 0.865 µm      Narrow NIR (20m)
    "B09",  # 0.945 µm      Water vapor (60m)
    "B10",  # 1.375 µm      Cirrus (60m)
    "B11",  # 1.610 µm      SWIR1 (20m)
    "B12",  # 2.190 µm      SWIR2 (20m)
]

sentinel2b_composites = [
    'true_color',
    'natural_color',
    'false_color',
    'urban_color',
    'ndvi',
    # 'aerosol_optical_thickness',
]

# (bands, (satpy modifiers) )
terra_load_recipe = [
    (terra_geometry, ()),
    (terra_ir, ()),
    (terra_swir_nir, ('sunz_corrected',)),
    (terra_visible, ('sunz_corrected', 'rayleigh_corrected')),
]

noaa20_load_recipe = [
    (noaa20_geometry, ()),
    (noaa20_iband_ir, ()),
    (noaa20_mband_ir, ()),
    (noaa20_mband_swir_nir, ('sunz_corrected',)),
    (noaa20_iband_swir_nir, ('sunz_corrected',)),
    (noaa20_iband_visible, ('sunz_corrected_iband', 'rayleigh_corrected_iband')),
    (noaa20_mband_visible, ('sunz_corrected', 'rayleigh_corrected')),
]

sentinel2b_load_recipe = [
    (sentinel2b_geometry, ()),
    (sentinel2b_swir_nir, ('sunz_corrected',)),
    (sentinel2b_bands_visible, ('sunz_corrected','rayleigh_corrected')),
]


satellite_data_info = {
    'terra':{'reader':'modis_l1b',
             'load_recipe':terra_load_recipe,
             'instrument':'MODIS',
             'load_composites_recipe':terra_composites,
             },
    'noaa20':{'reader':'viirs_l1b',
              'load_recipe':noaa20_load_recipe,
              'instrument': 'VIIRS',
              'load_composites_recipe':noaa20_composites,
              },
    'sentinel2b':{'reader':'msi_safe',
                  'load_recipe':sentinel2b_load_recipe,
                  'instrument': 'MSI',
                  'load_composites_recipe':sentinel2b_composites,
                  },
}


sat_pixel_class_points = {
    'terra':{'class1_lon_lat':(),
             'class2_lon_lat':(),},
    'noaa20':[],
    'sentinel2b':[],
}
