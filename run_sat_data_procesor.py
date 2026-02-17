import numpy as np
import xarray as xr
from satpy import Scene, find_files_and_readers, DataQuery
from datetime import datetime
import os
from pathlib import Path
from satellite_data_processor import *
from sat_info import *
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.CRITICAL)

# logging.getLogger("satpy").setLevel(logging.ERROR)

DATA_DIR = Path("~/Downloads/sars_p1_data").expanduser()
save_path = Path("~/Downloads/sars_p1_data/processed_output").expanduser()

# Create symlink to fix the naming issue ... you may need to change this depending on your system
pyspectral_dir = Path.home() / 'Library/Application Support/pyspectral'
source = pyspectral_dir / 'rsr_modis_EOS-Terra.h5'
target = pyspectral_dir / 'rsr_modis_Terra.h5'

if source.exists() and not target.exists():
    os.symlink(source, target)
    print(f"Created symlink: {target} -> {source}")
else:
    print(f"Source exists: {source.exists()}, Target exists: {target.exists()}")


terra_dir = DATA_DIR / 'terra-modis'
noaa20_dir = DATA_DIR / 'noaa20-viirs'
sentinel2b_dir = DATA_DIR / 'sentinel2b-msi'

granules = {
    "terra": [sorted(map(str, terra_dir.rglob('*2024129*'))),
              sorted(map(str, terra_dir.rglob('*2025122*'))),
              sorted(map(str, terra_dir.rglob('*2025206*')))],
    "noaa20": [sorted(map(str, noaa20_dir.rglob('*2024129*'))),
               sorted(map(str, noaa20_dir.rglob('*2025122*'))),
               sorted(map(str, noaa20_dir.rglob('*2025206*')))],
    "sentinel2b": [[f] for f in sorted(map(str, sentinel2b_dir.glob('*')))[1:]],
}

for sat_name, data_info in satellite_data_info.items():
    for granule_files in granules[sat_name]:

        _gf = granule_files
        if sat_name == "sentinel2b":
            granule_files = find_files_and_readers(base_dir = granule_files[0], reader="msi_safe")

        scene = Scene(filenames=granule_files, reader=data_info['reader'])

        identifier = extract_identifier(_gf)
        print(sat_name, '      ',identifier)
        process_satellite_data(
                scene = scene,
                identifier=identifier,
                satellite_name = sat_name,
                satellite_instrument=data_info['instrument'],
                load_recipes = data_info['load_recipe'],
                load_composites_recipe=data_info['load_composites_recipe'],
                auto_correction = False,
                save_path = save_path,
                correction_type = "both",
                satpy_resample_option = "native",
                target_area = None,
            )
print("COMPLETE!!!!!!!!!!!!!!!!!!!!!!")