

# tokens for aquiring data from earthdata and europeans
ed_token =
eo_token =


def download_modis_granules(time_start, time_end, satellite_name):
    # download granules based on temporal bounds for modis
    # satellite name important for which satellite to download from
    return raw_modis_data_path

def download_modis_geolocation(time_start, time_end, satellite_name):
    return raw_modis_geolocation_path

def download_viirs_granules(time_start, time_end, satellite_name):
    return raw_viirs_data_path

def download_msi_granules(time_start, time_end, satellite_name):
    return raw_msi_data_path



##### satpy find_files_and_readers
def save_modis_dataset(granules):
    myfiles = find_files_and_readers(base_dir=DATA_DIR,
                                     sensor="modis",
                                     start_time=datetime(2016, 3, 30, 11, 40),
                                     end_time=datetime(2016, 3, 30, 12, 5),
                                     reader='modis_l1b')
    scn = Scene(filenames=myfiles)

    scn.save_dataset()

    save(combined_raw_modis_granules)

def save_viirs_granules(granules):
    save(combined_raw_viirs_granules)

def save_msi_granules(granules):
    save(combined_raw_msi_data)


