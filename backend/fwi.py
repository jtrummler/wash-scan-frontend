# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['fwi']:
    print('run fwi')
    
    import os
    import pandas as pd
    import geopandas as gpd
    import rasterio.mask
    import rasterio
    from shapely.geometry import box
    from pathlib import Path
    import glob
    import numpy as np
    from rasterio.crs import CRS
    from os.path import exists

    # SET UP ##############################################
    
    # load city inputs files, to be updated for each city scan
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_name_l = city_inputs['city_name'].replace(' ', '_').lower()

    # load global inputs, such as data sources that generally remain the same across scans
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # Read AOI shapefile --------
    print('read AOI shapefile')
    # transform the input shp to correct prj (epsg 4326)
    aoi_file = gpd.read_file(city_inputs['AOI_path']).to_crs(epsg = 4326)

    # Define output folder ---------
    output_folder = Path('../mnt/city-directories/02-process-output')

    if not exists(output_folder):
        os.mkdir(output_folder)


    # PARAMETERS #######################################
    aoi_buff = aoi_file.buffer(2).total_bounds
    features = gpd.GeoDataFrame({'geometry': box(*aoi_buff)}, index = [0], crs = CRS.from_epsg(4326)).geometry


    # PROCESSING ###################################
    if (not exists(output_folder / f'{city_name_l}_fwi.tif')) and (not exists(output_folder / f'{city_name_l}_fwi.csv')):
        fwi_raster_dict = {}

        # clip raster and store in dict --------------------
        for year in range(global_inputs['fwi_first_year'], global_inputs['fwi_last_year'] + 1):
            for r in glob.glob(f"{global_inputs['fwi_source']}/FWI.GEOS-5.Daily.Default.{year}*.tif"):
                with rasterio.open(r, 'r+') as src:
                    src.crs = CRS.from_epsg(4326)
                    
                    out_image, out_transform = rasterio.mask.mask(
                        src, features, all_touched = True, crop = True)
                    out_meta = src.meta.copy()
                
                if np.nansum(out_image) != 0:
                    fwi_raster_dict[r.split('.')[-2][-9:]] = out_image
                
                    out_meta.update({"driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform})
        q99_raster = np.nanpercentile(list(fwi_raster_dict.values()), 98.6, axis = 0)

        with rasterio.open(output_folder / f'{city_name_l}_fwi.tif', 'w', **out_meta) as dest:
            dest.write(q99_raster)
        
        # calculate 95th percentile FWI by week -------------------
        fwi_val_dict = {}

        for i in fwi_raster_dict:
            fwi_val_dict[i] = fwi_raster_dict[i].flatten().tolist()

        df = pd.DataFrame(list(fwi_val_dict.items()), columns=['date_str', 'fwi'])
        df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
        df['week'] = df['date'].dt.isocalendar().week
        df = df.explode('fwi', ignore_index=True)
        week_95th = df.groupby('week').agg(pctile_95 = ('fwi', lambda x: np.nanpercentile(x, 95)))

        week_95th.to_csv(output_folder / f'{city_name_l}_fwi.csv', index=True)
