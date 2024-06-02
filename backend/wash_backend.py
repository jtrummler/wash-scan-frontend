# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['raster_processing']:
    print('run raster_processing')

    import os
    import math
    import csv
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    import glob
    import rioxarray as rxr
    from osgeo import gdal, osr, ogr
    import rasterio.mask
    import rasterio
    import requests
    from pathlib import Path
    from rasterio.merge import merge
    from os.path import exists
    import zipfile
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from shutil import copyfile

# load city and catchment inputs
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_l = city_inputs['city_name'].replace(' ', '_').lower()

# read AOI + catchment
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # Read AOI shapefile --------
    print('read AOI shapefile')
    # transform the input shp to correct prj (epsg 4326)
    aoi_file = gpd.read_file(city_inputs['AOI_path']).to_crs(epsg = 4326)
    aoi_features = aoi_file.geometry
    aoi_bounds = aoi_file.bounds

    # Read AOI shapefile --------
    print('read catchment zone')
    # transform the input shp to correct prj (epsg 4326)
    cat_file = gpd.read_file(city_inputs['catchment_path']).to_crs(epsg = 4326)
    cat_features = cat_file.geometry
    cat_bounds = cat_file.bounds

    # file paths
    a_path = 'C:/Users/jtrum/Wash/data_processing/a_rasters/'
    b_path = 'C:/Users/jtrum/Wash/data_processing/b_shps/'
    c_path = 'C:/Users/jtrum/Wash/data_processing/c_geojsons/'

# FATHOM
    def fathom_rasters(file_list, cat):
        for file_path in file_list:
            scenario_name = os.path.basename(file_path).split('.')[0] # Extract base name
            scenario_raster = rxr.open_rasterio(file_path, masked=True).squeeze()
            scenario_raster = scenario_raster.rio.reproject(cat.crs) # Project to AOI CRS
            scenario_raster_crop = scenario_raster.rio.clip_box(*cat.total_bounds, crs=cat.crs).squeeze()
            scenario_raster_crop = scenario_raster_crop.where(scenario_raster_crop > 0.15) # Set threshold for binarization
            scenario_raster_crop = scenario_raster_crop.where(scenario_raster_crop != 999) # Drop values for ocean
            scenario_raster_crop.rio.to_raster(os.path.join(a_path, 'Fathom', f'{scenario_name}_crop_{city_l}.tif')) # Export tif
            ###
            raster = gdal.Open(os.path.join(a_path, 'Fathom', f'{scenario_name}_crop_{city_l}.tif')) # Read in tif through gdal to process as a shp file
            band = raster.GetRasterBand(1)
            band.ReadAsArray()
            proj = raster.GetProjection()
            shp_proj = osr.SpatialReference()
            shp_proj.ImportFromWkt(proj)
            output_file = os.path.join(b_path, 'Fathom', f'{scenario_name}_{city_l}.shp')
            call_drive = ogr.GetDriverByName('ESRI Shapefile')
            create_shp = call_drive.CreateDataSource(output_file)
            shp_layer = create_shp.CreateLayer('pct', srs=shp_proj)
            new_field = ogr.FieldDefn(str('ID'), ogr.OFTInteger)
            shp_layer.CreateField(new_field)
            gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)
            create_shp.Destroy()
            raster = None
            ###
            scenario_shp = gpd.read_file(os.path.join(b_path, 'Fathom', f'{scenario_name}_{city_l}.shp')) # Load in shp file
            scenario_shp.crs = cat.crs # Ensure = AOI CRS
            scenario_shp = gpd.clip(scenario_shp, cat) # Clip to AOI
            scenario_shp['ID'] = scenario_shp['ID'].apply(lambda x: 'flood' if x >= 0 else 'not flood') # Remove whatever value represents the clipping intersection
            scenario_shp = scenario_shp[scenario_shp.ID != 'not flood']
            scenario_shp.to_file(os.path.join(c_path, 'Fathom_Individual', f'{scenario_name}_{city_l}.geojson'), driver='GeoJSON') # Export as geojson

    def load_geojsons(file_names, c_path):
        file_paths = [os.path.join(c_path, 'Fathom_Individual', f'{file_name}_{city_l}.geojson') for file_name in file_names]
        dataframes = [gpd.read_file(file_path) for file_path in file_paths]
        return dataframes

    def create_10pct(gdf1, gdf2, name):
        pct_10 = gpd.overlay(gdf1, gdf2, how='intersection')
        pct_10.to_file(f'{c_path}Fathom_Over10pct_{name}_{city_l}.geojson', driver='GeoJSON')
        return pct_10

    def create_1to10(gdf1, gdf2, gdf3, name):
        intersect_1_2 = gpd.overlay(gdf1, gdf2, how='intersection')
        intersect_1_3 = gpd.overlay(gdf1, gdf3, how='intersection')
        intersect_2_3 = gpd.overlay(gdf2, gdf3, how='intersection')
        intersected_geometries = gpd.GeoDataFrame(pd.concat([intersect_1_2, intersect_1_3, intersect_2_3], ignore_index=True))
        geometry_counts = intersected_geometries.groupby('geometry').size().reset_index(name='count')
        filtered_geometries = geometry_counts[geometry_counts['count'] >= 1]['geometry']
        pct_1to10 = gpd.GeoDataFrame(geometry=filtered_geometries)
        pct_1to10.to_file(f'{c_path}Fathom_1to10pct_{name}_{city_l}.geojson', driver='GeoJSON')
        return pct_1to10

    def create_under1(gdf1, gdf2, gdf3, gdf4, gdf5, name):
        intersect_1_2 = gpd.overlay(gdf1, gdf2, how='intersection')
        intersect_1_3 = gpd.overlay(gdf1, gdf3, how='intersection')
        intersect_1_4 = gpd.overlay(gdf1, gdf4, how='intersection')
        intersect_1_5 = gpd.overlay(gdf1, gdf5, how='intersection')
        intersect_2_3 = gpd.overlay(gdf2, gdf3, how='intersection')
        intersect_2_4 = gpd.overlay(gdf2, gdf4, how='intersection')
        intersect_2_5 = gpd.overlay(gdf2, gdf5, how='intersection')
        intersect_3_4 = gpd.overlay(gdf3, gdf4, how='intersection')
        intersect_3_5 = gpd.overlay(gdf3, gdf5, how='intersection')
        intersect_4_5 = gpd.overlay(gdf4, gdf5, how='intersection')
        intersected_geometries = gpd.GeoDataFrame(pd.concat([intersect_1_2, intersect_1_3, intersect_1_4, intersect_1_5,
                                                            intersect_2_3, intersect_2_4, intersect_2_5,
                                                            intersect_3_4, intersect_3_5, intersect_4_5],
                                                            ignore_index=True))
        geometry_counts = intersected_geometries.groupby('geometry').size().reset_index(name='count')
        filtered_geometries = geometry_counts[geometry_counts['count'] >= 1]['geometry']
        pct_under1 = gpd.GeoDataFrame(geometry=filtered_geometries)
        pct_under1.to_file(f'{c_path}Fathom_Under1pct_{name}_{city_l}.geojson', driver='GeoJSON')
        return pct_under1

### OPEN STREET MAP
tags_list = [
    {'landuse': ['reservoir', 'basin']},
    {'amenity': ['drinking_water', 'watering_place', 'water_point']},
    {'man_made': ['water_well', 'water_tower', 'water_works', 'reservoir_covered', 'storage_tank', 'monitoring_station', 'wastewater_plant', 'watermill', 'pipeline']}
]

def extract_water_infra(tags_list, cat):
    for tags in tags_list:
        data = ox.geometries_from_polygon(cat.geometry[0], tags=tags)
        globals()[list(tags.keys())[0]] = data[['name', list(tags.keys())[0], 'geometry']]

    w_inf = pd.concat([landuse, amenity, man_made], ignore_index=True)
    w_inf = w_inf[['landuse', 'amenity', 'man_made', 'geometry']]
    w_inf['Feature'] = w_inf.apply(lambda x: x[['landuse', 'amenity', 'man_made']].first_valid_index(), axis=1)
    w_inf['Feature'] = w_inf.apply(lambda x: x[x['Feature']], axis=1)
    w_inf['Feature'] = w_inf['Feature'].str.title()
    w_inf['Feature'] = w_inf['Feature'].str.replace('_', ' ')
    w_inf = w_inf[['Feature', 'geometry']]
    w_inf['geometry'] = w_inf['geometry'].centroid
    w_inf.to_file(f'{c_path}water_infrastructure_{city_l}.geojson', driver='GeoJSON')
    return w_inf

def extract_schools_hospitals(tags_list, cat, aoi):
    aoi_polygon = aoi.geometry.unary_union
    data_list = []
    
    for tags in tags_list:
        data = ox.geometries_from_polygon(cat.geometry[0], tags=tags)
        data = data[['amenity', 'geometry']]
        data_list.append(data)
    
    sch_hos = gpd.GeoDataFrame(pd.concat(data_list, ignore_index=True), geometry='geometry', crs=aoi.crs)
    sch_hos['amenity'] = sch_hos['amenity'].str.capitalize()
    sch_hos['geometry'] = sch_hos['geometry'].centroid
    sch_hos['AOI'] = sch_hos['geometry'].apply(lambda geom: 'AOI' if geom.within(aoi_polygon) else 'Catchment')
    sch_hos['ID'] = sch_hos.index + 1
    sch_hos.to_file(os.path.join(c_path, f'schools_hospitals_{city_l}.geojson'), driver='GeoJSON')
    return sch_hos


def raster_processing(data, column, outName):
    raster = gdal.Open(data)
    band = raster.GetRasterBand(1) # can use whichever band, but 1 works fine in most cases
    band.ReadAsArray()

    proj = raster.GetProjection()
    shp_proj = osr.SpatialReference()
    shp_proj.ImportFromWkt(proj)

    output_file = f'{b_path}{outName}_{city_l}.shp' # change to your output path
    call_drive = ogr.GetDriverByName('ESRI Shapefile') # exports as shp
    create_shp = call_drive.CreateDataSource(output_file)
    shp_layer = create_shp.CreateLayer('pct', srs=shp_proj) # name of layer in shp
    new_field = ogr.FieldDefn(str(column), ogr.OFTReal)# will become the column name in geodataframe that raster values are extrapolated from
    new_field.SetPrecision(6)
    shp_layer.CreateField(new_field)

    gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)
    create_shp.Destroy()
    raster = None

def clipping(shp, aoi, name, column, val):
    shp = gpd.read_file(shp).to_crs('EPSG:4326')
    aoi = gpd.read_file(aoi).to_crs('EPSG:4326')
    clipped = gpd.clip(shp, aoi)
    # drop values that are null (eg. -9999)
    clipped = clipped[clipped[column] != val]
    clipped.to_file(f'{c_path}{name}_{city_l}.geojson', driver='GeoJSON')

def builtup_union(df):
    wsf_union = gpd.GeoSeries(df.unary_union)
    wsf_union = gpd.GeoDataFrame(wsf_union)
    wsf_union.columns = ['geometry']
    wsf_union.set_geometry('geometry', inplace=True)
    wsf_union['idx'] = 'Built-Up'
    wsf_union.crs = {'init': 'epsg:4326'}
    wsf_union.to_file(f'{c_path}wsf_final_{city_l}.geojson', driver='GeoJSON')
    return wsf_union