# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['elevation']:
    print('run gee_elevation')
    
    import ee
    import geopandas as gpd
    from pathlib import Path
    import numpy as np

    # SET UP #########################################
    # load city inputs files, to be updated for each city scan
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_name_l = city_inputs['city_name'].replace(' ', '_').lower()

    # load global inputs
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # set output folder
    output_folder = Path('output')

    # Initialize Earth Engine
    ee.Initialize()

    elevation = ee.Image("USGS/SRTMGL1_003")

    # Read AOI shapefile --------
    aoi_file = gpd.read_file(city_inputs['AOI_path']).to_crs(epsg = 4326)

    # Convert shapefile to ee.Geometry ------------
    jsonDict = eval(aoi_file['geometry'].to_json())

    if len(jsonDict['features']) > 1:
        print('Need to convert polygons into a multipolygon')
        print('or do something else, like creating individual raster for each polygon and then merge')
        exit()

    if len(jsonDict['features'][0]['geometry']['coordinates'][0][0]) > 2:
        jsonDict['features'][0]['geometry']['coordinates'][0] = [x[0:2] for x in jsonDict['features'][0]['geometry']['coordinates'][0]]
    AOI = ee.Geometry.MultiPolygon(jsonDict['features'][0]['geometry']['coordinates'])


    # PROCESSING #####################################
    no_data_val = -9999
    elevation_clip = elevation.clip(AOI).unmask(value = no_data_val, sameFootprint = False)

    # Export results to Google Cloud Storage bucket ------------------
    task0 = ee.batch.Export.image.toCloudStorage(**{'image': elevation_clip,
                                                    'description': f'{city_name_l}_elevation',
                                                    'region': AOI,
                                                    'scale': 30,
                                                    'bucket': global_inputs['cloud_bucket'],
                                                    'maxPixels': 1e9,
                                                    'fileFormat': 'GeoTIFF',
                                                    'formatOptions': {
                                                        'cloudOptimized': True,
                                                        'noData': no_data_val
                                                    }})
    task0.start()
