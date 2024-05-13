# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['nightlight']:
    print('run gee_nightlight')
    
    import ee
    import geopandas as gpd

    # SET UP #########################################
    # load city inputs files, to be updated for each city scan
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_name_l = city_inputs['city_name'].replace(' ', '_').lower()

    # load global inputs
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # Initialize Earth Engine
    ee.Initialize()

    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")

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


    # GEE PARAMETERS #####################################
    # Date filter ----------------------
    max_date = viirs.reduceColumns(ee.Reducer.max(), ["system:time_start"]).get('max')

    NTL_time = ee.DateRange(ee.Date(max_date.getInfo()).advance(-10, 'year'), ee.Date(max_date.getInfo()))

    # Function ------------------------
    def addTime(image):
        return image.addBands(image.metadata('system:time_start').divide(1000 * 60 * 60 * 24 * 365))


    # PROCESSING #########################################
    no_data_val = -9999

    viirs_cf_cvg = viirs.select('avg_rad')
    viirs_with_time = viirs.filterDate(NTL_time).map(addTime)
    viirs_with_time1 = viirs.filterDate(NTL_time).map(addTime)

    linear_fit = viirs_with_time.select(['system:time_start', 'avg_rad']).reduce(ee.Reducer.linearFit()).clip(AOI).unmask(value = no_data_val, sameFootprint = False)
    sum_of_light = viirs_with_time.select(['system:time_start', 'avg_rad']).reduce(ee.Reducer.sum()).clip(AOI).unmask(value = no_data_val, sameFootprint = False)

    # Export results to Google Cloud Storage bucket ------------------
    task0 = ee.batch.Export.image.toCloudStorage(**{
        'image': linear_fit.select('scale'),
        'description': f'{city_name_l}_linfit',
        'bucket': global_inputs['cloud_bucket'],
        'region': AOI,
        'scale': 100,
        'maxPixels': 1e9,
        'fileFormat': 'GeoTIFF',
        'formatOptions': {
            'cloudOptimized': True,
            'noData': no_data_val
        }
    })
    task0.start()

    task1 = ee.batch.Export.image.toCloudStorage(**{
       'image': sum_of_light.select('avg_rad_sum'),
        'description': f'{city_name_l}_avg_rad_sum',
        'bucket': global_inputs['cloud_bucket'],
        'region': AOI,
        'scale': 100,
        'maxPixels': 1e9,
        'fileFormat': 'GeoTIFF',
        'formatOptions': {
            'cloudOptimized': True,
            'noData': no_data_val
        }
    })
    task1.start()
