# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['ndmi']:
    print('run gee_ndmi')
    
    import ee
    import math
    import geopandas as gpd
    import xarray as xr
    import numpy as np
    import datetime as dt
    from pathlib import Path
    from os.path import exists

    # SET UP #########################################
    # load city inputs files, to be updated for each city scan
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_name_l = city_inputs['city_name'].replace(' ', '_').lower()

    # load global inputs
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # set output folder
    output_folder = Path('../mnt/city-directories/02-process-output')

    # Initialize Earth Engine
    ee.Initialize()

    # Read AOI shapefile --------
    aoi_file = gpd.read_file(city_inputs['AOI_path']).to_crs(epsg = 4326)
    centroid = aoi_file.centroid

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
    if not exists(output_folder / f'{city_name_l}_hottest_months.txt'):
        temp_dict = {}
        for i in range(math.floor(city_inputs['first_year'] / 10) * 10, math.ceil(city_inputs['last_year'] / 10) * 10, 10):
            if i >= 2020:
                continue
            nc = xr.open_dataset(f"{global_inputs['temperature_source']}/cru_ts4.06.{i+1}.{i+10}.tmp.dat.nc")

            for month in range(1, 13):
                temp_dict[month] = []
                for year in range(max(i, city_inputs['first_year']), min(i+11, city_inputs['last_year'])):
                    time = str(year) + '-' + str(month) + '-15'
                    val = nc.sel(lon = centroid.x[0], lat = centroid.y[0], time = time, method = 'nearest')['tmp'].to_dict()['data']
                    temp_dict[month].append(val)
                temp_dict[month] = np.nanmean(temp_dict[month])
        
        avg_temp_dict = {}
        for month in range(1, 13):
            avg_temp_dict[(month-1)%12+1] = np.nanmean([temp_dict[(month-1)%12+1], temp_dict[(month)%12+1], temp_dict[(month+1)%12+1]])

        first_hot_months = max(zip(avg_temp_dict.values(), avg_temp_dict.keys()))[1]
        hottest_months = [first_hot_months, (first_hot_months)%12+1, (first_hot_months+1)%12+1]
        
        # Write hottest months to text ---------------------------
        with open(output_folder / f'{city_name_l}_hottest_months.txt', 'w') as file:
            # Write each number to the file on a new line
            for number in hottest_months:
                file.write(f"{number}\n")
    
    else:
        hottest_months = []
        with open(output_folder / f'{city_name_l}_hottest_months.txt') as file:
            for line in file:
                # Convert each line to an integer and append to the list
                hottest_months.append(int(line.strip()))
    
    # Date filter -----------------
    def ee_filter_month(month):
        if 1 <= month <= 11:
            return [ee.Filter.date(f'{year}-{str(month).zfill(2)}-01', f'{year}-{month+1}-01') for year in range(city_inputs['first_year'], city_inputs['last_year'] + 1)]
        elif month == 12:
            return [ee.Filter.date(f'{year}-12-01', f'{year+1}-01-01') for year in range(city_inputs['first_year'], city_inputs['last_year'] + 1)]
        else:
            return
    
    range_list0 = ee_filter_month(hottest_months[0])
    range_list1 = ee_filter_month(hottest_months[1])
    range_list2 = ee_filter_month(hottest_months[2])

    rangefilter = ee.Filter.Or(range_list0 + range_list1 + range_list2)

    NDMI_last_year = ee.DateRange(f'{str(dt.date.today().year-1)}-01-01', f'{str(dt.date.today().year)}-01-01')

    # Functions --------------------
    def maskS2clouds(image):
        qa = image.select('QA60')

        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11

        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

        return image.updateMask(mask).divide(10000)


    # PROCESSING #########################################
    no_data_val = -9999

    s2a_Season = ee.ImageCollection('COPERNICUS/S2').filterBounds(AOI).filter(rangefilter).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).map(maskS2clouds)
    S2a_RecentAnnual = ee.ImageCollection('COPERNICUS/S2').filterBounds(AOI).filterDate(NDMI_last_year).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).map(maskS2clouds)

    s2a_med_Season = s2a_Season.median().clip(AOI).unmask(value = no_data_val, sameFootprint = False)
    s2a_med_RecentAnnual = S2a_RecentAnnual.median().clip(AOI).unmask(value = no_data_val, sameFootprint = False)

    ndmi_Season = s2a_med_Season.normalizedDifference(['B8', 'B11']).rename('NDMI')
    ndmi_recentannual = s2a_med_RecentAnnual.normalizedDifference(['B8', 'B11']).rename('NDMI_Annual')

    # Export results to Google Cloud Storage bucket ------------------
    task0 = ee.batch.Export.image.toCloudStorage(**{
        'image': ndmi_Season,
        'description': f'{city_name_l}_NDMI_Season',
        'bucket': global_inputs['cloud_bucket'],
        'region': AOI,
        'scale': 10,
        'maxPixels': 1e9,
        'fileFormat': 'GeoTIFF',
        'formatOptions': {
            'cloudOptimized': True,
            'noData': no_data_val
        }
    })
    task0.start()

    task1 = ee.batch.Export.image.toCloudStorage(**{
        'image': ndmi_recentannual,
        'description': f'{city_name_l}_NDMI_Annual',
        'bucket': global_inputs['cloud_bucket'],
        'region': AOI,
        'scale': 10,
        'maxPixels': 1e9,
        'fileFormat': 'GeoTIFF',
        'formatOptions': {
            'cloudOptimized': True,
            'noData': no_data_val
        }
    })
    task1.start()
