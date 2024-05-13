# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['landcover']:
    print('run gee_landcover')
    
    import ee
    import geopandas as gpd
    import csv
    from pathlib import Path

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

    lc = ee.ImageCollection('ESA/WorldCover/v200').first()

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
    lc_aoi = lc.clip(AOI)

    # Calculate stats -----------------------
    # Calculate the pixel counts for each land cover type
    lc_hist = lc_aoi.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=AOI,
        scale=10,
        maxPixels=1e9
    )
    lc_hist1 = lc_hist.getInfo()['Map']
    
    # Create a dictionary to store the counts
    counts_dict = {}

    # Create value dictionary
    class_values = {
        10: 'Tree cover',
        20: 'Shrubland',
        30: 'Grassland',
        40: 'Cropland',
        50: 'Built-up',
        60: 'Bare / sparse vegetation',
        70: 'Snow and ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        95: 'Mangroves',
        100: 'Moss and lichen'
    }

    # Populate the dictionary with counts for each land cover type
    for class_val in class_values:
        class_name = class_values[class_val]
        count = lc_hist1.get(str(class_val), 0)
        counts_dict[class_name] = count

    # Write the counts dictionary to a CSV file
    with open(output_folder / f'{city_name_l}_lc.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Land Cover Type', 'Pixel Count'])
        writer.writeheader()
        for class_name, count in counts_dict.items():
            writer.writerow({'Land Cover Type': class_name, 'Pixel Count': count})

    # Export results to Google Cloud Storage bucket ------------------
    task0 = ee.batch.Export.image.toCloudStorage(**{'image': lc_aoi,
                                                    'description': f'{city_name_l}_lc',
                                                    'region': AOI,
                                                    'scale': 10,
                                                    'bucket': global_inputs['cloud_bucket'],
                                                    'maxPixels': 1e9})
    task0.start()
