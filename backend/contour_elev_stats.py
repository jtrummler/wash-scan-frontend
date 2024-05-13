# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['raster_processing'] and menu['elevation']:
    print('run contour')
    
    # SET UP ##############################################

    from os.path import exists
    from pathlib import Path

    # load city inputs files, to be updated for each city scan
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_name_l = city_inputs['city_name'].replace(' ', '_').lower()

    # load global inputs, such as data sources that generally remain the same across scans
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # Define output folder ---------
    output_folder = Path('../mnt/city-directories/02-process-output')

    # Check if elevation raster exists ------------
    if not exists(output_folder / f'{city_name_l}_elevation.tif'):
        print('cannot generate contour lines or elevantion stats because elevation raster does not exist')
        exit()
    
    import os
    import math
    import csv
    import geopandas as gpd
    import numpy as np
    import rasterio


    # CONTOUR ##############################################
    print('generate contour lines')

    try:
        # Determine contour interval ---------------------
        with rasterio.open(output_folder / f'{city_name_l}_elevation.tif') as src:
            elev_array = src.read(1)
            elev_array = elev_array[elev_array != -9999]
            max_elev = np.nanmax(elev_array)
            min_elev = np.nanmin(elev_array)
        
        elev_diff = max_elev - min_elev
        
        # TODO: ask copilot about what's a good contour interval

        # Generate contour lines -----------------------
        # TODO
    except:
        print('generate contour lines failed')
    

    # ELEVATION STATS ##############################################
    print('calculate elevation stats')

    try:
        # Calculate equal interval bin edges
        num_bins = 5
        bin_edges = np.linspace(min_elev, max_elev, num_bins + 1)
        
        # TODO: use max_elev and min_elev to determine rounding
        # Round bin edges to the nearest 1
        bin_edges = np.round(bin_edges, 0)
        
        # Calculate histogram
        hist, _ = np.histogram(elev_array, bins = bin_edges)
        
        # Write bins and hist to a CSV file
        with open(output_folder / f'{city_name_l}_elevation.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Bin', 'Count'])
            for i, count in enumerate(hist):
                bin_range = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                writer.writerow([bin_range, count])
    except:
        print('calculate elevation stats failed')