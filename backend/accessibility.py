# DETERMINE WHETHER TO RUN THIS SCRIPT ##############
import yaml

# load menu
with open("../mnt/city-directories/01-user-input/menu.yml", 'r') as f:
    menu = yaml.safe_load(f)

if menu['accessibility']:
    print('run accessibility')
    
    from os.path import exists
    import os, sys
    import geopandas as gpd
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import box
    from pathlib import Path
    import pickle

    # SET UP #########################################
    # load city inputs files, to be updated for each city scan
    with open("../mnt/city-directories/01-user-input/city_inputs.yml", 'r') as f:
        city_inputs = yaml.safe_load(f)

    city_name_l = city_inputs['city_name'].replace(' ', '_').lower()

    # load global inputs, such as data sources that generally remain the same across scans
    with open("global_inputs.yml", 'r') as f:
        global_inputs = yaml.safe_load(f)

    # import GOSTnets -----------
    sys.path.append(global_inputs['gostnets_path'])
    import GOSTnets as gn
    from GOSTnets.fetch_pois import OsmObject  # TODO: fix GOSTnets

    # Read AOI shapefile --------
    # transform the input shp to correct prj (epsg 4326)
    aoi_orig = gpd.read_file(city_inputs['AOI_path']).to_crs(epsg = 4326)
    # buffer AOI by 5% of its width to capture roads immedately outside of city boundaries
    # TODO: determine whether to put this 5% buffer distance in one of the yaml files
    buff_dist = (aoi_orig.total_bounds[2] - aoi_orig.total_bounds[0]) * 0.05
    aoi_file = aoi_orig.buffer(buff_dist)
    features = aoi_file.geometry
    
    # Define output folder ---------
    output_folder = Path('../mnt/city-directories/02-process-output')

    if not exists(output_folder):
        os.mkdir(output_folder)


    # EXTRACT OSM POI ##############################
    queries = {}
    if ('osm_query' in city_inputs) and bool(city_inputs['osm_query']):
        for tags in city_inputs['osm_query'].items():
            if not exists(output_folder / f'{city_name_l}_osm_{tags[0]}' / f'{city_name_l}_osm_{tags[0]}.shp'):
                # create the OsmObject
                queries[tags[0]] = OsmObject(f'{tags[0]}', features[0], tags[1])
    
    if bool(global_inputs['osm_query']):
        for tags in global_inputs['osm_query'].items():
            if not tags[0] in queries:
                if not exists(output_folder / f'{city_name_l}_osm_{tags[0]}' / f'{city_name_l}_osm_{tags[0]}.shp'):
                    # create the OsmObject
                    queries[tags[0]] = OsmObject(f'{tags[0]}', features[0], tags[1])
    
    for query in queries.items():
        try:
            result = query[1].GenerateOSMPOIs()
        
            # if query is not empty
            if result.empty == False:
                # try:
                query[1].RemoveDupes(0.0005)
                
                if 'name' in query[1].df.columns:
                    query_results = query[1].df[['amenity','geometry','name']]
                else:
                    query_results = query[1].df[['amenity','geometry']]

                # convert to GeoDataFrame
                query_results_gpd = gpd.GeoDataFrame(query_results, crs = "epsg:4326", geometry = 'geometry')
                query_results_gpd_shp = f'{city_name_l}_osm_{query[0]}'
                if not exists(output_folder / query_results_gpd_shp):
                    os.mkdir(output_folder / query_results_gpd_shp)
                query_results_gpd.to_file(output_folder / query_results_gpd_shp / f'{query_results_gpd_shp}.shp')
        except:
            pass


    # PROCESS ROADS ##############################
    road_graph = output_folder / f"{city_name_l}_osm_roads.pickle"

    if not exists(road_graph):
        extent = box(*aoi_file.total_bounds)
        G = ox.graph_from_polygon(extent, network_type = 'drive_service')
        # This is how time is calculated from the OSMNX length attribute
        G = gn.convert_network_to_time(G, 'length')  # default walk_speed = 4.5 (km/h)
        # save the largest subgraph
    
        # compatible with NetworkX 2.4
        list_of_subgraphs = list(G.subgraph(c).copy() for c in nx.strongly_connected_components(G))
        max_graph = None
        max_edges = 0
        for i in list_of_subgraphs:
            if i.number_of_edges() > max_edges:
                max_edges = i.number_of_edges()
                max_graph = i

        # set graph equal to the largest sub-graph
        G = max_graph

        with open(road_graph, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(road_graph, 'rb') as f:
            G = pickle.load(f)
        
        G = gn.convert_network_to_time(G, 'length')

    roads = gn.edge_gdf_from_graph(G)

    def replace_hwy(x):
        if isinstance(x, list):
            x = x[0]
        return x

    roads['highway'] = roads.apply(lambda x: replace_hwy(x['highway']), axis = 1)

    if not exists(output_folder / f'{city_name_l}_osm_roads' / f'{city_name_l}_osm_roads.shp'):
        roads = roads[['length','time','mode','geometry']]
        roads_shp = f'{city_name_l}_osm_roads'
        if not exists(output_folder / roads_shp):
            os.mkdir(output_folder / roads_shp)
        roads.to_file(output_folder / roads_shp / f'{roads_shp}.shp')


    # COMPILE ISOCHRONE DICTIONARY #################
    if ('isochrone' in city_inputs) and bool(city_inputs['isochrone']):
        isochrones = city_inputs['isochrone']
    elif bool(global_inputs['isochrone']):
        isochrones = global_inputs['isochrone']
    else:
        print('No isochrone parameters found. Ending accessibility analysis.')
        quit()


    # SNAP POI TO ROADS ############################
    snapped_destinations_dict = {}
    for results_gpd in global_inputs['isochrone']:
        snapped_destinations = gn.pandana_snap(G, gpd.read_file(output_folder / f'{city_name_l}_osm_{results_gpd}' / f'{city_name_l}_osm_{results_gpd}.shp'))
        snapped_destinations_dict[results_gpd] = list(snapped_destinations['NN'].unique())


    # PROCESS ISOCHRONES ###########################
    # find graph utm zone
    G_utm = gn.utm_of_graph(G)

    def isochrone_processing(amenity_type):
        amenity_threshold_list = global_inputs['isochrone'].get(amenity_type)
        if amenity_threshold_list == None:
            return "Amenity type not found"
        # if no destinations for amenity type exist
        if snapped_destinations_dict.get(amenity_type) == None:
            print(f"no destinations for {amenity_type} exist")
        else:
            for threshold in amenity_threshold_list:
                gdf_out2_shp = f'{city_name_l}_accessibility_{amenity_type}_{threshold}m'
                if not exists(output_folder / gdf_out2_shp / f'{gdf_out2_shp}.shp'):
                    print(threshold)
                    iso_gdf = gn.make_iso_polys(G, snapped_destinations_dict[amenity_type], [threshold], edge_buff = 300, node_buff = 300, weight = 'length', measure_crs = G_utm)
                    dissolved = iso_gdf.dissolve(by = "thresh")
                    gdf_out = dissolved.explode(index_parts = True)
                    gdf_out2 = gdf_out.reset_index()
                    # save file
                    if not exists(output_folder / gdf_out2_shp):
                        os.mkdir(output_folder / gdf_out2_shp)
                    gdf_out2.to_file(output_folder / gdf_out2_shp / f'{gdf_out2_shp}.shp')
            
    for key in global_inputs['isochrone']:
        isochrone_processing(key)
