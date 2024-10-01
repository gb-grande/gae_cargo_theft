import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
import shapely

#reads df and converts to gdf with global crs, must convert crs before merging with another dataframe
def convertDfToGdf(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326")
    return gdf

#joins crime df with city graph nodes df
def joinCrimeWithCity(crimeGdf, cityGraphNodes):
    #first converts to the same crs
    crimeDf = crimeDf.to_crs("EPSG:31983")
    cityGraphNodes = cityGraphNodes.to_crs("EPSG:31983")
    merge =crimeGdf.sjoin_nearest(cityGraphNodes, distance_col="distance")
    #if there are points which are equidistant, this operation guarantee there will be only one point assigned to a node
    merge = merge[~merge.index.duplicated(keep="first")]
    return merge


#creates dictionary that maps point to geometry
def createGeoDicts(gdf, indexColumn):
    geoToIndex = {}
    indexToGeo = {}
    for i in range(len(gdf)):
        geoToIndex[gdf['geometry'][i]] = gdf[indexColumn][i]
        indexToGeo[gdf[indexColumn][i]] = gdf['geometry'][i]
    return geoToIndex, indexToGeo

def createEdgesDf(gdf, indexColumn, k_neighbors=3):
    #array to temp store edges
    edgesArray = np.zeros((len(gdf)*k_neighbors, 3), dtype=np.int64)
    #creates dictionary that maps point to geometry
    geoToIndex, indexToGeo = createGeoDicts(gdf, indexColumn)
    #transform geometry so that there are no two points with the same coordinates
    data = gdf['geometry'].unique()
    data = [[x.x, x.y] for x in data]
    #creates tree to index points
    tree = KDTree(data)
    for i, row in gdf.iterrows():
        currentGeo = row['geometry']
        currentPoint = (currentGeo.x, currentGeo.y)
        #finds k nearest points
        dist, indexes = tree.query(currentPoint, k=k_neighbors+1)
        #first point is the point which we used to query
        for j, index in enumerate(indexes):
            if j == 0:
                continue
            #gets point
            point = tree.data[index]
            point = shapely.Point(point)
            #gets index of gdf which the point represents
            actual_index = geoToIndex[point]
            edgesArray[i*k_neighbors+j-1] = [row[indexColumn], actual_index, dist[j]]
    edgesDf = pd.DataFrame(edgesArray, columns=['u', 'v', 'dist'])
    return edgesDf

#converts graph to gpd and saves it
def saveGraphToDisk(graph, graphName):
    print("Converting graph to gdf...")
    g_nodes, g_edges = ox.graph_to_gdfs(graph)
    print("Writing gdfs to csv...")
    g_nodes.to_csv(f"{graphName}_nodes.csv")
    g_edges.to_csv(f"{graphName}_edges.csv")
    print("Done writing gdfs to csv...")


#downloads graph and plot it

print("Download graph")
originalGraph = ox.graph_from_place("São Paulo", network_type="drive")
fig, ax = ox.plot_graph(originalGraph)
print("Converting to undirected graph")
undirectedGraph = ox.convert.to_undirected(originalGraph)
print("Finished converting")

saveGraphToDisk(undirectedGraph, "spCidade")