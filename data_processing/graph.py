import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
import shapely
import argparse
import os

import shapely.wkt

#reads df and converts to gdf with global crs, must convert crs before merging with another dataframe
def convertDfToGdf(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326")
    return gdf

#converts graph to gpd and saves it
def saveGraphToDisk(graph, folderPath, graphName):
    print("Converting graph to gdf...")
    g_nodes, _ = ox.graph_to_gdfs(graph)
    print("Writing gdf to csv...")
    g_nodes.to_csv(f"{folderPath}/{graphName}_nodes.csv")
    print("Done writing gdfs to csv...")
    return g_nodes

#joins crime df with city graph nodes df
def joinCrimeWithCity(crimeDf, cityGraphNodes):
    #first converts to the same crs
    crimeGdf = crimeDf.to_crs("EPSG:31983")
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

def normalize(string):
    string = string.lower()
    string = string.replace(" ", "-")
    string = string.replace("/", "-")
    #subsituti acentos
    string = string.replace("á", "a")
    string = string.replace("é", "e")
    string = string.replace("ú", "u")
    string = string.replace("ó", "o")
    string = string.replace("ç", "c")
    string = string.replace("ã", "a")
    string = string.replace("â", "a")
    string = string.replace("í", "i")
    return string

def createEdgesDf(gdf, indexColumn, kNeighbors=10, maxDist=1000):
    #array to temp store edges
    edgesArray = np.zeros((len(gdf)*kNeighbors, 3), dtype=np.int64)
    #creates dictionary that maps point to geometry
    geoToIndex, indexToGeo = createGeoDicts(gdf, indexColumn)
    #transform geometry so that there are no two points with the same coordinates
    data = gdf['geometry'].unique()
    data = [[x.x, x.y] for x in data]
    #creates tree to index points
    tree = KDTree(data)
    indexInEdgesArray=0
    for i, row in gdf.iterrows():
        currentGeo = row['geometry']
        currentPoint = (currentGeo.x, currentGeo.y)
        #finds k nearest points
        dist, indexes = tree.query(currentPoint, k=kNeighbors+1, distance_upper_bound=maxDist)
        #first point is the point which we used to query
        for j, index in enumerate(indexes):
            if j == 0:
                continue
            #point's distance is greater than max_dist so ignore
            if index == len(tree.data):
                break
            #gets point
            point = tree.data[index]
            point = shapely.Point(point)
            #gets index of gdf which the point represents
            actual_index = geoToIndex[point]
            edgesArray[indexInEdgesArray] = [row[indexColumn], actual_index, dist[j]]
            indexInEdgesArray+=1
    edgesArray = edgesArray[:indexInEdgesArray]
    edgesDf = pd.DataFrame(edgesArray, columns=['u', 'v', 'dist'])
    return edgesDf


#receives crimedf joined with city graph nodes
def createNodesDf(mergedGdf, occurrencesDf=True):
    nodeOccurDf = None
    if occurrencesDf:
        nodeOccurDf = pd.DataFrame(columns=["node_id", "time", "local_subtype"])

    dfSize = len(mergedGdf['osmid'].unique())
    columns=["node_id", "latitude", "longitude", "amount", "dawn", "morning", "afternoon", "night"]
    #index for first time column
    TIME = columns.index("dawn")
    #index for amount colum
    AMOUNT= columns.index("amount")
    GEO = columns.index("latitude")
    #constructed using stolen vehicles dataset from 2017 - 2024
    local_subtype = ['obra-construcao',
    'semaforo-interior-de-veiculo-de-carga',
    'praca-de-pedagio-cabine-posto',
    'mercado',
    'loja-de-material-de-construcao',
    'praca',
    'construcao-abandonada',
    'distribuidora',
    'praca-de-pedagio-interior-de-veiculo-de-carga',
    'ciclofaixa',
    'via-publica-ciclofaixa',
    'praca-interior-de-veiculo-de-carga',
    'tunel-viaduto-ponte-outros',
    'posto-de-auxilio-cabine-posto-escritorio',
    'feira-livre-outros',
    'posto-de-auxilio-outros',
    'interior-de-transporte-coletivo',
    'restaurante',
    'acostamento-outros',
    'restaurante-outros',
    'semaforo-outros',
    'interior-de-veiculo-de-carga',
    'bar-botequim',
    'posto-de-auxilio',
    'tunel-viaduto-ponte-transeunte',
    'metroviario-e-ferroviario-metropolitano',
    'rodoviario-estacioanamento',
    'tunel-viaduto-ponte-interior-de-veiculo-de-carga',
    'mecanica-borracharia',
    'oficina',
    'via-publica',
    'posto-de-auxilio-interior-de-veiculo-de-carga',
    'lote-de-terreno',
    'sala-de-reuniões-convencões',
    'posto-de-gasolina',
    'praca-de-pedagio-estacionamento',
    'delegacia',
    'hospital-outros',
    'posto-policial-outros',
    'lanchonete-pastelaria-pizzaria-outros',
    'terreno-baldio',
    'padaria-confeitaria',
    'outros-estacionamento',
    'posto-de-auxilio-interior-de-veiculo-particular',
    'praca-de-pedagio-outros',
    'delegacia-distrito-policial',
    'outros',
    'posto-de-fiscalizacao-outros',
    'transportadora',
    'conveniência',
    'armazem-emporio',
    'rodoviario-outros',
    'area-de-descanso',
    'fabrica-industria',
    'desmanche',
    'deposito',
    'hospital',
    'area-de-descanso-interior-de-veiculo-particular',
    'atacadista',
    'area-de-descanso-outros',
    'rodoviario',
    'area-comum',
    'rodoviario-garagem',
    'restaurante-estacionamento',
    'aeroportuario',
    'veiculo-em-movimento',
    'tunel-viaduto-ponte',
    'metalurgica-outros',
    'estacionamento',
    'acostamento',
    'transeunte',
    'farmacia-drogaria',
    'agência-outros',
    'interior-veiculo-de-carga',
    'via-publica-transeunte',
    'praca-de-pedagio',
    'praca-outros',
    'portuario-outros',
    'area-de-descanso-transeunte',
    'bar-botequim-outros',
    'estacionamento-particular',
    'lanchonete-pastelaria-pizzaria-estacionamento',
    'parque-bosque-horto-reserva',
    'semaforo-interior-de-veiculo-particular',
    'balanca',
    'interior-de-veiculo-particular',
    'acougue-frigorifico-outros',
    'posto-de-auxilio-estacionamento',
    'acostamento-transeunte',
    'lojas',
    'acougue-frigorifico-camara-frigorifica',
    'area-de-descanso-interior-de-veiculo-de-carga',
    'posto-de-fiscalizacao',
    'acostamento-interior-de-veiculo-de-carga',
    'loteamento',
    'via-publica-interior-de-veiculo-particular',
    'acougue-frigorifico',
    'estacionamento-publico',
    'outros-banheiro',
    'hospital-almoxarifado',
    'via-publica-interior-de-veiculo-de-carga',
    'auto-pecas',
    'balanca-outros',
    'semaforo'

]
    SUBTYPE = len(columns)
    subTypeIndex = {}
    #associates subtype to index in order to encode
    for i, v in enumerate(local_subtype):
        subTypeIndex[v] = i
    rowsNpArray = np.zeros((dfSize, len(columns)+len(local_subtype)))
    #maps node id to index in df
    nodeIdToIndex = {}
    #to keep track of nodes id
    currentIndex = 0
    #loops through rows building the df
    for i, row in mergedGdf.iterrows():
        #creates empty nodeOccurDfRow
        newNodeOccurRow = {"node_id" : "", "time" : "", "local_subtype" : ""}
        newNodeOccurRow['node_id'] = row['osmid']

        #checks if nodeId already in rowsArray
        indexInDf = nodeIdToIndex.get(row['osmid'])
        #must add node entry to array
        if indexInDf == None:
            nodeIdToIndex[row['osmid']] = currentIndex
            indexInDf = currentIndex
            rowsNpArray[indexInDf][0] = row['osmid']
            rowsNpArray[indexInDf][GEO] = row['LATITUDE']
            rowsNpArray[indexInDf][GEO+1] = row['LONGITUDE']
            currentIndex += 1
        #adds crime ocorrence to graph
        rowsNpArray[indexInDf][AMOUNT]+=1
        #parses time
        time = row['HORA_OCORRENCIA']
        if not pd.isnull(time):
            time = int(time[0:2])
            #so we can get the index
            timeShift = time//6
            if timeShift == 0:
                newNodeOccurRow['time'] = "dawn"
            elif timeShift == 1:
                newNodeOccurRow['time'] = "morning"
            elif timeShift == 2:
                newNodeOccurRow['time'] = "afternoon"
            elif timeShift == 3:
                newNodeOccurRow['time'] = "night"
            else:
                newNodeOccurRow['time'] = 'unknown'
            rowsNpArray[indexInDf][TIME + timeShift] += 1
        #parses subtype
        subType = row['DESCR_SUBTIPOLOCAL']
        if not pd.isnull(subType) and subTypeIndex.get(normalize(subType)) is not None:
            encodedIndex = subTypeIndex[normalize(subType)]
            rowsNpArray[indexInDf][SUBTYPE + encodedIndex] += 1
            newNodeOccurRow['local_subtype'] = normalize(subType)
        if occurrencesDf:
            nodeOccurDf = pd.concat([nodeOccurDf, pd.DataFrame([newNodeOccurRow])], ignore_index=True)
            #nodeOccurDf = nodeOccurDf.append(newNodeOccurRow, ignore_index=True)
    #update columns to have descr_subtipolocal
    columns = columns + local_subtype
    nodesDf = pd.DataFrame(rowsNpArray, columns=columns)
    nodesDf['node_id'] = nodesDf['node_id'].astype('int64')
    if occurrencesDf:
        return nodesDf, nodeOccurDf
    return nodesDf

def createCrimeGraphDf(crimeOcorrencesGdf, cityGraphNodes, kNeighbors=10, maxDist=1000):
    print("Merging crimesDf with graph nodes")
    merge = joinCrimeWithCity(crimeOcorrencesGdf, cityGraphNodes)
    print("creating edges df")
    edgesDf = createEdgesDf(merge, "osmid", kNeighbors, maxDist)
    print("creating nodes")
    nodesDf, crimeOcorrencesGdf = createNodesDf(merge, True)
    return nodesDf, edgesDf, crimeOcorrencesGdf



parser = argparse.ArgumentParser(
    prog="Graph builder",
    description="Creates a graph and occurrences df to be used for further processing",
)

path, _= os.path.split(__file__)

#args handling
parser.add_argument("crimeDf", action='store')
parser.add_argument("graph_name")
parser.add_argument('-g', '--graph', action='store')
#downloads graph and plot it
arg = parser.parse_args()
if arg.graph == None:
    print("Download graph")
    originalGraph = ox.graph_from_place("São Paulo", network_type="drive")
    print("Converting to undirected graph")
    undirectedGraph = ox.convert.to_undirected(originalGraph)
    print("Finished converting")
    city_nodes = saveGraphToDisk(undirectedGraph, path + "/../data_processed", "spCidade")
else:
    print('reading nodes file')
    city_nodes = pd.read_csv(arg.graph)
    city_nodes['geometry'] = city_nodes['geometry'].apply(shapely.wkt.loads)
    print('read node files')
    city_nodes = gpd.GeoDataFrame(city_nodes, crs="EPSG:4326")
    
print("Reading Crime df")
crimeDf = pd.read_csv(arg.crimeDf)
print("Using only S.PAULO points")
crimeDf = crimeDf[crimeDf['CIDADE'] == "S.PAULO"]
crimeDf.reset_index(inplace=True)
crimeGdf = convertDfToGdf(crimeDf)
gNodes, gEdges, occurrencesDf = createCrimeGraphDf(crimeGdf, city_nodes)
pathWithName = path + "/../data_processed/" + arg.graph_name
gNodes.to_csv(pathWithName + "_nodes.csv", index=False)
gEdges.to_csv(pathWithName + "_edges.csv", index=False)
occurrencesDf.to_csv(pathWithName + "_occurrences.csv", index=False)




