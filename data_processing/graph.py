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

def createEdgesDf(gdf, indexColumn, kNeighbors=3):
    #array to temp store edges
    edgesArray = np.zeros((len(gdf)*kNeighbors, 3), dtype=np.int64)
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
        dist, indexes = tree.query(currentPoint, k=kNeighbors+1)
        #first point is the point which we used to query
        for j, index in enumerate(indexes):
            if j == 0:
                continue
            #gets point
            point = tree.data[index]
            point = shapely.Point(point)
            #gets index of gdf which the point represents
            actual_index = geoToIndex[point]
            edgesArray[i*kNeighbors+j-1] = [row[indexColumn], actual_index, dist[j]]
    edgesDf = pd.DataFrame(edgesArray, columns=['u', 'v', 'dist'])
    return edgesDf

#receives crimedf joined with city graph nodes
def createNodesDf(mergedGdf):
    dfSize = len(mergedGdf['index_right'].unique())
    columns=["node_id", "latitude", "longitude", "amount", "dawn", "morning", "afternoon", "night"]
    #index for first time column
    TIME = columns.index("dawn")
    #index for amount colum
    AMOUNT= columns.index("amount")
    GEO = columns.index("latitude")
    #constructed using stolen vehicles dataset from 2017 - 2024
    local_subtype = [
    'acougue/frigorífico-câmara frigorífica', 'via pública-transeunte', 'area de descanso-interior de veiculo particular',
    'mercado', 'oficina', 'parque/bosque/horto/reserva',
    'auto-peças', 'praça de pedágio-cabine/posto', 'bar/botequim',
    'delegacia/distrito policial', 'padaria/confeitaria', 'armazém/empório',
    'lanchonete/pastelaria/pizzaria-outros', 'via pública-ciclofaixa', 'interior veículo de carga',
    'metroviário e ferroviário metropolitano', 'feira-livre-outros', 'praça de pedágio-outros',
    'açougue/frigorífico', 'rodoviário', 'obra/construção',
    'área de descanso', 'posto de auxílio', 'lote de terreno',
    'tunel/viaduto/ponte-transeunte', 'transportadora', 'atacadista',
    'conveniência', 'estacionamento', 'area de descanso-transeunte',
    'praça', 'acougue/frigorífico-outros', 'posto de auxílio-interior de veiculo de carga',
    'rodoviário-outros', 'fabrica/indústria', 'fábrica/indústria',
    'estacionamento particular', 'balança-outros', 'loja de material de construção',
    'mecânica/borracharia', 'restaurante-outros', 'sala de reuniões/convenções',
    'interior de transporte coletivo', 'tunel/viaduto/ponte-interior de veiculo de carga', 'estacionamento público',
    'via pública', 'lojas', 'posto de gasolina',
    'via pública-interior de veiculo de carga', 'area de descanso-interior de veiculo de carga', 'interior de veículo particular',
    'praça de pedágio', 'semáforo-outros', 'farmácia/drogaria',
    'area de descanso-outros', 'terreno baldio', 'restaurante-estacionamento',
    'ciclofaixa', 'semáforo-interior de veiculo particular', 'posto policial-outros',
    'rodoviário-estacioanamento', 'acostamento-interior de veiculo de carga', 'hospital-outros',
    'distribuidora', 'praça de pedágio-estacionamento', 'veículo em movimento',
    'posto de auxílio-interior de veiculo particular', 'praça-outros', 'construção abandonada',
    'posto de fiscalização-outros', 'interior de veículo de carga',
    'outros', 'hospital-almoxarifado', 'balança',
    'semáforo', 'agência-outros', 'tunel/viaduto/ponte-outros',
    'restaurante', 'depósito', 'outros-estacionamento',
    'posto de fiscalização', 'praça-interior de veículo de carga', 'lanchonete/pastelaria/pizzaria-estacionamento',
    'loteamento', 'acostamento-outros', 'posto de auxílio-estacionamento',
    'bar/botequim-outros', 'rodoviário-garagem', 'delegacia',
    'hospital', 'praça de pedágio-interior de veiculo de carga', 'túnel/viaduto/ponte',
    'via pública-interior de veiculo particular', 'posto de auxílio-cabine/posto/escritório', 'transeunte',
    'desmanche', 'metalúrgica-outros', 'posto de auxílio-outros',
    'aeroportuário', 'semáforo-interior de veiculo de carga', 'acostamento-transeunte',
    'acostamento', 'outros-banheiro', 'portuário-outros',
    'área comum'
    ]
    SUBTYPE = len(columns)
    subTypeIndex = {}
    #associates subtype to index in order to encode
    for i, v in enumerate(local_subtype):
        subTypeIndex[v] = i
    rowsNpArray = np.zeros((dfSize, len(columns)+len(local_subtype)))
    #maps node id to idnex in df
    nodeIdToIndex = {}
    #to keep track of nodes id
    currentIndex = 0
    #loops through rows building the df
    for i, row in mergedGdf.iterrows():
        #checks if nodeId already in rowsArray
        indexInDf = nodeIdToIndex.get(row['index_right'])
        #must add node entry to array
        if indexInDf == None:
            nodeIdToIndex[row['index_right']] = currentIndex
            indexInDf = currentIndex
            rowsNpArray[indexInDf][0] = row['index_right']
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
        rowsNpArray[indexInDf][TIME + timeShift] += 1
        #parses subtype
        subType = row['DESCR_SUBTIPOLOCAL']
        if not pd.isnull(subType) and subTypeIndex.get(subType.lower()) is not None:
            encodedIndex = subTypeIndex[subType.lower()]
            rowsNpArray[indexInDf][SUBTYPE + encodedIndex] += 1
    #update columns to have descr_subtipolocal
    columns = columns + local_subtype
    nodesDf = pd.DataFrame(rowsNpArray, columns=columns)
    return nodesDf

def createCrimeGraphDf(crimeOcorrencesGdf, cityGraphNodes, kNeighbors=3):
    #projects ocorrences into citygraph
    merge = joinCrimeWithCity(crimeOcorrencesGdf, cityGraphNodes)
    edgesDf = createEdgesDf(merge, "index_right", kNeighbors)
    nodesDf = createNodesDf(merge)
    return nodesDf, edgesDf


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