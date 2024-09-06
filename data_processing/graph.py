import osmnx as ox
import networkx as nx

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