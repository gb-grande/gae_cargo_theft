import argparse
import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
import torch_geometric.nn as gnn
import torch_geometric as tg
import matplotlib as mp
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans
import plotly.express as px

def createGraphFromDf(nodesDf, edgesDf):
    G = nx.Graph()
    #iter rows of nodesDf
    for index, row in nodesDf.iterrows():
        G.add_node(row['node_id'], numeric=row.iloc[3:].values, nodeId=row['node_id'])
    #iter rows of edgeDf
    for index, row in edgesDf.iterrows():
        G.add_edge(row['u'], row['v'], weight=row['dist'])
    G = G.to_undirected()
    return G

#uses pca to determine possible embedding dim
def getEmbedDimSuggestion(numericFeatures, explainedVariancePercentage=0.9999):
    pca = PCA(n_components=explainedVariancePercentage, svd_solver='full')
    pca.fit(numericFeatures)
    return len(pca.components_)


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GraphAutoEncoder(gnn.GAE):
    def __init__(self, encoder, device, decoder=None, lossFunc=None):
        super().__init__(encoder=encoder, decoder=decoder)
        self.device = device
        if lossFunc is None:
            self.lossFunc = self.recon_loss
        else:
            self.lossFunc = lossFunc
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    #data is expected to be from a random link split transform
    def trainModel(self, data):
        self.train()
        self.optimizer.zero_grad()
        z = self.encode(data.numeric, data.edge_index)
        loss = self.recon_loss(z, data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    #data is expected to be from a random link split transform
    @torch.no_grad()
    def testModel(self, data):
        self.eval()
        z = self.encode(data.numeric, data.edge_index)
        return self.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    #data is expected to be from a random link split transform
    def trainLoop(self, trainData, testData, epochs=100):
        aucArray = []
        apArray = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            loss = self.trainModel(trainData)
            auc, ap = self.testModel(testData)
            aucArray.append(auc)
            apArray.append(ap)
            if epochs >=1000:
              if epoch%100==0:
                  print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
            else:
              if epoch%10==0:
                  print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        return aucArray, apArray

    @torch.no_grad()
    def genEmbeddings(self, data):
        self.eval()
        data.to(self.device)
        return self.encode(data.numeric, data.edge_index).detach().cpu().numpy()



argParser = argparse.ArgumentParser(prog="Graph Autoencoder"
                                    , description="Trains a GAE to produce embeddings for analysis")

argParser.add_argument("nodesDf", action="store")
argParser.add_argument("edgesDf", action="store")
argParser.add_argument("-e", "--epochs", action="store")
argParser.add_argument("-d", "--dimEmbed", action="store")

