import igraph
from igraph import Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pickle

"""
#create example data for graph, this will have to be switched out with a dataLoader
types = [0, 1, 2]
pos = [0, 2, 1]
edges = [(0, 1), (1, 2)]
y = 0.5

#creates graph object, probably part of the dataLoader process?
g = Graph(directed=True)
g.add_vertices(3)
g.vs['type'] = types
g.vs['pos'] = pos
g.add_edges(edges)

#print out feature information for each node
for v in g.vs:
    print(v.attributes())

#print out adjacency matrix, note that because directed graph, only 2 adjacencies
A = g.get_adjacency()
print(A)

#print image rep of graph at position 0
igraph.plot(g, 'testGraph.png', vertex_label=types, vertex_size=30)
"""

#import OCBdataset pickle file and load into variable, creates tuple of igraph.Graph objects
OCBdataset = pickle.load(open(".\data\OCBdataset\ckt_bench_101.pkl", 'rb'))

#create GraphDataset object and define the constructor method
class GraphDataset(Data):
    def __init__(self, types=None, pos=None, edge_index=None, y=None):
        super().__init__(edge_index=edge_index, y=y)
        self.types = types
        self.pos = pos

#create array and tensorise graph information to create Data object to place in array
G = []
types = torch.tensor(g.vs['type'])
pos = torch.tensor(g.vs['pos'])
y = torch.tensor([y], dtype=torch.float32)
edge_index = torch.tensor(g.get_edgelist(), dtype=torch.long).permute(1, 0)
#print(edge_index)

#creating the first batch?
g_pyg = GraphDataset(types=types, pos=pos, edge_index=edge_index, y=y)
G.append(g_pyg)

#creating the second batch?
g_pyg2 = GraphDataset(types=types, pos=pos, edge_index=edge_index, y=y)
G.append(g_pyg2)

dataloader = DataLoader(G, batch_size=2)

#create Model object and define constructor method
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.gcn = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.2, act='relu')
        self.out = nn.Linear(hidden_dim, 1)
    
    #define forward method
    def forward(self, batch_data, device):
        x_t = batch_data.types
        x_p = batch_data.pos
        e = batch_data.edge_index

        #One-hot Embedding
        x_t = F.one_hot(x_t, num_classes=3).to(torch.float32)
        x_p = F.one_hot(x_p, num_classes=3).to(torch.float32)

        x_t = self.linear1(x_t)  #(B*N, dx)
        x_p = self.linear2(x_p)  #(B*N, dx)

        z = torch.cat((x_t, x_p), dim=-1)  #(B, N, dx*2)

        z = self.gcn(z, e) #(B*N, dx)

        # Graph pooling
        z = global_mean_pool(z, batch_data.batch)
        pred = self.out(z)
        return pred

epochs = 10
input_dim = 3
hidden_dim = 16
num_layers = 3
# If you run under Nvidia GPU have cuda installed, you could accelerate the computation through GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(input_dim, hidden_dim, num_layers).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

"""
for i in range(0, epochs):
    print(f"------------------ epoch_{i} -----------------")
    for i, batch_data in enumerate(dataloader):
        batch_data.to(device)
        optimizer.zero_grad()
        pred = model(batch_data, device)
        loss = criterion(pred.squeeze(), batch_data.y)
        loss.backward()
        optimizer.step()
        print(loss.item())
"""