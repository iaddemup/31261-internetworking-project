import igraph
from igraph import Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pprint, pickle

#import OCBdataset pickle file and load into variable, creates tuple of igraph.Graph objects
OCBdataset = pickle.load(open("data\OCBdataset\ckt_bench_101.pkl", 'rb'))
OCBtensorset = pickle.load(open("data\OCBdataset\ckt_bench_101_tensor.pkl", 'rb'))
OCBpygraphset = pickle.load(open("data\OCBdataset\ckt_bench_101_pygraph.pkl", 'rb'))
OCBtestset = open("data\OCBdataset\perform101.csv", 'rb')

#since the subg lists can be of variable lengths, this provides a length for each array available
def flatten_with_ptr(lists):
    flat = []
    ptr = [0]
    for lst in lists:
        flat.extend(lst)
        ptr.append(len(flat))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(ptr, dtype=torch.long)

#create GraphDataset object and define the constructor method
class GraphDataset(Data):
    def __init__(self, c=None, gm=None, pos=None, r=None, subg_adj_flat=None, subg_adj_ptr=None, subg_nfeats_flat=None, subg_nfeats_ptr=None, subg_ntypes_flat=None, subg_ntypes_ptr=None, type=None, vid=None, edge_index=None, y=None):
        super().__init__(edge_index=edge_index, y=y)
        self.c = c
        self.gm = gm
        self.pos = pos
        self.r = r
        self.subg_adj_flat = subg_adj_flat
        self.subg_adj_ptr = subg_adj_ptr
        self.subg_nfeats_flat = subg_nfeats_flat
        self.subg_nfeats_ptr = subg_nfeats_ptr
        self.subg_ntypes_flat = subg_ntypes_flat
        self.subg_ntypes_ptr = subg_ntypes_ptr
        self.type = type
        self.vid = vid

trainingDataset = OCBdataset[0]
testingDataset = OCBdataset[1]

#create array and tensorise graph information to create Data object to place in array
gTrain = []

for richGraph, simpleGraph in trainingDataset:
    c = torch.tensor(richGraph.vs['c'])
    gm = torch.tensor(richGraph.vs['gm'])
    pos = torch.tensor(richGraph.vs['pos'])
    r = torch.tensor(richGraph.vs['r'])
    subg_adj_flat, subg_adj_ptr = flatten_with_ptr(richGraph.vs['subg_adj'])
    subg_nfeats_flat, subg_nfeats_ptr = flatten_with_ptr(richGraph.vs['subg_nfeats'])
    subg_ntypes_flat, subg_ntypes_ptr = flatten_with_ptr(richGraph.vs['subg_ntypes'])
    type = torch.tensor(richGraph.vs['type'])
    vid = torch.tensor(richGraph.vs['vid'])
    edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).permute(1, 0)
    y = torch.tensor([richGraph.vcount()], dtype=torch.float32)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, subg_adj_flat=subg_adj_flat, subg_adj_ptr=subg_adj_ptr, subg_nfeats_flat=subg_nfeats_flat, subg_nfeats_ptr=subg_nfeats_ptr, subg_ntypes_flat=subg_ntypes_flat, subg_ntypes_ptr=subg_ntypes_ptr, type=type, vid=vid, edge_index=edge_index, y=y)
    gTrain.append(richGraph_pyg)

gTest = []

for richGraph, simpleGraph in testingDataset:
    c = torch.tensor(richGraph.vs['c'])
    gm = torch.tensor(richGraph.vs['gm'])
    pos = torch.tensor(richGraph.vs['pos'])
    r = torch.tensor(richGraph.vs['r'])    
    subg_adj_flat, subg_adj_ptr = flatten_with_ptr(richGraph.vs['subg_adj'])
    subg_nfeats_flat, subg_nfeats_ptr = flatten_with_ptr(richGraph.vs['subg_nfeats'])
    subg_ntypes_flat, subg_ntypes_ptr = flatten_with_ptr(richGraph.vs['subg_ntypes'])
    type = torch.tensor(richGraph.vs['type'])
    vid = torch.tensor(richGraph.vs['vid'])
    edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).permute(1, 0)
    y = torch.tensor([richGraph.vcount()], dtype=torch.float32)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, subg_adj_flat=subg_adj_flat, subg_adj_ptr=subg_adj_ptr, subg_nfeats_flat=subg_nfeats_flat, subg_nfeats_ptr=subg_nfeats_ptr, subg_ntypes_flat=subg_ntypes_flat, subg_ntypes_ptr=subg_ntypes_ptr, type=type, vid=vid, edge_index=edge_index, y=y)
    gTest.append(richGraph_pyg)


trainDataloader = DataLoader(gTrain, batch_size=2)
testDataloader = DataLoader(gTest, batch_size=2)

#debugging check 
for batch in trainDataloader:
    print(batch)
    print("Node feature shape:", batch.x.shape if batch.x is not None else "None")
    print("Edge index shape:", batch.edge_index.shape)
    print("Target shape:", batch.y.shape)
    break

for X, y in testDataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

for X, y in trainDataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



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

        i = 0
        start, end = batch_data.subg_adj_ptr[i], batch_data.subg_adj_ptr[i+1]
        node_i_adj = batch_data.subg_adj_flat[start:end]

        i = 0
        start, end = batch_data.subg_nfeats_ptr[i], batch_data.subg_nfeats_ptr[i+1]
        node_i_adj = batch_data.subg_nfeats_flat[start:end]

        i = 0
        start, end = batch_data.subg_ntypes_ptr[i], batch_data.subg_ntypes_ptr[i+1]
        node_i_adj = batch_data.subg_ntypes_flat[start:end]

        #One-hot Embedding
        x_t = F.one_hot(x_t, num_classes=5).to(torch.float32)
        x_p = F.one_hot(x_p, num_classes=5).to(torch.float32)

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