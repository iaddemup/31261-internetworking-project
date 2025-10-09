import igraph
from igraph import Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pprint, pickle
import pandas as pd

#import OCBdataset pickle file and load into variable, creates tuple of igraph.Graph objects
OCBdataset = pickle.load(open(r"data\OCBdataset\ckt_bench_101.pkl", 'rb'))
OCBtensorset = pickle.load(open(r"data\OCBdataset\ckt_bench_101_tensor.pkl", 'rb'))
OCBpygraphset = pickle.load(open(r"data\OCBdataset\ckt_bench_101_pygraph.pkl", 'rb'))

#import perform101.csv for ground-truth values
csv = pd.read_csv(r"data\OCBdataset\perform101.csv")

#create GraphDataset object and define the constructor method
class GraphDataset(Data):
    def __init__(self, c=None, gm=None, pos=None, r=None, type=None, vid=None, edge_index=None, y=None):
        super().__init__(edge_index=edge_index, y=y)
        self.c = c
        self.gm = gm
        self.pos = pos
        self.r = r
        self.type = type
        self.vid = vid

trainingDataset = OCBdataset[0]
testingDataset = OCBdataset[1]

#create array and tensorise graph information to create Data object to place in array
gTrain = []
gTest = []

for index, (richGraph, simpleGraph) in enumerate(trainingDataset):
    c = torch.tensor(richGraph.vs['c'], dtype=torch.float32)
    gm = torch.tensor(richGraph.vs['gm'], dtype=torch.float32)
    pos = torch.tensor(richGraph.vs['pos'], dtype=torch.float32)
    r = torch.tensor(richGraph.vs['r'], dtype=torch.float32)
    type = torch.tensor(richGraph.vs['type'], dtype=torch.long)
    vid = torch.tensor(richGraph.vs['vid'], dtype=torch.float32)
    edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).permute(1, 0)
    y = torch.tensor(csv.loc[index, ['gain', 'pm', 'bw', 'fom']].values, dtype=torch.float32)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type, vid=vid, edge_index=edge_index, y=y)
    gTrain.append(richGraph_pyg)

for index, (richGraph, simpleGraph) in enumerate(testingDataset):
    c = torch.tensor(richGraph.vs['c'], dtype=torch.float32)
    gm = torch.tensor(richGraph.vs['gm'], dtype=torch.float32)
    pos = torch.tensor(richGraph.vs['pos'], dtype=torch.float32)
    r = torch.tensor(richGraph.vs['r'], dtype=torch.float32)    
    type = torch.tensor(richGraph.vs['type'], dtype=torch.long)
    vid = torch.tensor(richGraph.vs['vid'], dtype=torch.float32)
    edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).permute(1, 0)
    y = torch.tensor(csv.loc[len(trainingDataset) + index, ['gain', 'pm', 'bw', 'fom']].values, dtype=torch.float32)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type, vid=vid, edge_index=edge_index, y=y)
    gTest.append(richGraph_pyg)


trainDataloader = DataLoader(gTrain, batch_size=32)
testDataloader = DataLoader(gTest, batch_size=32)


#create Model object and define constructor method
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        #self.gcn = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.2, act='relu')
        self.out = nn.Linear(hidden_dim, 4)
    
    #define forward method
    def forward(self, batch_data, device):
        x_t = F.one_hot(batch_data.type, num_classes=25).to(torch.float32) #need to figure out exactly how many type values there can be, or else one-hot won't work
        x_num = torch.stack([batch_data.c, batch_data.gm, batch_data.pos, batch_data.r, batch_data.vid], dim=-1).to(torch.float32)
        e = batch_data.edge_index

        x_t = self.linear1(x_t)  #(B*N, dx)
        x_num = self.linear2(x_num)  #(B*N, dx)

        z = torch.cat((x_t, x_num), dim=-1)  #(B, N, dx*2)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for i in range(0, epochs):
    print(f"------------------ epoch_{i} -----------------")
    for i, batch_data in enumerate(trainDataloader):
        batch_data.to(device)
        optimizer.zero_grad()
        pred = model(batch_data, device)
        loss = criterion(pred.squeeze(), batch_data.y)
        loss.backward()
        optimizer.step()
        print(loss.item())


#predictions = model(graph_batch)
#pd.DataFrame(predictions, columns=["valid", "gain", "pm", "bw", "fom"]).to_csv("predictions.csv", index=False)