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

max_node_types = 0
for (richGraph, _) in trainingDataset + testingDataset:
    max_node_types = max(max_node_types, max(richGraph.vs['type']))
max_node_types += 1

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
    y = torch.tensor(csv.loc[index, ['gain', 'pm', 'bw', 'fom']].values, dtype=torch.float32).unsqueeze(0)
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
    y = torch.tensor(csv.loc[len(trainingDataset) + index, ['gain', 'pm', 'bw', 'fom']].values, dtype=torch.float32).unsqueeze(0)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type, vid=vid, edge_index=edge_index, y=y)
    gTest.append(richGraph_pyg)

batch_size = 30

trainDataloader = DataLoader(gTrain, batch_size, shuffle=True)
testDataloader = DataLoader(gTest, batch_size)

#create Model object and define constructor method
class GNN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.linear1 = nn.Linear(max_node_types, hidden_dim)
        self.linear2 = nn.Linear(5, hidden_dim)
        self.gcn = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.2, act='relu')
        self.out = nn.Linear(hidden_dim, 4)
    
    #define forward method
    def forward(self, batch_data, device):
        x_t = F.one_hot(batch_data.type, max_node_types).to(torch.float32)
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
    
    def training(dataloader, model, loss_fn, optimizer, device):
        model.train()
        size = len(dataloader.dataset)

        for i, batch_data in enumerate(dataloader):
            batch_data.to(device)
            
            #Compute prediction error
            pred = model(batch_data, device)
            loss = loss_fn(pred, batch_data.y)
            
            #Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if(i + 1) % 30 == 0:
                loss, current = loss.item(), (i + 1) * len(batch_data)
                print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def testing(dataloader, model, loss_fn, device):
        model.eval()
        num_batches = len(dataloader)
        total_loss = 0.0
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                batch_data.to(device)
                pred = model(batch_data, device)
                loss = loss_fn(pred, batch_data.y)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Test Loss ({loss_fn.__class__.__name__}): {avg_loss:.6f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 25
num_layers = 5

model = GNN(hidden_dim, num_layers).to(device)
print("Would you like to load in previous learnings?")
if input() == "Yes":
    try:
        model.load_state_dict(torch.load("model.pth", weights_only=True))
        print(f"The following model was found in directory: \n {model}")
    except:
        print("No saved model detected.")

MSE = nn.MSELoss()
ADAM = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    GNN.training(trainDataloader, model, MSE, ADAM, device)
    GNN.testing(testDataloader, model, MSE, device)
    print("\n")

print("Would you like to save these learnings?")
if input() == "Yes":
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")