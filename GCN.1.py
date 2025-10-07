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
OCBdataset = pickle.load(open(r"data\OCBdataset\ckt_bench_101.pkl", 'rb'))
OCBtensorset = pickle.load(open(r"data\OCBdataset\ckt_bench_101_tensor.pkl", 'rb'))
OCBpygraphset = pickle.load(open(r"data\OCBdataset\ckt_bench_101_pygraph.pkl", 'rb'))
OCBtestset = open(r"data\OCBdataset\perform101.csv", 'rb')

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

for richGraph, simpleGraph in trainingDataset:
    c = torch.tensor(richGraph.vs['c'])
    gm = torch.tensor(richGraph.vs['gm'])
    pos = torch.tensor(richGraph.vs['pos'])
    r = torch.tensor(richGraph.vs['r'])
    type = torch.tensor(richGraph.vs['type'])
    vid = torch.tensor(richGraph.vs['vid'])
    edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).permute(1, 0)
    y = torch.tensor([richGraph.vcount()], dtype=torch.float32)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type, vid=vid, edge_index=edge_index, y=y)
    gTrain.append(richGraph_pyg)

for richGraph, simpleGraph in testingDataset:
    c = torch.tensor(richGraph.vs['c'])
    gm = torch.tensor(richGraph.vs['gm'])
    pos = torch.tensor(richGraph.vs['pos'])
    r = torch.tensor(richGraph.vs['r'])    
    type = torch.tensor(richGraph.vs['type'])
    vid = torch.tensor(richGraph.vs['vid'])
    edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).permute(1, 0)
    y = torch.tensor([richGraph.vcount()], dtype=torch.float32)
    richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type, vid=vid, edge_index=edge_index, y=y)
    gTest.append(richGraph_pyg)


trainDataloader = DataLoader(gTrain, batch_size=4)
testDataloader = DataLoader(gTest, batch_size=4)

#debugging check 
for batch in trainDataloader:
    print(batch)
    print("Node feature shape:", batch.x.shape if batch.x is not None else "None")
    print("Edge index shape:", batch.edge_index.shape)
    print("Target shape:", batch.y.shape)
    break

for batch in testDataloader:
    print("Node feature shape:", batch.x.shape if batch.x is not None else "None")
    print("Edge index shape:", batch.edge_index.shape)
    print("Target shape:", batch.y.shape)
    print("Number of nodes in batch:", batch.num_nodes)
    print("Number of graphs in batch:", batch.num_graphs)
    break

#create Model object and define constructor method
#Defining the GNN model architecture 
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim) 
        self.linear2 = nn.Linear(input_dim, hidden_dim) 
        self.gcn = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.2, act='relu')
        self.out = nn.Linear(hidden_dim, 1)
    
    #define forward method
    def forward(self, batch_data, device):
        num_classes = int(batch_data.type.max().item() + 1) #automatic detection of the largest num classes
        x_t = F.one_hot(batch_data.type, num_classes=num_classes).to(torch.float32)
        x_num = torch.stack([batch_data.c, batch_data.gm, batch_data.pos, batch_data.r], dim=-1).to(torch.float32)
        #batch_data.(var) added because processing a batch of graphs not single graph + outside the scope 
        e = batch_data.edge_index

        if not hasattr(self, 'linear1') or self.linear1.in_features != num_classes:
            self.linear1 = nn.Linear(num_classes, self.linear1.out_features).to(device)

        x_t = self.linear1(x_t)  #(B*N, dx) matrix multiplication 
        x_num = self.linear2(x_num)  #(B*N, dx)

        z = torch.cat((x_t, x_num), dim=-1)  #(B, N, dx*2)

        z = self.gcn(z, e) #(B*N, dx)

        # Graph pooling
        z = global_mean_pool(z, batch_data.batch)
        pred = self.out(z)
        return pred

epochs = 10 #number of iterations
input_dim = 4 #is 4 due to 4 numeric features 
hidden_dim = 32
num_layers = 5
# If you run under Nvidia GPU have cuda installed, you could accelerate the computation through GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(input_dim, hidden_dim, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss() #loss function 

for i in range(0, epochs):
    print(f"------------------ epoch_{i} -----------------")
    model.train()  # set model to training mode
    #epoch_mape = 0
    epoch_loss = 0  # accumulator for total loss in this epoch
    num_batches = 0  # count batches
    for i, batch_data in enumerate(testDataloader):
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        pred = model(batch_data, device)
        loss = criterion(pred.squeeze(), batch_data.y) #loss 
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mape = torch.mean(torch.abs((batch_data.y - pred) / batch_data.y)) * 100
            #epoch_mape += mape.item()
            num_batches += 1
            epoch_loss += loss.item()
            num_batches += 1
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epochs+1} Average Loss : {avg_loss:.6f}")
                
    

        
    
    #print(loss.item())
    # 
    # #avg_mape = epoch_mape / num_batches
    #print(f"Epoch {epochs+1} Average MAPE: {avg_mape:.2f}%")