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

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

#import OCBdataset pickle file and load into variable, creates tuple of igraph.Graph objects
OCBdataset = pickle.load(open(r"data\OCBdataset\ckt_bench_101.pkl", 'rb'))

#import perform101.csv for ground-truth values
csv = pd.read_csv(r"data\OCBdataset\perform101.csv")

trainingDataset = OCBdataset[0]
testingDataset = OCBdataset[1]

#create GraphDataset object and define the constructor method
class GraphDataset(Data):
    def __init__(self, c=None, gm=None, pos=None, r=None, type=None, vid=None, x_num=None, edge_index=None, y=None):
        super().__init__(edge_index=edge_index, y=y)
        self.c = c
        self.gm = gm
        self.pos = pos
        self.r = r
        self.type = type
        self.vid = vid
        self.x_num = x_num

#find the maximum amount of types available in the dataset
max_node_types = 0
for (richGraph, simpleGraph) in trainingDataset + testingDataset:
    max_node_types = max(max_node_types, max(richGraph.vs['type']))
max_node_types += 1

#collect and fit node scaler once
node_features_train = []

for (richGraph, simpleGraph) in trainingDataset:
    features = np.column_stack([
        richGraph.vs['c'],
        richGraph.vs['gm'],
        richGraph.vs['pos'],
        richGraph.vs['r'],
        richGraph.vs['vid']
    ])
    node_features_train.append(features)

node_features_train = np.vstack(node_features_train)

node_scaler = StandardScaler()
node_scaler.fit(node_features_train)

#fit y scaler once
y_scaler = StandardScaler()
y_train = csv.iloc[:len(trainingDataset)][['gain', 'bw', 'pm', 'fom']].values
y_scaler.fit(y_train)

#build pygraph.Data objects for train/test
def buildData(dataset, start_idx=0):
    out = []
    for i, (richGraph, simpleGraph) in enumerate(dataset):
        features = np.column_stack([
            richGraph.vs['c'],
            richGraph.vs['gm'],
            richGraph.vs['pos'],
            richGraph.vs['r'],
            richGraph.vs['vid']
        ])

        features = node_scaler.transform(features)

        c, gm, pos, r, vid = [torch.tensor(features[:, i], dtype=torch.float32) for i in range(5)]
        x_num = torch.tensor(features, dtype=torch.float32)

        type_t = torch.tensor(richGraph.vs['type'], dtype=torch.long)

        edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).t().contiguous()

        y_values = csv.loc[start_idx + i, ['gain', 'bw', 'pm', 'fom']].values
        y_norm = y_scaler.transform(y_values.reshape(1, -1))
        y = torch.tensor(y_norm, dtype=torch.float32)

        richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type_t, vid=vid, x_num=x_num, edge_index=edge_index, y=y)
        out.append(richGraph_pyg)
    return out

#create Model object and define constructor method
class GNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, activation):
        super().__init__()
        self.linear1 = nn.Linear(max_node_types, hidden_dim)
        self.linear2 = nn.Linear(5, hidden_dim)
        
        self.gcn_cat = GCN(hidden_dim, hidden_dim, num_layers, dropout=0.3, norm='layernorm')
        self.gcn_num = GCN(hidden_dim, hidden_dim, num_layers, dropout=0.3, norm='layernorm')

        self.linear_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        self.activation = activation
        self.out = nn.Linear(hidden_dim, 4)
    
    #define forward method
    def forward(self, batch_data):
        x_t = F.one_hot(batch_data.type, max_node_types).to(torch.float32)
        x_num = batch_data.x_num.to(torch.float32)
        e = batch_data.edge_index

        x_t = self.linear1(x_t)
        x_num = self.linear2(x_num)

        z_t = self.activation(self.gcn_cat(x_t, e))
        z_num = self.activation(self.gcn_num(x_num, e))

        z = torch.cat((z_t, z_num), dim=-1)
        z = self.activation(self.linear_fuse(z))

        z = global_mean_pool(z, batch_data.batch)
        pred = self.out(z)
        return pred
    
    @staticmethod
    def training(dataloader, model, loss_fn, optimizer, device):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            # Compute prediction error
            pred = model(batch_data)
            loss = loss_fn(pred, batch_data.y)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        return epoch_loss / num_batches

    @staticmethod
    def testing(dataloader, model, loss_fn, device):
        model.eval()
        num_batches = len(dataloader)
        total_loss = 0.0

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                loss = loss_fn(pred, batch_data.y)
                total_loss += loss.item()
        
        return total_loss / num_batches



#create array and tensorise graph information to create Data object to place in array
gTrain = buildData(trainingDataset, start_idx=0)
gTest = buildData(testingDataset, start_idx=len(trainingDataset))

batch_size = 128
trainDataloader = DataLoader(gTrain, batch_size, shuffle=True)
testDataloader = DataLoader(gTest, batch_size)

#configure model environment + layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN(hidden_dim=128, num_layers=3, activation=F.elu).to(device)

loss_fn = nn.L1Loss()

optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.7, patience=5)

best_test_loss = float('inf')
best_epoch = 0
patience_early_stop = 10
patience_counter = 0

epochs = 100

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
    
    train_loss = GNN.training(trainDataloader, model, loss_fn, optimiser, device)
    print(f"Training Loss: {train_loss:.6f}")

    test_loss = GNN.testing(testDataloader, model, loss_fn, device)
    print(f"Testing Loss: {test_loss:.6f}")

    scheduler.step(test_loss)

    if test_loss < best_test_loss - 1e-5:
        best_test_loss = test_loss
        best_epoch = epoch + 1
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience_early_stop})")

    if patience_counter >= patience_early_stop:
        print("Early stop triggered")
        print(f"Best model was from epoch {best_epoch} with test loss = {best_test_loss:.6f}")
        break