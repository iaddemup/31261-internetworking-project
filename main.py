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

seed = 31261
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

#import OCBdataset pickle file and load into variable, creates tuple of igraph.Graph objects
OCBdataset = pickle.load(open(r"data\OCBdataset\ckt_bench_101.pkl", 'rb'))

#import perform101.csv for ground-truth values
csv = pd.read_csv(r"data\OCBdataset\perform101.csv")

trainingDataset = OCBdataset[0]
testingDataset = OCBdataset[1]

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

#find the maximum amount of types available in the dataset
max_node_types = 0
for (richGraph, _) in trainingDataset + testingDataset:
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

        type_t = torch.tensor(richGraph.vs['type'], dtype=torch.long)

        edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).t().contiguous()

        csv_idx = start_idx + i
        y_values = csv.loc[csv_idx, ['gain', 'pm', 'bw', 'fom']].values.reshape(1, -1)
        y_values_df = pd.DataFrame(y_values, columns=['gain', 'pm', 'bw', 'fom'])
        y_norm = y_scaler.transform(y_values_df)
        y = torch.tensor(y_norm, dtype=torch.float32)

        richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type_t, vid=vid, edge_index=edge_index, y=y)
        out.append(richGraph_pyg)
    return out

y_train = csv.iloc[:len(trainingDataset)][['gain', 'pm', 'bw', 'fom']]
y_scaler.fit(y_train)

#create array and tensorise graph information to create Data object to place in array
gTrain = buildData(trainingDataset, start_idx=0)
gTest = buildData(testingDataset, start_idx=len(trainingDataset))

batch_size = 128
trainDataloader = DataLoader(gTrain, batch_size, shuffle=True)
testDataloader = DataLoader(gTest, batch_size)

#create Model object and define constructor method
class GNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(max_node_types, hidden_dim)
        self.linear2 = nn.Linear(5, hidden_dim)
        self.gcn = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.4, norm='batchnorm')
        
        act_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        self.activation = act_map[activation]
    
        self.out = nn.Linear(hidden_dim, 4)
    
    #define forward method
    def forward(self, batch_data, device):
        x_t = F.one_hot(batch_data.type, max_node_types).to(torch.float32)
        x_num = torch.stack([batch_data.c, batch_data.gm, batch_data.pos, batch_data.r, batch_data.vid], dim=-1).to(torch.float32)
        e = batch_data.edge_index

        x_t = self.linear1(x_t)
        x_num = self.linear2(x_num)

        z = torch.cat((x_t, x_num), dim=-1)

        z = self.gcn(z, e)
        z = self.activation(z)

        z = global_mean_pool(z, batch_data.batch)
        pred = self.out(z)
        return pred
    
    def training(dataloader, model, loss_fn, optimizer, device):
        model.train()
        size = len(dataloader.dataset)
        epoch_loss = 0.0
        num_batches = 0

        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)

            # Compute prediction error
            pred = model(batch_data, device)
            loss = loss_fn(pred, batch_data.y)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            if (i + 1) % 70 == 0:
                loss_val, current = loss.item(), (i + 1) * len(batch_data)
                print(f"Loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return epoch_loss / num_batches

    def testing(dataloader, model, loss_fn, device):
        model.eval()
        num_batches = len(dataloader)
        total_loss = 0.0
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(device)
                pred = model(batch_data, device)
                #print("Preds: ", pred[0].cpu().numpy())
                #print("Y: ", batch_data.y[0].cpu().numpy())
                loss = loss_fn(pred, batch_data.y)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Test Loss ({loss_fn.__class__.__name__}): {avg_loss:.6f}")
        return avg_loss

#configure model environment + layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = GNN(hidden_dim=128, num_layers=3).to(device)

#loading past learnings
print("Would you like to load in previous learnings?")
if input() == "Yes":
    try:
        model.load_state_dict(torch.load(r"savedModels\model.pth", weights_only=True))
        print("Model loaded.")
        try:
            node_scaler = joblib.load(r"savedModels\node_scaler.pkl")
            y_scaler = joblib.load(r"savedModels\y_scaler.pkl")
            print("Scalers loaded.")
        except:
            print("No saved scalers detected.")
    except:
        print("No saved model detected.")

#activation functions

activations = ["relu", "leaky_relu", "elu", "gelu"]

#loss functions
MSE = nn.MSELoss()
def rMSE(pred, batch_data):
    return torch.sqrt(MSE(pred, batch_data))
MAE = nn.L1Loss()
Huber = nn.HuberLoss(delta=1.0)

loss_functions = {
    "MSE": MSE,
    "rMSE": rMSE,
    "MAE": MAE,
    "Huber": Huber
}

#optimisers

optimisers = {
    "SGD" : torch.optim.SGD,
    "RMSprop" : torch.optim.RMSprop,
    "ADAM" : torch.optim.Adam,
    "ADAMw" : torch.optim.AdamW
}

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ADAMw, mode='min', factor=0.5, patience=3, min_lr=1e-6)

best_test_loss = float('inf')
best_epoch = 0
patience_early_stop = 5
patience_counter = 0

results = []
best_config = None

for opt_name, opt_class in optimisers.items():
    for loss_name, loss_func in loss_functions.items():
        for act_name in activations:

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            print(f"\nTesting: {opt_name}, {loss_name}, {act_name} ")
            print("------------------------------------------------------")


            model = GNN(hidden_dim=128, num_layers=3, activation=act_name).to(device)
            optimiser = opt_class(model.parameters(), lr=5e-4)

            total_loss = []

            for epoch in range(5):
                GNN.training(trainDataloader, model, loss_func, optimiser, device)

            val_loss = GNN.testing(testDataloader, model, loss_func, device)
    
            results.append((opt_name, loss_name, act_name, val_loss))

            if val_loss < best_test_loss:
                best_test_loss = val_loss
                best_config = (opt_name, loss_name, act_name)

# --- Print summary of results ---
print("\n=== Summary of All Tests ===")
for r in results:
    print(f"Optimizer={r[0]}, Loss={r[1]}, Activation={r[2]}, ValLoss={r[3]:.6f}")

# Find and print the best config
print(f"\nBest configuration: Optimizer={best_config[0]}, Loss={best_config[1]}, Activation={best_config[2]}, ValLoss={best_test_loss:.6f}")


"""
epochs = 50

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    GNN.training(trainDataloader, model, MAE, ADAMw, device)
    test_loss = GNN.testing(testDataloader, model, MAE, device)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch + 1
        patience_counter = 0
    else:
        patience_counter += 1
        print("no improvement")

    scheduler.step(test_loss)

    print("\n")

    if patience_counter >= patience_early_stop:
        print("early stop triggered")
        break

print("Would you like to save these learnings?")
if input() == "Yes":
    torch.save(model.state_dict(), r"savedModels\model.pth")
    print("Saved PyTorch Model State to model.pth")
    joblib.dump(node_scaler, r"savedModels\node_scaler.pkl")
    joblib.dump(y_scaler, r"savedModels\y_scaler.pkl")
    print("Saved scalers.")

"""