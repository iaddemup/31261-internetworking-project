import igraph
from igraph import Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, global_mean_pool, GraphSAGE, Node2Vec
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pprint, pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import pearsonr

seed = 31261
def set_seed(s):   
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

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
y_train = csv.iloc[:len(trainingDataset)][['gain', 'bw', 'pm', 'fom']]
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

        type_t = torch.tensor(richGraph.vs['type'], dtype=torch.long)

        edge_index = torch.tensor(richGraph.get_edgelist(), dtype=torch.long).t().contiguous()

        csv_idx = start_idx + i
        y_values = csv.loc[csv_idx, ['gain', 'bw', 'pm', 'fom']].values.reshape(1, -1)
        y_values_df = pd.DataFrame(y_values, columns=['gain', 'bw', 'pm', 'fom'])
        y_norm = y_scaler.transform(y_values_df)
        y = torch.tensor(y_norm, dtype=torch.float32)

        richGraph_pyg = GraphDataset(c=c, gm=gm, pos=pos, r=r, type=type_t, vid=vid, edge_index=edge_index, y=y)
        out.append(richGraph_pyg)
    return out

#create array and tensorise graph information to create Data object to place in array
gTrain = buildData(trainingDataset, start_idx=0)
gTest = buildData(testingDataset, start_idx=len(trainingDataset))

batch_size = 128
trainDataloader = DataLoader(gTrain, batch_size, shuffle=True)
testDataloader = DataLoader(gTest, batch_size)

#create Model object and define constructor method
class GNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, embedding_type="OneHot", activation="relu"):
        super().__init__()
        self.embedding_type = embedding_type

        if embedding_type == "OneHot":
            self.linear1 = nn.Linear(max_node_types, hidden_dim)
            self.linear2 = nn.Linear(5, hidden_dim)
            self.gnn_layer = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.4, norm='batchnorm')
        elif embedding_type == "GraphSAGE":
            self.linear_num = nn.Linear(5, hidden_dim)
            self.gnn_layer = GraphSAGE(hidden_dim, hidden_dim, num_layers=num_layers)
        elif embedding_type == "Node2Vec":
            self.node_emb = nn.Embedding(max_node_types, hidden_dim)
            self.linear_num = nn.Linear(5, hidden_dim)
            self.gnn_layer = GCN(hidden_dim*2, hidden_dim, num_layers, dropout=0.4, norm='batchnorm')
        elif embedding_type == "PinSAGE":
            self.linear_num = nn.Linear(5, hidden_dim)
            self.gnn_layer = GraphSAGE(hidden_dim, hidden_dim, num_layers=num_layers)
        
        act_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        self.activation = act_map[activation]

        self.out = nn.Linear(hidden_dim, 4)
    
    #define forward method
    def forward(self, batch_data):
        e = batch_data.edge_index

        if self.embedding_type == "OneHot":
            x_t = F.one_hot(batch_data.type, max_node_types).to(torch.float32)
            x_num = torch.stack([batch_data.c, batch_data.gm, batch_data.pos, batch_data.r, batch_data.vid], dim=-1).to(torch.float32)
            x_t = self.linear1(x_t)
            x_num = self.linear2(x_num)
            z = torch.cat((x_t, x_num), dim=-1)
            z = self.gnn_layer(z, e)

        elif self.embedding_type in ["GraphSAGE", "PinSAGE"]:
            x_num = torch.stack([batch_data.c, batch_data.gm, batch_data.pos, batch_data.r, batch_data.vid], dim=-1).to(torch.float32)
            z = self.linear_num(x_num)
            z = self.gnn_layer(z, e)

        elif self.embedding_type == "Node2Vec":
            x_num = torch.stack([batch_data.c, batch_data.gm, batch_data.pos, batch_data.r, batch_data.vid], dim=-1).to(torch.float32)
            x_num = self.linear_num(x_num)
            node_emb = self.node_emb(batch_data.type)
            z = torch.cat((node_emb, x_num), dim=-1)
            z = self.gnn_layer(z, e)
            
        z = self.activation(z)
        z = global_mean_pool(z, batch_data.batch)
        pred = self.out(z)
        return pred

def training(dataloader, model, loss_fn, optimizer, device, clip_norm=1.0):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for i, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        pred = model(batch_data)
        loss = loss_fn(pred, batch_data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    return running_loss / (num_batches+1e-12)

def testing(dataloader, model, loss_fn, device):
    model.eval()
    preds_list = []
    targets_list = []
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            pred = model(batch_data)
            loss = loss_fn(pred, batch_data.y)
            total_loss += loss.item()
            num_batches += 1
            preds_list.append(pred.cpu().numpy())
            targets_list.append(batch_data.y.cpu().numpy())
    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)
    avg_loss = total_loss / num_batches
    preds_unscaled = y_scaler.inverse_transform(preds)
    targets_unscaled = y_scaler.inverse_transform(targets)
    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    rmse = root_mean_squared_error(targets_unscaled, preds_unscaled)
    r2 = r2_score(targets_unscaled, preds_unscaled, multioutput='uniform_average')
    return avg_loss, mae, rmse, r2

def evaluate_paper_metrics(dataloader, model, y_scaler, device):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            pred = model(batch_data)

            # Denormalize predictions and labels
            y_pred_real = y_scaler.inverse_transform(pred.cpu().numpy())
            y_true_real = y_scaler.inverse_transform(batch_data.y.cpu().numpy())

            all_preds.append(y_pred_real)
            all_true.append(y_true_real)

    y_pred_real = np.vstack(all_preds)
    y_true_real = np.vstack(all_true)

    # match paper’s variable order (Gain, BW, PM, FoM)
    target_names = ['Gain', 'BW', 'PM', 'FoM']
    results = {}

    for i, name in enumerate(target_names):
        rmse = root_mean_squared_error(y_true_real[:, i], y_pred_real[:, i])
        r, _ = pearsonr(y_true_real[:, i], y_pred_real[:, i])
        results[name] = {'RMSE': rmse, 'Pearson_r': r}

    return results

#configure model environment + layers

embeddings = ["OneHot", "PinSAGE", "GraphSAGE", "Node2Vec"]

MSE = nn.MSELoss()
def rMSE(pred, target):
    return torch.sqrt(MSE(pred, target) + 1e-8)
MAE = nn.L1Loss()
Huber = nn.HuberLoss(delta=1.0)

loss_functions = {
    "MSE": MSE,
    "rMSE": rMSE,
    "MAE": MAE,
    "Huber": Huber
}

optimisers = {
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "ADAM": torch.optim.Adam,
    "ADAMw": torch.optim.AdamW
}
    
activations = ["relu", "leaky_relu", "elu", "gelu"]


learning_rates = [5e-4]
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_table = []

for emb in embeddings:
    for opt_name, opt_class in optimisers.items():
        for loss_name, loss_func in loss_functions.items():
            for act_name in activations:
                print(f"\nRunning config: Embedding={emb}, Optimiser={opt_name}, Loss={loss_name}, Activation={act_name}")
                
                set_seed(seed)

                model = GNN(hidden_dim=128, num_layers=3, embedding_type=emb, activation=act_name).to(device)
                optimiser = opt_class(model.parameters(), lr=learning_rates[0])

                for epoch in range(num_epochs):
                    train_loss = training(trainDataloader, model, loss_func, optimiser, device)

                results = evaluate_paper_metrics(testDataloader, model, y_scaler, device)
                results_table.append({
                    'Embedding' : emb,
                    'Activation': act_name,
                    'Optimizer': opt_name,
                    'Loss': loss_func.__class__.__name__,
                    'Gain_RMSE': results['Gain']['RMSE'],
                    'Gain_r': results['Gain']['Pearson_r'],
                    'BW_RMSE': results['BW']['RMSE'],
                    'BW_r': results['BW']['Pearson_r'],
                    'PM_RMSE': results['PM']['RMSE'],
                    'PM_r': results['PM']['Pearson_r'],
                    'FoM_RMSE': results['FoM']['RMSE'],
                    'FoM_r': results['FoM']['Pearson_r'],
                })

print("\nPaper-Style Evaluation Results:")
print(f"{'Model':<14} {'Gain RMSE':<12} {'Gain r':<12} {'BW RMSE':<12} {'BW r':<12} "
        f"{'PM RMSE':<12} {'PM r':<12} {'FoM RMSE':<12} {'FoM r':<12}")
print("-" * 108)

for result in results_table:
    print(f"{result['Embedding']+'-'+result['Activation']+'-'+result['Optimizer']:<14} "
          f"{result['Gain_RMSE']:<12.3f}{result['Gain_r']:<12.3f}"
          f"{result['BW_RMSE']:<12.3f}{result['BW_r']:<12.3f}"
          f"{result['PM_RMSE']:<12.3f}{result['PM_r']:<12.3f}"
          f"{result['FoM_RMSE']:<12.3f}{result['FoM_r']:<12.3f}")