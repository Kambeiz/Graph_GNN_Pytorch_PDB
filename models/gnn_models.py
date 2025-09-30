"""
Graph Neural Network models for Drug-Target Interaction prediction
Includes: GCN, GAT, GraphSAGE, and GIN implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout


class GCN_DTI(nn.Module):
    """Graph Convolutional Network for DTI prediction"""
    
    def __init__(self, num_features, hidden_dim=128, num_layers=3, dropout=0.2):
        super(GCN_DTI, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Readout layers
        self.fc1 = Linear(hidden_dim * 3, hidden_dim)  # *3 for three pooling methods
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 1)
        
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level readout
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Final MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


class GAT_DTI(nn.Module):
    """Graph Attention Network for DTI prediction"""
    
    def __init__(self, num_features, hidden_dim=128, num_layers=3, 
                 heads=8, dropout=0.2):
        super(GAT_DTI, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm1d(hidden_dim * heads))
        
        # Last layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Readout layers
        self.fc1 = Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 1)
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, x, edge_index, batch, return_attention=False):
        attention_weights = []
        
        # Graph attention layers
        for i in range(self.num_layers):
            if return_attention and i == self.num_layers - 1:
                x, (edge_index_out, alpha) = self.convs[i](x, edge_index, return_attention_weights=True)
                attention_weights.append(alpha)
            else:
                x = self.convs[i](x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention:
            self.attention_weights = attention_weights
        
        # Graph-level readout
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Final MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


class GraphSAGE_DTI(nn.Module):
    """GraphSAGE for DTI prediction"""
    
    def __init__(self, num_features, hidden_dim=128, num_layers=3, dropout=0.2):
        super(GraphSAGE_DTI, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(num_features, hidden_dim))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Readout layers
        self.fc1 = Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 1)
        
    def forward(self, x, edge_index, batch):
        # GraphSAGE convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level readout
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Final MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


class GIN_DTI(nn.Module):
    """Graph Isomorphism Network for DTI prediction"""
    
    def __init__(self, num_features, hidden_dim=128, num_layers=3, dropout=0.2):
        super(GIN_DTI, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        nn1 = Sequential(
            Linear(num_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            nn_hidden = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_hidden))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Readout layers
        self.fc1 = Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 1)
        
    def forward(self, x, edge_index, batch):
        # GIN convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level readout
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Final MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


class MultiModalDTI(nn.Module):
    """
    Multi-modal DTI model combining molecular graphs and protein sequences
    """
    
    def __init__(self, drug_encoder, protein_encoder, hidden_dim=256, dropout=0.2):
        super(MultiModalDTI, self).__init__()
        
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.dropout = dropout
        
        # Fusion layers
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Final prediction layers
        self.fc1 = Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 1)
        
    def forward(self, drug_data, protein_data):
        # Encode drug
        drug_embedding = self.drug_encoder(
            drug_data.x, 
            drug_data.edge_index, 
            drug_data.batch
        )
        
        # Encode protein
        protein_embedding = self.protein_encoder(protein_data)
        
        # Ensure same dimension
        if drug_embedding.shape[-1] != protein_embedding.shape[-1]:
            # Project to same dimension if needed
            drug_embedding = self.drug_projection(drug_embedding)
        
        # Cross-attention fusion
        attended_drug, _ = self.fusion_attention(
            drug_embedding.unsqueeze(0),
            protein_embedding.unsqueeze(0),
            protein_embedding.unsqueeze(0)
        )
        attended_drug = attended_drug.squeeze(0)
        
        # Combine representations
        combined = torch.cat([drug_embedding, attended_drug], dim=1)
        
        # Final prediction
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


def get_model(model_name, num_features, **kwargs):
    """Factory function to get model by name"""
    
    models = {
        'gcn': GCN_DTI,
        'gat': GAT_DTI,
        'graphsage': GraphSAGE_DTI,
        'gin': GIN_DTI
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    return models[model_name.lower()](num_features, **kwargs)
