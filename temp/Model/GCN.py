import numpy as np
import torch
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = TransformerConv(in_channels, 128)
        self.conv2 = TransformerConv(128, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

        self.classifier = torch.nn.Linear(2 * out_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        return self.forward(x, edge_index)

    # def decode(self, z, edge_index):
    #     # b = z.cpu().detach().numpy()
    #     # print(b.shape)
    #     # print(edge_index[0].cpu().detach().numpy())
    #
    #     # print(edge_index)
    #     # print((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1).shape)
    #
    #     return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode(self, z, edge_index):
        total_edge_index = torch.cat(
            [z[edge_index[0]] * z[edge_index[1]]], dim=1)
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        # de = self.classifier(edge_features)
        return self.classifier(edge_features).T[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(3, 16).to(device)
