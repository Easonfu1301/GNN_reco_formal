import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd


def create_node_edge_data(sample_df):
    df_sorted = sample_df.sort_values(by=['particle_index', 'z'])
    # print(df_sorted)

    edge_list = []
    last_particle_index = 0
    for idx, _ in enumerate(df_sorted["hit_id"].iloc):
        if df_sorted["particle_index"].iloc[idx] == last_particle_index and last_particle_index != 0:
            edge = [df_sorted["hit_id"].iloc[idx - 1] - 1, df_sorted["hit_id"].iloc[idx] - 1]
            edge_list.append(edge)
        last_particle_index = df_sorted["particle_index"].iloc[idx]

    # print(df_sorted["hit_id"].iloc[1])
    edge_df = pd.DataFrame(np.array(edge_list), columns=['start', 'end'])
    # print(edge_df)
    # edge_df.to_csv('edge_test.csv', index=False)
    sample_df_cp = sample_df.copy()
    return sample_df_cp, edge_df

def hit2graph(sample_df):
    # 加载节点和边数据
    nodes_df, edges_df = create_node_edge_data(sample_df)

    # print(nodes_df)

    nodes_df["x"] = nodes_df["x"] / np.max(np.abs(nodes_df["x"]))
    nodes_df["y"] = nodes_df["y"] / np.max(np.abs(nodes_df["y"]))
    nodes_df["z"] = nodes_df["z"] / np.max(np.abs(nodes_df["z"]))

    # 提取节点特征和标签
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)
    y = torch.tensor(nodes_df.iloc[:, -1].values, dtype=torch.long)

    # 提取边的起点和终点
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
    print(edge_index)
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    return data