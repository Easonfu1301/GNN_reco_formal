import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd


def create_node_edge_data(sample_df):
    df_sorted = sample_df.sort_values(by=['particle_index', 'z'])
    # print(df_sorted)

    edge_list = []
    idx = 0
    while idx < len(df_sorted["particle_index"]):
        if df_sorted["particle_index"].iloc[idx] == 0:
            idx += 1
            continue
        tmp_idx = idx
        while True:
            if tmp_idx == len(df_sorted):
                break
            if df_sorted["particle_index"].iloc[idx] == df_sorted["particle_index"].iloc[tmp_idx]:
                tmp_idx += 1
            else:
                break


        for i in range(idx, tmp_idx):
            for j in range(i + 1, tmp_idx):
                if i != j:
                    edge = [df_sorted["hit_id"].iloc[i] - 1, df_sorted["hit_id"].iloc[j] - 1]
                    # print(edge)
                    edge_list.append(edge)

        idx = tmp_idx








        # if df_sorted["particle_index"].iloc[idx] == last_particle_index and last_particle_index != 0:
        #     edge = [df_sorted["hit_id"].iloc[idx - 1] - 1, df_sorted["hit_id"].iloc[idx] - 1]
        #     edge_list.append(edge)
        # last_particle_index = df_sorted["particle_index"].iloc[idx]

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
    # print(edge_index)
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    return data