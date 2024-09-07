import numpy as np
import torch
from torch_geometric.data import Data
from trkr.TRecA.Hit2graph_truth import hit2graph as hg_truth
import random
import trkr.Sample as Sample

import pandas as pd

default_gen_mode = {
    "z_range": [2500, 2600, 2700, 2800],
    "phi_range": [0, 2 * np.pi],
    "ctheta_range": [0.5, 1],
    "gaussian_noise": 0.05
}


def hit2graph(hit_df, gen_mode=default_gen_mode):
    # 加载节点和边数据
    edges = cal_edge(hit_df.copy(), gen_mode)
    nodes_df = cal_node(hit_df.copy())

    # print(nodes_df)



    # 提取节点特征和标签
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)
    y = torch.tensor(nodes_df.iloc[:, -1].values, dtype=torch.long)

    # 提取边的起点和终点
    edge_index = torch.tensor(edges, dtype=torch.long)
    # print(edge_index)

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    return data

def cal_node(hit_df):
    hit_df["x"] = hit_df["x"] / np.max(np.abs(hit_df["x"]))
    hit_df["y"] = hit_df["y"] / np.max(np.abs(hit_df["y"]))
    hit_df["z"] = hit_df["z"] / np.max(np.abs(hit_df["z"]))
    return hit_df



def cal_edge(hit_df, gen_mode):
    print(hit_df)
    edge_list = []
    # print(hit_df)
    # print(hit_df[hit_df['z'] == 2500.])
    z_mode = gen_mode['z_range']

    hit_df_l0 = hit_df[hit_df['z'] == z_mode[0]]
    hit_df_l1 = hit_df[hit_df['z'] == z_mode[1]]
    hit_df_l2 = hit_df[hit_df['z'] == z_mode[2]]
    hit_df_l3 = hit_df[hit_df['z'] == z_mode[3]]

    st0 = 0
    st1 = len(hit_df_l0)
    st2 = st1 + len(hit_df_l1)
    st3 = st2 + len(hit_df_l2)

    x0 = hit_df_l0['x'].values
    y0 = hit_df_l0['y'].values
    z0 = hit_df_l0['z'].values

    x1 = hit_df_l1['x'].values
    y1 = hit_df_l1['y'].values
    z1 = hit_df_l1['z'].values

    x2 = hit_df_l2['x'].values
    y2 = hit_df_l2['y'].values
    z2 = hit_df_l2['z'].values

    x3 = hit_df_l3['x'].values
    y3 = hit_df_l3['y'].values
    z3 = hit_df_l3['z'].values

    k0 = cal_k(x0, y0, z0, 0, 0, 0)
    k1 = cal_k(x1, y1, z1, 0, 0, 0)
    k2 = cal_k(x2, y2, z2, 0, 0, 0)

    k01 = cal_k(x0, y0, z0, x1, y1, z1) - k0
    k02 = cal_k(x0, y0, z0, x2, y2, z2) - k0
    k03 = cal_k(x0, y0, z0, x3, y3, z3) - k0
    k12 = cal_k(x1, y1, z1, x2, y2, z2) - k1
    k13 = cal_k(x1, y1, z1, x3, y3, z3) - k1
    k23 = cal_k(x2, y2, z2, x3, y3, z3) - k2

    edge01 = cal_edge_index(k01, st0, st1)
    edge02 = cal_edge_index(k02, st0, st2)
    edge03 = cal_edge_index(k03, st0, st3)
    edge12 = cal_edge_index(k12, st1, st2)
    edge13 = cal_edge_index(k13, st1, st3)
    edge23 = cal_edge_index(k23, st2, st3)

    edge_all = np.hstack((edge01, edge02, edge03, edge12, edge13, edge23))
    print(edge_all.shape)
    return edge_all


def cal_edge_index(k, stk1, stk2):
    po_edge_st = np.where(np.abs(k) < 0.05)[0] + stk1
    po_edge_ed = np.where(np.abs(k) < 0.05)[1] + stk2

    po_edge = np.vstack((po_edge_st, po_edge_ed))
    return po_edge


def cal_k(x0, y0, z0, x1, y1, z1):
    # A 和 B 分别是形状为 (n, 2) 和 (m, 2) 的二维数组
    r_A, z_A = (x0 ** 2 + y0 ** 2) ** 0.5, z0
    r_B, z_B = (x1 ** 2 + y1 ** 2) ** 0.5, z1
    # 将 A 的 x 和 y 分别扩展成列向量
    r_A_expanded = r_A[:, np.newaxis]
    z_A_expanded = z_A[:, np.newaxis]

    # 计算斜率矩阵
    slope_matrix = (z_B - z_A_expanded) / (r_B - r_A_expanded)
    # print(slope_matrix)

    return slope_matrix


def generate_train_data(hit_df, gen_mode, frac=0.7):
    data_truth = hg_truth(hit_df.copy())
    true_edge = data_truth.edge_index
    nodes_df = cal_node(hit_df.copy())
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)


    edge_potential = cal_edge(hit_df, gen_mode)
    edge_potential = torch.tensor(edge_potential, dtype=torch.long)

    _, n_edge = edge_potential.shape
    n_train = int(n_edge * frac)
    selected_numbers = random.sample(range(0, n_edge), n_train)

    positive_edge_index = edge_potential[:, selected_numbers]
    negative_edge_index = sample_n_negative_sample(edge_potential, n_train)
    edge_index = positive_edge_index

    edge_label = torch.zeros((2, n_train * 2), dtype=torch.long)
    edge_label_index = torch.zeros((2, n_train * 2), dtype=torch.long)


    train_data = Data(x=x, edge_index=edge_index, edge_label=edge_label, edge_label_index = edge_label_index)
    print(train_data)

    return train_data






def sample_n_negative_sample(edge_list, N):
    pass


# def cal_k2(x0, y0, z0, x1, y1, z1):
#     # A 和 B 分别是形状为 (n, 2) 和 (m, 2) 的二维数组
#     x_A, y_A = x0, y0
#     x_B, y_B = x1, y1
#     # 将 A 的 x 和 y 分别扩展成列向量
#     x_A_expanded = x_A[:, np.newaxis]
#     y_A_expanded = y_A[:, np.newaxis]
#
#     # 计算斜率矩阵
#     slope_matrix = (y_B - y_A_expanded) / (x_B - x_A_expanded)
#     # print(slope_matrix)
#
#     return slope_matrix


if __name__ == '__main__':
    sample = Sample.HitSample(100, 0)
    sample.generate_samples(1)  # here we just generate one sample, but we should generate more samples
    sample0 = sample.gen_samples[0]

    generate_train_data(sample0, default_gen_mode, 0.7)
