import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd

default_gen_mode = {
    "z_range": [2500, 2600, 2700, 2800],
    "phi_range": [0, 2 * np.pi],
    "ctheta_range": [0.5, 1],
    "gaussian_noise": 20
}

def hit2graph(hit_df, gen_mode=default_gen_mode):
    # 加载节点和边数据
    edges = cal_edge(hit_df, gen_mode)
    nodes_df = hit_df

    # print(nodes_df)

    nodes_df["x"] = nodes_df["x"] / np.max(np.abs(nodes_df["x"]))
    nodes_df["y"] = nodes_df["y"] / np.max(np.abs(nodes_df["y"]))
    nodes_df["z"] = nodes_df["z"] / np.max(np.abs(nodes_df["z"]))

    # 提取节点特征和标签
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)
    y = torch.tensor(nodes_df.iloc[:, -1].values, dtype=torch.long)

    # 提取边的起点和终点
    edge_index = torch.tensor(edges, dtype=torch.long)
    print(edge_index)

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    return data









def cal_edge(hit_df, gen_mode=default_gen_mode):
    edge_list = []
    print(hit_df)
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
    r_A, z_A = (x0**2+y0**2) **0.5, z0
    r_B, z_B = (x1**2+y1**2) **0.5, z1
    # 将 A 的 x 和 y 分别扩展成列向量
    r_A_expanded = r_A[:, np.newaxis]
    z_A_expanded = z_A[:, np.newaxis]

    # 计算斜率矩阵
    slope_matrix = (z_B - z_A_expanded) / (r_B - r_A_expanded)
    # print(slope_matrix)

    return slope_matrix

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
    x0 = np.random.rand(10000)
    y0 = np.random.rand(10000)
    x1 = 0
    y1 = 0
    k = cal_k(x0, y0, x1, y1)
    # print((k>0).shape)
    pass
