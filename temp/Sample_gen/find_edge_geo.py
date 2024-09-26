import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
from temp.Sample_gen.mode_default import gen_mode


from temp.Sample_gen.find_edge_truth import get_true_edge


# def hit2graph(hit_df):
#     # 加载节点和边数据
#     edges = cal_edge(hit_df)
#     return torch.tensor(edges)


def get_geo_edge(hit_df, kmax = 0.05):
    hit_df = hit_df.copy()

    edge_list = []
    # print(hit_df)
    z_mode = gen_mode['z_range']
    # hit_df = hit_df.copy().sort_values(by=['particle_index', 'z'])


    hit_df_l0 = hit_df[hit_df['z'] == z_mode[0]]
    hit_df_l1 = hit_df[hit_df['z'] == z_mode[1]]
    hit_df_l2 = hit_df[hit_df['z'] == z_mode[2]]
    hit_df_l3 = hit_df[hit_df['z'] == z_mode[3]]


    # print(hit_df_l0)
    # print(hit_df_l1)
    # print(hit_df_l2)
    # print(hit_df_l3)


    # st0 = 0
    # st1 = len(hit_df_l0)
    # st2 = st1 + len(hit_df_l1)
    # st3 = st2 + len(hit_df_l2)
    # # print(st0, st1, st2, st3)

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

    edge01 = cal_edge_index(k01, hit_df_l0, hit_df_l1, kmax)
    # print(k01)
    edge02 = cal_edge_index(k02, hit_df_l0, hit_df_l2, kmax)
    edge03 = cal_edge_index(k03, hit_df_l0, hit_df_l3, kmax)
    edge12 = cal_edge_index(k12, hit_df_l1, hit_df_l2, kmax)
    edge13 = cal_edge_index(k13, hit_df_l1, hit_df_l3, kmax)
    edge23 = cal_edge_index(k23, hit_df_l2, hit_df_l3, kmax)

    edge_all = np.hstack((edge01, edge02, edge03, edge12, edge13, edge23), dtype=np.int64)
    edge_all_label = get_label(edge_all, hit_df)
    # print(edge_all.shape)
    return torch.tensor(edge_all), torch.tensor(edge_all_label)


def cal_edge_index(k, stk1, stk2, kmax):
    # print(stk2)
    # print(stk2["x"].iloc[10])



    # exit(0)
    # print(np.array(stk1["hit_id"].iloc[:]-1))
    po_edge_st = np.array(stk1["hit_id"].iloc[:])[np.where(np.abs(k) < kmax)[0]]
    # print("po_edge_st", po_edge_st)
    po_edge_ed = np.array(stk2["hit_id"].iloc[:])[np.where(np.abs(k) < kmax)[1]]
    # print(po_edge_st)
    # print(po_edge_ed)

    po_edge = np.vstack((po_edge_st, po_edge_ed))
    # print(po_edge)
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


def get_label(edge_geo, hit_df):
    true_edge = get_true_edge(hit_df).numpy()

    edge_num = edge_geo.shape[1]


    edge_label = np.zeros(edge_num)

    for i in range(edge_num):
        if np.any(np.all(edge_geo[:, i].reshape(2,1) == true_edge, axis=0)):
            edge_label[i] = 1

    print(edge_label)

    return edge_label



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
    from temp.Sample import Sample
    from temp.Sample_gen.find_edge_truth import get_true_edge
    from temp.Sample_gen.gen_graph import construct_graph, visualize_graph
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    sample = Sample(1000)
    sample.generate_sample(1)
    sam = sample.get_sample(0)
    # print(sam)
    # sample.visualize_sample(0)

    true_edge = get_true_edge(sam)
    geo_edge = get_geo_edge(sam)

    print(true_edge.shape)
    print(geo_edge.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    graph = construct_graph(sam, true_edge)
    # visualize_graph(graph)

    # ax2 = fig.add_subplot(122)
    graph2 = construct_graph(sam, geo_edge)
    # visualize_graph(graph2)

    plt.show()

