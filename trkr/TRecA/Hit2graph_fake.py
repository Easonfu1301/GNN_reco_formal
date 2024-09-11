import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd


def create_node_edge_data(sample_df, frac_true=0.7, frac_fake=0, test_num=100):
    df_sorted = sample_df.sort_values(by=['particle_index', 'z'])
    # print(df_sorted)

    true_edge_list = []
    visible_true_edge_list = []
    visible_fake_edge_list = []

    test_positive = []
    test_negative = []

    po_label = []
    ne_label = []

    last_particle_index = 0
    for idx, _ in enumerate(df_sorted["hit_id"].iloc):
        if df_sorted["particle_index"].iloc[idx] == last_particle_index and last_particle_index != 0:
            edge = [df_sorted["hit_id"].iloc[idx - 1] - 1, df_sorted["hit_id"].iloc[idx] - 1]
            true_edge_list.append(edge)
            if np.random.rand() < frac_true:
                visible_true_edge_list.append(edge)

            if np.random.rand() < 0.5 and len(test_positive) < test_num:
                test_positive.append(edge)
                po_label.append(1)

        # else:
        #     edge = [df_sorted["hit_id"].iloc[idx - 1] - 1, df_sorted["hit_id"].iloc[idx] - 1]
        #     if np.random.rand() < frac_fake:
        #         fake_edge_list.append(edge)
        #
        #     if np.random.rand() < 0.5 and len(test_negative) < test_num:
        #         test_negative.append(edge)
        #         ne_label.append(0)

        last_particle_index = df_sorted["particle_index"].iloc[idx]
    print(frac_fake)
    visible_fake_edge_list = np.random.randint(0, len(df_sorted), (int(len(true_edge_list) * frac_fake), 2))
    test_negative = np.random.randint(0, len(df_sorted), (len(test_positive), 2))
    ne_label = np.zeros(len(test_negative))

    # print(df_sorted["hit_id"].iloc[1])
    # true_edge_list = pd.DataFrame(np.array(true_edge_list), columns=['start', 'end'])
    # visible_true_edge_list = pd.DataFrame(np.array(visible_true_edge_list), columns=['start', 'end'])
    # fake_edge_list = pd.DataFrame(np.array(fake_edge_list), columns=['start', 'end'])
    # print(edge_df)
    # edge_df.to_csv('edge_test.csv', index=False)
    true_edge_list = np.array(true_edge_list)
    visible_true_edge_list = np.array(visible_true_edge_list)
    visible_fake_edge_list = np.array(visible_fake_edge_list)

    print(true_edge_list.shape)
    print(visible_true_edge_list.shape)
    print(visible_fake_edge_list.shape)

    sample_df_cp = sample_df.copy()

    test_index = np.hstack([np.array(test_positive).T, np.array(test_negative).T])
    test_label = np.hstack([np.array(po_label), np.array(ne_label)])

    visible_index = np.vstack([np.array(visible_true_edge_list), np.array(
        visible_fake_edge_list)]).T  # if len(fake_edge_list) > 0 else np.array(visible_true_edge_list).T
    print(visible_index.shape)

    return sample_df_cp, true_edge_list, visible_index, test_index, test_label


def hit2graph_fake(sample_df, frac_true=0.7, frac_fake=0):
    # 加载节点和边数据
    nodes_df, true_edge_list, visible_index, test_index, test_label = create_node_edge_data(sample_df,
                                                                                            frac_true=frac_true,
                                                                                            frac_fake=frac_fake,
                                                                                            test_num=sample_df.shape[0])

    # print(nodes_df)

    nodes_df["x"] = nodes_df["x"] / np.max(np.abs(nodes_df["x"]))
    nodes_df["y"] = nodes_df["y"] / np.max(np.abs(nodes_df["y"]))
    nodes_df["z"] = nodes_df["z"] / np.max(np.abs(nodes_df["z"]))

    # 提取节点特征和标签
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)
    # y = torch.tensor(nodes_df.iloc[:, -1].values, dtype=torch.long)

    # 提取边的起点和终点
    visible_index = torch.tensor(visible_index, dtype=torch.long)
    test_index = torch.tensor(test_index, dtype=torch.long)
    test_label = torch.tensor(test_label, dtype=torch.long)
    # print(edge_index)
    # 创建图数据对象
    data = Data(x=x, edge_index=visible_index, edge_label_index=test_index, edge_label=test_label)
    return data
