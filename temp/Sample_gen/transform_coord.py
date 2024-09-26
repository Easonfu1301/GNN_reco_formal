import numpy as np
import torch






def get_node_tensor(nodes_df):# do the preprocessing

    nodes_df = nodes_df.copy()

    nodes_df["x"] = nodes_df["x"] / np.max(np.abs(nodes_df["x"]))
    nodes_df["y"] = nodes_df["y"] / np.max(np.abs(nodes_df["y"]))
    nodes_df["z"] = nodes_df["z"] / np.max(np.abs(nodes_df["z"]))

    # 提取节点特征和标签
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)

    return x



