import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd

def predict_link(hit_df):
    link_list = []
    for idx, _ in enumerate(hit_df["hit_id"].iloc):
        link = [hit_df["hit_id"].iloc[idx - 1] - 1, hit_df["hit_id"].iloc[idx] - 1]
        link_list.append(link)




    pass



