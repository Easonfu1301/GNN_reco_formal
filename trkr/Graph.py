import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib
from trkr.TRecA.Hit2graph_truth import hit2graph
import time

from trkr.TRecA.Hit2graph_potential import hit2graph as hg_test

from torch_geometric.utils import to_networkx
import networkx as nx


class Graph:
    def __init__(self, samples):
        self.samples = samples

    def warning(self, message):
        print(f"\033[93m{message}\033[0m")

    def Log(self, text):
        print(f"\033[94m{text}\033[0m")

    def sample2graph(self):
        graphs = []
        self.Log(f"Converting {len(graphs)} samples to graphs")
        for sample in tqdm(self.samples):
            graphs.append(hit2graph(sample))
            # graphs.append(hg_test(sample, self.gen_mode))
            # t = time.time()
            # hg_test(sample, self.gen_mode)
            # print(time.time() - t)
        self.gen_graphs = graphs
        self.Log(f"Converting Done! {len(graphs)} graphs generated")
        self.ifgraph = True

        pass

    def getgraph(self, index=0):
        if not self.ifgraph:
            self.warning("No graph generated, Therefore, no graph to return")
            return -1
        return self.gen_graphs[index]

    def save_graphs(self, folder_path):
        if not self.ifgraph:
            self.warning("No graph generated, Therefore, no graph to save")
            return -1
        # save the PYG DATA
        for i, graph in enumerate(self.gen_graphs):
            torch.save(graph, f"{folder_path}/graph_{i}.pt")

        pass

    def load_graphs(self, folder_path):
        if self.ifgraph:
            self.warning("Using load graph... \n \t Notice the original graph will be replaced")

        graphs = []
        for file in os.listdir(folder_path):
            if file.endswith(".pt"):
                graph = torch.load(os.path.join(folder_path, file))
                graphs.append(graph)
        self.gen_graphs = graphs

        self.ifgraph = True
        self.Log(f"Load {len(graphs)} graphs from '{folder_path}'")

        pass

    def visualize_graph(self, index=0):
        if not self.ifgraph:
            self.warning("No graph generated, Therefore, no graph to visualize")
            return -1
        graph = self.gen_graphs[index]
        # visualize the graph with networkx

        G = to_networkx(graph, to_undirected=False)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=['skyblue'], node_size=80, font_size=10, font_color='black')
        plt.show()

        pass
