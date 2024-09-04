import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib
from trkr.TRecA.Hit2graph_truth import hit2graph
from torch_geometric.utils import to_networkx
import networkx as nx

matplotlib.use('tkAgg')

default_gen_mode = {
    "z_range": [2500, 2600, 2700, 2800],
    "phi_range": [0, 2 * np.pi],
    "ctheta_range": [0.5, 1],
    "gaussian_noise": 20
}


class HitSample:
    def __init__(self, N_track=100, N_noise=0):
        self.N_track = N_track
        self.N_noise = N_noise
        self.ifgen = False
        self.ifgraph = False

        self.gen_mode = None
        self.gen_samples = None

        self.gen_graphs = None

        pass

    def __len__(self):
        return self.N_track

    def __str__(self):
        return f"HitSample(N_track={self.N_track}, ifgen={self.ifgen})"

    def warning(self, message):
        print(f"\033[93m{message}\033[0m")

    def Log(self, text):
        print(f"\033[94m{text}\033[0m")

    def generate_samples(self, N_graph, gen_mode=None):
        """
        :param N_graph: generate N_graph samples
        :param gen_mode: must be a dir with z_range, phi_range, ctheta_range, N_noise, gaussian_noise
        :return:
        """

        if gen_mode is None:
            gen_mode = default_gen_mode
            self.gen_mode = gen_mode

        samples = []
        for i in range(N_graph):
            sample = self.generate_one_sample(gen_mode)
            samples.append(sample)

        self.gen_samples = samples

        self.ifgen = True
        pass

    def generate_one_sample(self, gen_mode):
        """
        :param gen_mode: must be a dir with z_range, phi_range, ctheta_range, N_noise, gaussian_noise
        :return:
        """

        # check the gen_mode key
        for key in default_gen_mode.keys():
            if key not in gen_mode:
                raise ValueError(f"gen_mode must have key {key}")

        hit_num = self.N_track

        # Generate the hit data by random sampling phi theta
        rand_angle = np.random.rand(hit_num, 2)
        phi = rand_angle[:, 0] * (gen_mode["phi_range"][1] - gen_mode["phi_range"][0]) + gen_mode["phi_range"][0]
        cos_theta = rand_angle[:, 1] * (gen_mode["ctheta_range"][1] - gen_mode["ctheta_range"][0]) + \
                    gen_mode["ctheta_range"][0]
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        r = np.zeros((hit_num, 4))
        x = np.zeros((hit_num, 4))
        y = np.zeros((hit_num, 4))
        for idx, z in enumerate(gen_mode["z_range"]):
            r[:, idx] = z / cos_theta
            x[:, idx] = r[:, idx] * np.cos(phi) * sin_theta + np.random.normal(0, 20, hit_num)
            y[:, idx] = r[:, idx] * np.sin(phi) * sin_theta + np.random.normal(0, 20, hit_num)

        particle_index = np.linspace(1, hit_num, hit_num, dtype=int)

        hit = np.zeros((hit_num * 4, 5))
        hit[:, 0] = np.linspace(1, 4 * hit_num, 4 * hit_num, dtype=int)
        for layer, z in enumerate(gen_mode["z_range"]):
            hit[layer * hit_num:(layer + 1) * hit_num, 1] = x[:, layer]
            hit[layer * hit_num:(layer + 1) * hit_num, 2] = y[:, layer]
            hit[layer * hit_num:(layer + 1) * hit_num, 3] = z
            hit[layer * hit_num:(layer + 1) * hit_num, 4] = particle_index
        hit_df = pd.DataFrame(hit, columns=['hit_id', 'x', 'y', 'z', 'particle_index'])
        hit_df['particle_index'] = hit_df['particle_index'].astype(int)
        hit_df['hit_id'] = hit_df['hit_id'].astype(int)

        if self.N_noise > 0 and self.N_noise is not None:
            hit_df = self.gen_noise(hit_df)

        self.ifgen = True
        return hit_df

    def gen_noise(self, hit_df):
        N_noise = self.N_noise*4
        if N_noise == 0:
            return hit_df
        squrerange = np.max(self.gen_mode["z_range"]) / np.tan(self.gen_mode["ctheta_range"][0]) * 1.2
        x = np.random.random(N_noise) * squrerange * 2 - squrerange
        y = np.random.random(N_noise) * squrerange * 2 - squrerange
        z = np.random.randint(0, 3, N_noise)
        z = np.array(self.gen_mode["z_range"])[z]
        # print(z)
        particle_index = 0
        df = pd.DataFrame({'hit_id': np.linspace(1 + self.N_track*4, N_noise + self.N_track*4, N_noise, dtype=int),
                           'x': x,
                           'y': y,
                           'z': z,
                           'particle_index': particle_index})
        hit_df = pd.concat([hit_df, df], ignore_index=True)
        # print(hit_df)
        return hit_df


    def visualie_sample(self, index=0):
        if not self.ifgen:
            self.warning("No sample generated, Therefore, no sample to visualize")
            return -1

        sample = self.gen_samples[index]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sample['x'], sample['y'], sample['z'])
        ax.scatter(0, 0, 0)

        # 创建 x 和 y 数据
        squrerange = np.max(self.gen_mode["z_range"]) / np.tan(self.gen_mode["ctheta_range"][0]) * 1.2
        x = np.linspace(-squrerange, squrerange, 2)
        y = np.linspace(-squrerange, squrerange, 2)
        X, Y = np.meshgrid(x, y)
        for Z in self.gen_mode["z_range"]:
            Z = np.ones(X.shape) * Z

            # 创建图形

            # 绘制平面
            ax.plot_surface(X, Y, Z, alpha=0.2, rstride=100, cstride=100)

        # 设置轴标签
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()

        plt.show()

        pass

    def save_samples(self, folder_path):
        if not self.ifgen:
            self.warning("No sample generated, Therefore, no sample to save")
            return -1
        for i, sample in enumerate(self.gen_samples):
            sample.to_csv(f"{folder_path}/sample_{i}.csv", index=False)

        # export the gen_mode
        with open(f"{folder_path}/gen_mode.json", "w") as f:
            json.dump(self.gen_mode, f)

    def load_samples(self, folder_path):
        if self.ifgen:
            self.warning("Using load sample... \n \t Notice the original sample will be replaced")

        samples = []
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                sample = pd.read_csv(os.path.join(folder_path, file))
                samples.append(sample)
        self.gen_samples = samples

        # load the gen_mode
        with open(f"{folder_path}/gen_mode.json", "r") as f:
            self.gen_mode = json.load(f)

        self.ifgen = True
        self.N_track = len(samples[0]["x"]) // 4

        self.Log(f"Load {len(samples)} samples from '{folder_path}' and set N_track to {self.N_track}")

        pass

    def sample2graph(self):
        graphs = []
        self.Log(f"Converting {len(graphs)} samples to graphs")
        for sample in tqdm(self.gen_samples):
            graphs.append(hit2graph(sample))
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
