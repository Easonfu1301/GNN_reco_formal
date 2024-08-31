import numpy as np
import pandas as pd
import torch


class HitSample:
    def __init__(self, N_track=100):
        self.N_track = N_track
        self.ifgen = False

        self.gen_mode = None
        self.gen_sample = None

        self.sample_graph = None

        pass

    def __len__(self):
        return self.N_track

    def __str__(self):
        return f"HitSample(N_track={self.N_track}, ifgen={self.ifgen})"

    def generate_sample(self, gen_mode, N_graph):
        """
        :param gen_mode: must be a dir with z_range, phi_range, ctheta_range, N_noise
        :return:
        """

        self.ifgen = True
        pass

    def save_samples(self, folder_path):
        pass

    def load_samples(self, folder_path):
        pass

    def sample2graph(self):
        pass

    def save_graphs(self, folder_path):
        pass

    def visualize_graph(self, index):
        pass
