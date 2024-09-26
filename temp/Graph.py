import numpy as np
from torch_geometric.data import Data
from temp.Sample import Sample
from temp.Train import Train
from temp.Evaluate import RestoreTrk
from temp.Sample_gen.find_edge_truth import get_true_edge
from temp.Sample_gen.find_edge_geo import get_geo_edge
from temp.Sample_gen.transform_coord import get_node_tensor
from temp.Sample_gen.sample_edges_from_true_edge import sample_edges_from_true_edge


class Graph:
    def __init__(self):
        pass

    def get_true_edges(self, hit_sample):
        true_edge = get_true_edge(hit_sample)
        return true_edge

    def get_graph_edges(self, hit_sample):  # based on the geometry
        graph_edges, _ = get_geo_edge(hit_sample, 0.05)
        return graph_edges

    def get_graph_nodes(self, hit_sample):
        nodes = get_node_tensor(hit_sample)
        return nodes

    def sample_positive_negative_edges(self, hit_df, true_edges, ratio):
        max_index = len(hit_df)
        train_edge_label_index, test_edge_label_index, train_edge_label, test_edge_label = sample_edges_from_true_edge(true_edges, ratio, maxindex=max_index)
        return train_edge_label_index, test_edge_label_index, train_edge_label, test_edge_label

    def construct_full_graph(self, hit_sample):
        nodes = self.get_graph_nodes(hit_sample)
        edges = self.get_graph_edges(hit_sample)

        graph = Data(x=nodes, edge_index=edges)
        return graph

    def construct_train_graph(self, hit_sample, ratio=0.7):
        nodes = self.get_graph_nodes(hit_sample)
        edges = self.get_graph_edges(hit_sample)
        true_edges = self.get_true_edges(hit_sample)
        max_index = len(hit_sample)


        train_edge_label_index, test_edge_label_index, train_edge_label, test_edge_label = self.sample_positive_negative_edges(hit_sample, true_edges, ratio)

        train_graph = Data(x=nodes, edge_index=edges, edge_label_index=train_edge_label_index,
                           edge_label=train_edge_label)
        test_graph = Data(x=nodes, edge_index=edges, edge_label_index=test_edge_label_index,
                          edge_label=test_edge_label)
        return train_graph, test_graph
