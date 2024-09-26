import numpy as np
from torch_geometric.data import Data

from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
from temp.Sample import Sample
from temp.Train import Train
from temp.Evaluate import RestoreTrk
from temp.Sample_gen.find_edge_truth import get_true_edge
from temp.Sample_gen.transform_coord import get_node_tensor



def get_graph_edge():
    pass


# def get_true_edge():
#     pass
def construct_graph(hit_df, edges, edge_label=None, edge_label_index=None):

    nodes = get_node_tensor(hit_df)



    graph = Data(x=nodes, edge_index=edges, edge_label_index=edge_label_index, edge_label=edge_label)
    return graph


def get_node():
    pass


def sample_positive_negative_link():
    pass


def visualize_graph(graph):
    print(graph)
    G = to_networkx(graph, to_undirected=False)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=['skyblue'], node_size=80, font_size=10, font_color='black')
    # plt.show()
    pass

if __name__ == "__main__":
    sample = Sample(10)
    sample.generate_sample(1)
    sam = sample.get_sample(0)
    # print(sam)
    # sample.visualize_sample(0)

    true_edge = get_true_edge(sam)

    graph = construct_graph(sam, true_edge)
    visualize_graph(graph)

    # g = Graph()
    #
    # t_graph, v_graph = g.construct_train_graph(sample.get_sample(0))
