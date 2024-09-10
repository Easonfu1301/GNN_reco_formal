from trkr.Sample import HitSample
from trkr.Graph import Graph
from trkr.Model.GCN import GCN
from trkr.Train import Train
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Create a sample object
    sample = HitSample(2000, 5)
    sample.generate_samples(1)  # here we just generate one sample, but we should generate more samples
    # sample.visualie_sample()
    # sample.save_samples("sample_store")
    # # sample.load_samples("sample_store")
    # sample.visualie_sample()


    graph = Graph(sample.get_samples())

    graph.sample2graph()
    # # sample.save_graphs("graph_store")
    # sample.load_graphs("graph_store")
    # graph.visualize_graph()

    graph0 = graph.getgraph(0)
    print(graph0.num_features)

    model = GCN(graph0.num_features, 8)
    #
    train = Train(model, graph.gen_graphs)
    # train.train(20000, True, path="pth_store")
    train.train(20000, True)
    print(sample)
    # print(train)
