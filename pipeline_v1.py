from trkr.Sample import HitSample
from trkr.Model.GCN import GCN
from trkr.Train import Train
import matplotlib.pyplot as plt
import numpy as np






if __name__ == "__main__":
    # Create a sample object
    sample = HitSample(10000, 10000)
    sample.generate_samples(1)
    # sample.visualie_sample()
    # sample.save_samples("sample_store")
    # # sample.load_samples("sample_store")
    # sample.visualie_sample()


    sample.sample2graph()
    # sample.save_graphs("graph_store")
    # sample.load_graphs("graph_store")
    # sample.visualize_graph()

    graph = sample.getgraph(0)

    model = GCN(graph.num_features, 8)

    train = Train(model, sample.gen_graphs)
    train.train(20000, True, path="pth_store")
    print(sample)
    print(train)
