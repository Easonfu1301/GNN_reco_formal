from trkr.Sample import HitSample
from trkr.Model.GCN import GCN
from trkr.Train import Train
from trkr.Evaluate import RestoreTrk
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Create a sample object
    sample = HitSample(1000, 0)
    sample.generate_samples(1)  # here we just generate one sample, but we should generate more samples
    # sample.visualie_sample()
    # sample.save_samples("sample_store")
    # # sample.load_samples("sample_store")
    # sample.visualie_sample()

    sample.sample2graph()
    # # sample.save_graphs("graph_store")
    # sample.load_graphs("graph_store")
    # sample.visualize_graph()

    graph = sample.getgraph(0)

    model = GCN(graph.num_features, 8)

    train = Train(model, sample.gen_samples)
    train.train(2200, True, path="pth_store")

    # sample = HitSample(1000, 0)
    # sample.generate_samples(1)
    model = GCN(3, 8)


    eva = RestoreTrk(model, r"D:\files\pyproj\GNN_reco_formal\pth_store\epoch_2000.pth", sample.gen_samples[0])
    eva.restore_track()



    print(sample)
    # print(train)
