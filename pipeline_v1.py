from trkr.Sample import HitSample
from trkr.Graph import Graph
from trkr.Model.GCN import GCN
from trkr.Train import Train
from trkr.Test_model import Test_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
import numpy as np

if __name__ == "__main__":
    # # # Create a sample object
    # sample = HitSample(2000, 100)
    # sample.generate_samples(1)  # here we just generate one sample, but we should generate more samples
    # # sample.visualie_sample()
    # # sample.save_samples("sample_store")
    # # # sample.load_samples("sample_store")
    # # sample.visualie_sample()
    #
    #
    # graph = Graph(sample.get_samples())
    #
    # graph.sample2graph()
    # # # sample.save_graphs("graph_store")
    # # sample.load_graphs("graph_store")
    # # graph.visualize_graph()
    #
    # graph0 = graph.getgraph(0)
    # print(graph0.num_features)
    #
    # model = GCN(graph0.num_features, 128)
    # #
    # train = Train(model, graph.gen_graphs)
    # # train.train(20000, True, path="pth_store")
    # train.train(2500, True, path="pth_store")
    # print(sample)
    # plt.close('all')


    ######################## Evaluate ############################


    plt.ioff()
    model = GCN(3, 128)
    sample = HitSample(2000, 100)
    sample.generate_samples(1)
    sample.visualie_sample()
    sample0 = sample.get_samples()[0]

    tst = Test_model(model, "pth_store/epoch_2000.pth", sample0)

    z = np.zeros((6, 20))
    N=100
    for tt in range(N):
        data = tst.fake_hit2graph(frac_true=0.5, frac_fake=0)
        pred, true = tst.predict(data)
        tst.draw_ROC(pred, true)

        print(pred, true)
        for idx, i in enumerate(np.linspace(0.5, 0.99, 20)):
            accuracy, po_acc, ne_acc, interpretable, frac_t, frac_f = tst.cal_acc(pred, true, data, i)
            z[0, idx] += accuracy
            z[1, idx] += po_acc
            z[2, idx] += ne_acc
            z[3, idx] += interpretable
            z[4, idx] += frac_t
            z[5, idx] += frac_f
    plt.show()
    z /= N
    plt.figure()
    plt.plot(np.linspace(0.5, 1, 20), z[0, :], 'ro-')
    plt.plot(np.linspace(0.5, 1, 20), z[1, :], 'bo-')
    plt.plot(np.linspace(0.5, 1, 20), z[2, :], 'go-')
    plt.plot(np.linspace(0.5, 1, 20), z[3, :], 'yo-')
    plt.plot(np.linspace(0.5, 1, 20), z[4, :], 'mo-')
    plt.plot(np.linspace(0.5, 1, 20), z[5, :], 'co-')

    plt.legend(["Accuracy", "Positive Accuracy", "Negative Accuracy", "Interpretable", "Fraction True", "Fraction Fake"])

    plt.show()


    # print(train)
