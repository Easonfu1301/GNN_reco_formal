import numpy as np
from temp.Sample import Sample
from temp.Graph import Graph
from temp.Train import Train
from temp.Evaluate import RestoreTrk
from temp.TRecA.restore_acc import restore_acc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')







if __name__ == "__main__":
    sample = Sample(1000)
    sample.generate_sample(1)
    sam = sample.get_sample(0)
    # print(sam)
    # sample.visualize_sample(0)

    g = Graph()
    #
    t_graph, v_graph = g.construct_train_graph(sample.get_sample(0))
    #
    t = Train(train_data=t_graph, test_data=v_graph)
    t.train(2000)
    t.save_model("model.pth")
    #
    s_test = Sample(1000)
    s_test.generate_sample(1)
    # print(s_test.get_sample(0)[s_test.get_sample(0)["z"] == 2652.5])

    #
    r = RestoreTrk("model.pth")
    r.judge_all_edge(s_test.get_sample(0))
    df0123 = r.find_best_track_seed(s_test.get_sample(0))

    print(df0123)
    # print(np.max(df0123["hit1"]))
    # print(np.max(df0123["hit2"]))
    # print(np.max(df0123["hit3"]))
    # print(np.max(df0123["hit4"]))

    # r.restore()

    df_with_score = restore_acc(s_test.get_sample(0), df0123)

    score = df_with_score["score"].values
    print(score)
    fig = plt.figure()
    plt.ioff()
    plt.hist(score)
    plt.show()








