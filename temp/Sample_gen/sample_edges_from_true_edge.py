import numpy as np
import torch


def sample_edges_from_true_edge(true_edges, ratio, maxindex):
    true_edges = true_edges.numpy()
    num_true_edges = true_edges.shape[1]
    num_train_edges = int(num_true_edges * ratio)

    train_edges = np.zeros((2, num_train_edges))
    test_edges = np.zeros((2, num_true_edges - num_train_edges))

    random_index = np.random.choice(range(0, num_true_edges), num_train_edges, replace=False)
    remaining_index = np.setdiff1d(range(0, num_true_edges), random_index)


    train_positive_edges = true_edges[:, random_index]
    test_positive_edges = true_edges[:, remaining_index]


    negative_edges = sample_negative_sample(num_true_edges, true_edges, maxindex)

    train_negative_edges = negative_edges[:, random_index]
    test_negative_edges = negative_edges[:, remaining_index]



    train_edges = np.hstack((train_positive_edges, train_negative_edges))
    test_edges = np.hstack((test_positive_edges, test_negative_edges))

    print(train_edges.shape)
    print(test_edges.shape)

    train_edges_label = np.hstack((np.ones((1, num_train_edges)), np.zeros((1, num_train_edges))))
    test_edges_label = np.hstack((np.ones((1, num_true_edges - num_train_edges)), np.zeros((1, num_true_edges - num_train_edges))))
    print(train_edges_label.shape)
    print(test_edges_label.shape)

    train_edges = torch.tensor(train_edges, dtype=torch.int64)
    test_edges = torch.tensor(test_edges, dtype=torch.int64)
    train_edges_label = torch.tensor(train_edges_label, dtype=torch.int64)[0]
    test_edges_label = torch.tensor(test_edges_label, dtype=torch.int64)[0]




    return train_edges, test_edges, train_edges_label, test_edges_label


def sample_negative_sample(number, true_edges, maxindex):

    negative_sample = np.zeros((2, number))
    # print(negative_sample.shape)
    count = 0

    while True:
        negative_edges = np.random.randint(0, maxindex, (2, 1))
        # print(negative_edges.shape)
        # print(np.any(np.all(negative_edges == true_edges, axis=0)))
        if not np.any(np.all(negative_edges == true_edges, axis=0)) and not np.any(np.all(negative_edges == negative_sample, axis=0)):
            negative_sample[:, count] = negative_edges[:, 0]
            count += 1
        if count == number:
            break

    return negative_sample

if __name__ == "__main__":
    from temp.Sample import Sample
    from temp.Sample_gen.find_edge_truth import get_true_edge
    from temp.Sample_gen.gen_graph import construct_graph, visualize_graph
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')
    sample = Sample(1000)

    sample.generate_sample(1)

    sam = sample.get_sample(0)
    print(sam)
    max_index = len(sam)-1
    print(max_index)
    # print(sam)
    # sample.visualize_sample(0)

    true_edge = get_true_edge(sam)
    sample_edges_from_true_edge(true_edge.numpy(), 0.8, 1000)