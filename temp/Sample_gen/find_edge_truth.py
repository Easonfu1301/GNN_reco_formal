import numpy as np
import torch
# from temp.Sample import Sample
# from temp.Sample_gen.gen_graph import visualize_graph



def get_true_edge(hit_sample):
    hit_sample = hit_sample.copy()
    edge_list = []
    index = np.unique(hit_sample["particle_index"])
    # print(index)
    for i in index:
        if i == 0:
            continue
        particle_hit_df = hit_sample[hit_sample["particle_index"] == i]
        for j in range(len(particle_hit_df)):
            for k in range(j + 1, len(particle_hit_df)):
                edge_list.append([particle_hit_df.iloc[j]["hit_id"], particle_hit_df.iloc[k]["hit_id"]])

    edges = torch.tensor(np.array(edge_list, dtype=np.int64)).T
    return edges

if __name__ == "__main__":
    from temp.Sample import Sample
    sample = Sample(1000)
    sample.generate_sample(1)
    sam = sample.get_sample(0)
    print(sam)
    true_edge = get_true_edge(sam)
    print(true_edge)




