
import numpy as np
import torch
from temp.Sample_gen.mode_default import gen_mode
import itertools

def get_all_edge(hit_sample):
    
    z_mode = gen_mode['z_range']

    
    
    
    hit_df_l0 = hit_sample[hit_sample['z'] == z_mode[0]]
    hit_df_l1 = hit_sample[hit_sample['z'] == z_mode[1]]
    hit_df_l2 = hit_sample[hit_sample['z'] == z_mode[2]]
    hit_df_l3 = hit_sample[hit_sample['z'] == z_mode[3]]


    pair01 = []
    pair02 = []
    pair03 = []
    pair12 = []
    pair13 = []
    pair23 = []

    pid0 = hit_df_l0['hit_id'].values
    pid1 = hit_df_l1['hit_id'].values
    pid2 = hit_df_l2['hit_id'].values
    pid3 = hit_df_l3['hit_id'].values

    pair01 = np.array(list(itertools.product(pid0, pid1)))
    pair02 = np.array(list(itertools.product(pid0, pid2)))
    pair03 = np.array(list(itertools.product(pid0, pid3)))
    pair12 = np.array(list(itertools.product(pid1, pid2)))
    pair13 = np.array(list(itertools.product(pid1, pid3)))
    pair23 = np.array(list(itertools.product(pid2, pid3)))

    pair01 = torch.tensor(pair01, dtype=torch.long).t().contiguous()
    pair02 = torch.tensor(pair02, dtype=torch.long).t().contiguous()
    pair03 = torch.tensor(pair03, dtype=torch.long).t().contiguous()
    pair12 = torch.tensor(pair12, dtype=torch.long).t().contiguous()
    pair13 = torch.tensor(pair13, dtype=torch.long).t().contiguous()
    pair23 = torch.tensor(pair23, dtype=torch.long).t().contiguous()

    return pair01, pair02, pair03, pair12, pair13, pair23


    
    
    
    


if __name__ == "__main__":
    from temp.Sample import Sample
    sample = Sample(1000)
    sample.generate_sample(1)
    sam = sample.get_sample(0)
    print(sam)
    true_edge = get_all_edge(sam)
    print(true_edge)
