import numpy as np

from collections import Counter






def restore_acc(hit_df, restore_df):
    restore_df = restore_df.copy()
    hit_df = hit_df.copy()

    score = np.zeros(len(restore_df))

    for i in range(len(restore_df)):
        hit_id1 = restore_df.at[i, "hit1"]
        hit_id2 = restore_df.at[i, "hit2"]
        hit_id3 = restore_df.at[i, "hit3"]
        hit_id4 = restore_df.at[i, "hit4"]
        hit_ids = [hit_id1, hit_id2, hit_id3, hit_id4]

        # print(hit_ids)


        hit_particle_index = np.array([hit_df[hit_df["hit_id"] == hit_ids[j]]["particle_index"].values[0] for j in range(4)])

        # print(hit_particle_index)
        counter = Counter(hit_particle_index)
        repeated_elements = [count for key, count in counter.items() if key != 0]
        print(counter)

        print(repeated_elements)
        max_repeated = np.max(repeated_elements) if len(repeated_elements) > 0 else 0


        score[i] = max_repeated

    restore_df["score"] = score
    print(restore_df)
    return restore_df


    #     restore_df.at[i, "hit_id"] = hit_df.at[restore_df.at[i, "hit_id"], "hit_id"]
    #     restore_df.at[i, "particle_index"] = hit_df.at[restore_df.at[i, "particle_index"], "particle_index"]

    # print(len(restore_df))
    # print(restore_df[0])