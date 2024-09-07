import uproot
import numpy as np
import pandas as pd

filename = "mini15_allDets_hits_1000eV_noCorrect.root"



if __name__ == "__main__":
    with uproot.open(filename) as f:
        print(f.keys())
        tree = f["layer0"]
        print(tree)
        # 假设要绘制名为"some_variable"的变量的直方图
        print(tree.keys())
        x = "layer0_x"
        y = "layer0_y"
        z = "layer0_z"
        ids = "layer0_id"
        MCid  = "mcparticleID_L0"
        evtID = "eventID_L0"
        evtnum = tree[z].array()
        print(evtnum)
        df = pd.DataFrame()
        df["x"] = np.array(tree[x].array())
        df["y"] = np.array(tree[y].array())
        df["z"] = np.array(tree[z].array())
        df["id"] = np.array(tree[ids].array())
        df["MCid"] = np.array(tree[MCid].array())
        df["evtID"] = np.array(tree[evtID].array())
        df = df[df["evtID"] == 1]
        print(df)




        # datax = np.array(tree[x].array())
        # datay = np.array(tree[y].array())[:, 1]