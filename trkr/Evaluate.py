import matplotlib.pyplot as plt
import numpy as np
from trkr.TRecA.Hit2graph_potential import cal_edge
from trkr.TRecA.Hit2graph_truth import hit2graph as hg_truth
import torch

default_gen_mode = {
    "z_range": [2500, 2600, 2700, 2800],
    "phi_range": [0, 2 * np.pi],
    "ctheta_range": [0.5, 1],
    "gaussian_noise": 0.05
}
class RestoreTrk:
    def __init__(self, model, wight_path, hitsample):
        self.model = model
        self.wight_path = wight_path




        self.hitsample = hitsample
        pass

    def potential_edge(self):
        edge = cal_edge(self.hitsample.copy(), default_gen_mode)
        return edge

    def predict_link(self):
        # load pth file
        self.model.load_state_dict(torch.load(self.wight_path))



        self.model.eval()
        edge = self.potential_edge()
        edge_index = torch.tensor(edge, dtype=torch.long)
        x = hg_truth(self.hitsample.copy()).x
        # x = torch.tensor(self.hitsample.iloc[:, 1:-1].values, dtype=torch.float)
        # hi
        print(x)

        with torch.no_grad():
            z = self.model.encode(x, edge_index)
            pred = self.model.decode(z, edge_index)
            pred = pred[np.abs(pred) < 100]
            print(pred)
            plt.ioff()
            plt.figure()
            plt.hist(pred.numpy(), bins=10000)
            plt.show()
            pred = torch.sigmoid(pred)
            print(np.sum(pred.numpy() < 0.99) / len(pred))
        return edge, pred


        pass

    def restore_track(self):
        pre_edge, pred = self.predict_link()
        print(pre_edge)
        print(pred)

        print(pre_edge[:, pred.numpy() < 0.2])
        pass

    def cal_accuracy(self):
        pass
