import numpy as np
import torch
from trkr.TRecA.Hit2graph_fake import hit2graph_fake
from sklearn.metrics import roc_auc_score, roc_curve


class Test_model:
    def __init__(self, model, wight_path, Hitsample):
        self.model = model
        self.wight_path = wight_path
        self.Hitsample = Hitsample

        model.load_state_dict(torch.load(wight_path))

        pass

    def fake_hit2graph(self, frac_true=0.7, frac_fake=0):
        data = hit2graph_fake(self.Hitsample, frac_true=frac_true, frac_fake=frac_fake)
        return data

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model.encode(data.x, data.edge_index)
            pred = self.model.decode(out, data.edge_label_index)
            true = data.edge_label
            pred = torch.sigmoid(pred)
        return pred, true

    def draw_ROC(self, pred, true, ax):
        fpr, tpr, _ = roc_curve(true.cpu(), pred.cpu())
        auc = roc_auc_score(true.cpu(), pred.cpu())
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        pass

    def cal_AUC(self, pred, true):
        auc = roc_auc_score(true.cpu(), pred.cpu())
        return auc

    def cal_acc(self, pred, true, data, frac=0.5):
        pos_pred = (pred > frac).float() * true
        neg_pred = (pred < 1 - frac).float() * (1 - true)

        no_judge = (pred < frac).float() * (pred > 1 - frac).float()
        no_judge_true = true * no_judge
        no_judge_false = (1 - true) * no_judge

        # no_judge = no_judge.float().sum().item() / data.edge_label.size(0)

        correct_pos = pos_pred.cpu().sum().item()
        correct_neg = neg_pred.cpu().sum().item()
        if (data.edge_label.size(0) - no_judge.float().sum().item()) == 0:
            return np.nan
        else:

            po_acc = correct_pos / (true.sum().item() - no_judge_true.sum().item() + 1e-10)
            ne_acc = correct_neg / ((1 - true).sum().item() - no_judge_false.sum().item() + 1e-10)

            accuracy = (correct_pos + correct_neg) / (data.edge_label.size(0) - no_judge.float().sum().item())
            interpretable = 1 - no_judge.float().sum().item() / data.edge_label.size(0)

            frac_t = 1 - no_judge_true.sum().item() / true.sum().item()
            frac_f = 1 - no_judge_false.sum().item() / (1 - true).sum().item()

            return accuracy, po_acc, ne_acc, interpretable, frac_t, frac_f
