import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def draw_ROC(pred, true, ax):
    fpr, tpr, _ = roc_curve(true.cpu(), pred.cpu())
    auc = roc_auc_score(true.cpu(), pred.cpu())
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    pass


def cal_AUC(pred, true):
    auc = roc_auc_score(true.cpu(), pred.cpu())
    return auc


def cal_acc(pred, true, data, frac=0.5):
    pos_pred = (pred > frac).float() * true
    neg_pred = (pred < 1 - frac).float() * (1 - true)

    no_judge = (pred < frac).float() * (pred > 1 - frac).float()
    # no_judge = no_judge.float().sum().item() / data.edge_label.size(0)

    correct_pos = pos_pred.cpu().sum().item()
    correct_neg = neg_pred.cpu().sum().item()
    if (data.edge_label.size(0) - no_judge.float().sum().item()) == 0:
        return np.nan
    else:
        accuracy = (correct_pos + correct_neg) / (data.edge_label.size(0) - no_judge.float().sum().item())
        return accuracy