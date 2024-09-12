import numpy as np
from torch_geometric.transforms import RandomLinkSplit
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train:
    def __init__(self, model, graphs):
        self.model = model.to(device)
        self.graphs = graphs
        pass

    def __len__(self):
        return len(self.graphs)

    def __str__(self):
        return f"BEGIN-----Trainning Info -----------\n" \
               f"\tTraining object with {len(self.graphs)} graphs \n\n \t Model: {self.model}\n" \
               f"END-------------------------------\n"

    def warning(self, message):
        print(f"\033[93m{message}\033[0m")

    def Log(self, text):
        print(f"\033[94m{text}\033[0m")

    def split_dataset(self):
        graph = self.graphs[0]  # start with only one graph after we would like dataloader
        transform = RandomLinkSplit(is_undirected=False, add_negative_train_samples=True)
        train_data, val_data, test_data = transform(graph)
        return train_data, val_data, test_data

    def split_dataset2(self):
        
        pass

    def initial_model(self):
        train_data, val_data, test_data = self.split_dataset()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.test_data = test_data.to(device)
        self.optimizer = optimizer

        pass

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            self.model.decode(z, self.train_data.edge_label_index),
            self.train_data.edge_label.to(torch.float32)
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs=100, visualize=False, path=None):
        self.initial_model()
        if visualize:
            fig = plt.figure(figsize=(10, 8))
            plt.ion()
            # plt.title("Training Loss")
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            plt.pause(1)

        for epoch in tqdm(range(0, epochs), desc="Training Epochs"):
            loss = self.train_one_epoch()
            if epoch % np.floor(epochs / 100) == 0:
                pred, true = self.test()
                if visualize:
                    ax2.clear()
                    self.draw_ROC(pred, true, ax2)
                    ax2.set_title("ROC Curve")

                    ax1.plot(epoch, loss, 'ro')
                    ax1.semilogy()
                    ax1.set_title("Training Loss")

                    ax3.plot(epoch, self.cal_AUC(pred, true), 'ro')
                    ax3.set_title("AUC")

                    ax4.plot(epoch, self.cal_acc(pred, true, self.test_data, 0.5), 'ko', label="0.5")
                    ax4.plot(epoch, self.cal_acc(pred, true, self.test_data, 0.7), 'ro', label="0.7")
                    ax4.plot(epoch, self.cal_acc(pred, true, self.test_data, 0.8), 'bo', label="0.8")
                    ax4.plot(epoch, self.cal_acc(pred, true, self.test_data, 0.9), 'go', label="0.9")
                    # only label once
                    if epoch == 0:
                        ax4.legend()
                    ax4.set_title("Accuracy")
                    plt.pause(0.001)
                if path:
                    filename = path + f"\\epoch_{epoch}.pth"
                    self.save_model(filename)

        plt.savefig("training.png", dpi=600)

    def test(self):
        data = self.test_data
        self.model.eval()
        with torch.no_grad():
            # where the edge index come ? i dont get it :(
            z = self.model.encode(data.x, data.edge_index)

            true = data.edge_label
            pred = self.model.decode(z, data.edge_label_index)
            pred = torch.sigmoid(pred)

        return pred, true

    def evaluate(self, smaples):
        pass

    def save_model(self, path):
        # torch.save(self.model, path)
        torch.save(self.model.state_dict(), path)
        self.Log(f"Model saved at {path}")

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
        # no_judge = no_judge.float().sum().item() / data.edge_label.size(0)

        correct_pos = pos_pred.cpu().sum().item()
        correct_neg = neg_pred.cpu().sum().item()
        if (data.edge_label.size(0) - no_judge.float().sum().item()) == 0:
            return np.nan
        else:
            accuracy = (correct_pos + correct_neg) / (data.edge_label.size(0) - no_judge.float().sum().item())
            return accuracy
