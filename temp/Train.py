import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
from temp.Model.visualize_result import *

from tqdm import tqdm

from temp.Model.GCN import model


class Train:
    def __init__(self, train_data, test_data, visualize=True):
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.model = model
        self.loss_fn = F.binary_cross_entropy_with_logits
        self.vis = visualize
        pass

    def warning(self, message):
        print(f"\033[93m{message}\033[0m")

    def Log(self, text):
        print(f"\033[94m{text}\033[0m")

    def initial_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer = optimizer


        if self.vis:
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

        pass

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.loss_fn(
            self.model.decode(z, self.train_data.edge_label_index),
            self.train_data.edge_label.to(torch.float32)
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train(self, epochs):




        self.initial_model()
        for epoch in tqdm(range(epochs)):
            loss = self.train_one_epoch()
            if epoch % 10 == 0:
                self.Log(f"Epoch {epoch} | Loss: {loss}")
                if self.vis:
                    self.visualize(epoch)
                    plt.pause(1e-5)
            # print(f"Epoch {epoch} | Loss: {loss}")
        pass


    def save_model(self, path):
        # torch.save(self.model, path)
        torch.save(self.model.state_dict(), path)
        self.Log(f"Model saved at {path}")


    def visualize(self, epoch):
        fig = plt.gcf()
        axs = fig.get_axes()

        model = self.model.eval()
        with torch.no_grad():
            z = model.encode(self.test_data.x, self.test_data.edge_index)

            pred = model.decode(z, self.test_data.edge_label_index)
            pred = torch.sigmoid(pred)

            true = self.test_data.edge_label.to(torch.float32)

            loss = self.loss_fn(
                self.model.decode(z, self.test_data.edge_label_index),
                self.test_data.edge_label.to(torch.float32)
            )


        axs[0].plot(epoch, loss.item(), 'ro')

        acc1 = cal_acc(pred, true, self.test_data, 0.5)
        acc2 = cal_acc(pred, true, self.test_data, 0.7)
        acc3 = cal_acc(pred, true, self.test_data, 0.8)
        acc4 = cal_acc(pred, true, self.test_data, 0.9)
        axs[1].plot(epoch, acc1, 'k.')
        axs[1].plot(epoch, acc2, 'r.')
        axs[1].plot(epoch, acc3, 'b.')
        axs[1].plot(epoch, acc4, 'g.')


        AUC = cal_AUC(pred, true)
        axs[2].plot(epoch, AUC, 'ro')

        axs[3].cla()
        draw_ROC(pred, true, axs[3])

        # axs[3].plot(epoch, loss.item(), 'ro')
        # print(pred)

