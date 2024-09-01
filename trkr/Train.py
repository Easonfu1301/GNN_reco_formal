import numpy as np
from torch_geometric.transforms import RandomLinkSplit
import torch
import torch.nn.functional as F
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
        graph = self.graphs[0] # start with only one graph after we would like dataloader
        transform = RandomLinkSplit(is_undirected=False, add_negative_train_samples=True)
        train_data, val_data, test_data = transform(graph)
        return train_data, val_data, test_data


    def initial_model(self):
        train_data, val_data, test_data = self.split_dataset()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.test_data = test_data.to(device)
        self.optimizer = optimizer


        pass





    def train_one_step(self):
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


    def train(self, epochs=10, visualize=False):
        self.initial_model()
        plt.ion()
        plt.figure()
        plt.title("Training Loss")
        name = ''
        for epoch in tqdm(range(0, epochs), desc="Training Epochs"):
            loss = self.train_one_step()
            if epoch % 10 == 0 and visualize:
                plt.plot(epoch, loss, 'ro')
                plt.semilogy()
                plt.pause(0.1)
            # name = f'Epoch {epoch}, Loss: {loss:.4f}'

        plt.ioff()


    def save_model(self, path):
        torch.save(self.model, path)
        self.Log(f"Model saved at {path}")


    def cal_AUC(self, data):
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(data.x, data.edge_index)
            pos_out = self.model.decode(z, data.edge_label_index)
            neg_out = self.model.decode(z, data.edge_label_index)
        pos_pred = (pos_out > 0).float()
        neg_pred = (neg_out < 0).float()
        return pos_pred, neg_pred
