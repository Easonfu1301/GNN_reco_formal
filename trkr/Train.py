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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)

        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.test_data = test_data.to(device)
        self.optimizer = optimizer


        pass





    def train_one_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)

        # print(self.model.decode(z, self.train_data.edge_label_index).shape)
        # print(self.train_data.edge_label.to(torch.float32).shape)

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
            plt.ion()
            plt.figure()
            plt.title("Training Loss")
        for epoch in tqdm(range(0, epochs), desc="Training Epochs"):
            loss = self.train_one_step()
            if epoch % 100 == 0:
                self.test()
            if epoch % 10 == 0 and visualize:

                plt.plot(epoch, loss, 'ro')
                plt.semilogy()
                plt.pause(0.1)
            if path:
                self.save_model(path)


        plt.ioff()

    def test(self):
        data = self.test_data
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(data.x, data.edge_index)
            true = data.edge_label
            pred = self.model.decode(z, data.edge_label_index)
            pred = torch.sigmoid(pred)



            pos_pred = (pred > 0.7).float() * true
            neg_pred = (pred < 0.3).float() * (1-true)

            no_judge = (pred < 0.7).float() * (pred > 0.3).float()
            no_judge = no_judge.float().sum().item() / data.edge_label.size(0)

            correct_pos = pos_pred.cpu().sum().item()
            correct_neg = neg_pred.cpu().sum().item()

            accuracy = (correct_pos + correct_neg) / (data.edge_label.size(0))

        self.Log(f"Accuracy: {accuracy}, No Judge: {no_judge}")

        return accuracy

    def save_model(self, path):
        torch.save(self.model, path)
        self.Log(f"Model saved at {path}")

    def draw_ROC(self, data):
        pass

    def cal_AUC(self, data):
        pass
