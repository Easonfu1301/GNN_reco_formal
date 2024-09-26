import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from temp.TRecA.edge2Track import edge2Track
from temp.Sample_gen.sample_edges_from_true_edge import sample_edges_from_true_edge

from temp.Model.GCN import model

from temp.Sample_gen.find_edge_truth import get_true_edge
from temp.Sample_gen.find_edge_geo import get_geo_edge
from temp.Sample_gen.find_all_edge import get_all_edge
from temp.Sample_gen.transform_coord import get_node_tensor
from sklearn.metrics import roc_auc_score, roc_curve
import time
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd


class RestoreTrk:
    def __init__(self, weight_dict_path):
        self.weight_dict_path = weight_dict_path
        self.model = model
        self.model.load_state_dict(torch.load(weight_dict_path))
        pass

    # def get_graph_edges(self, hit_sample):  # based on the geometry
    #     pass
    #
    # def get_graph_nodes(self, hit_sample):
    #     pass

    def get_judge_edges(self, hit_sample):
        geo_edge, geo_edge_label = get_geo_edge(hit_sample, 0.05)
        true_edge = get_true_edge(hit_sample)
        # true_edge_label = torch.ones(true_edge.shape[1])
        train_edges, _, train_edges_label, _ = sample_edges_from_true_edge(true_edge, 1, maxindex=len(hit_sample))
        print("detect number", geo_edge_label.shape)
        # return geo_edge, geo_edge_label
        return train_edges, train_edges_label

    def get_all_edges(self, hit_sample):
        hit01, hit02, hit03, hit12, hit13, hit23 = get_all_edge(hit_sample)
        # hits = np.hstack((hit01, hit02, hit03, hit12, hit13, hit23))
        return [hit01.to(device), hit12.to(device), hit23.to(device)]

    # def construct_full_graph(self, hit_sample):
    #
    #     return graph

    def judge_all_edge(self, hit_sample):
        nodes = get_node_tensor(hit_sample)
        true_edges = get_true_edge(hit_sample)
        edges, _ = get_geo_edge(hit_sample, 0.01)
        t = time.time()
        judge_edges_index, judge_edges_label = self.get_judge_edges(hit_sample)
        print("time preprocess", time.time() - t)
        graph = Data(x=nodes.to(device), edge_index=edges.to(device), edge_label_index=judge_edges_index.to(device),
                     edge_label=judge_edges_label.to(device))

        self.model.eval()
        with torch.no_grad():
            t = time.time()

            z = self.model.encode(graph.x, graph.edge_index)
            judge_edges = self.model.decode(z, graph.edge_label_index)
            print("time decode", time.time() - t)
            print(judge_edges)

            judge_edges = torch.sigmoid(judge_edges)

        true = graph.edge_label
        self.draw_AUC(judge_edges, true)

        return judge_edges, true

    def eval_all_edge(self, hit_sample):
        nodes = get_node_tensor(hit_sample)
        edges, _ = get_geo_edge(hit_sample, 0.01)
        t = time.time()
        all_edges = self.get_all_edges(hit_sample)
        print(all_edges)
        print("time preprocess", time.time() - t)
        graph = Data(x=nodes.to(device), edge_index=edges.to(device))
        layer_link = []

        self.model.eval()

        preds = []

        for edge in all_edges:
            with torch.no_grad():
                t = time.time()
                print()
                z = self.model.encode(graph.x, graph.edge_index)
                judge_edges = self.model.decode(z, edge)
                judge_edges = torch.sigmoid(judge_edges)
                preds.append(judge_edges)
                # print(judge_edges.shape)
                #
                # print("time decode", time.time() - t)

        #     judge_edges = torch.sigmoid(judge_edges)
        #
        # true = graph.edge_label
        # self.draw_AUC(judge_edges, true)
        # print(judge_edges)
        return preds

    def find_best_track_seed(self, hit_sample):
        edges = self.get_all_edges(hit_sample)
        preds = self.eval_all_edge(hit_sample)

        # coffes = []
        # print(edge)
        # print(np.unique(edge[0, :]).shape, np.unique(edge[1, :]).shape)

        matches = []

        for i in range(len(preds)):

            edge = edges[i].cpu().numpy()


            hit_st = np.unique(edge[0, :])
            hit_ed = np.unique(edge[1, :])

            len_st = hit_st.shape[0]
            len_ed = hit_ed.shape[0]

            hit_lut_st = np.zeros((2, len_st))
            hit_lut_st[0, :] = hit_st
            hit_lut_st[1, :] = np.array(range(0, len_st ))

            hit_lut_ed = np.zeros((2, len_ed))
            hit_lut_ed[0, :] = hit_ed
            hit_lut_ed[1, :] = np.array(range(0, len_ed))


            # replace the hit index with the new index
            # 用向量化替换循环，以提高效率
            edge[0, :] = np.where(np.isin(edge[0, :], hit_lut_st[0, :]),
                                  np.take(hit_lut_st[1, :], np.searchsorted(hit_lut_st[0, :], edge[0, :])),
                                  edge[0, :])

            edge[1, :] = np.where(np.isin(edge[1, :], hit_lut_ed[0, :]),
                                  np.take(hit_lut_ed[1, :], np.searchsorted(hit_lut_ed[0, :], edge[1, :])),
                                  edge[1, :])

            coff = np.zeros((len_st, len_ed))
            coff[edge[0, :], edge[1, :]] = preds[i].cpu().numpy()

            best_match = self.select_best_match(coff)

            # replace with the original hit index
            for j in range(best_match.shape[1]):
                best_match[0, j] = hit_lut_st[0, best_match[0, j] == hit_lut_st[1, :]]
                best_match[1, j] = hit_lut_ed[0, best_match[1, j] == hit_lut_ed[1, :]]




            print(best_match)
            matches.append(best_match)

        match01, match12, match23 = matches
        # matching_line = np.isin(match01[1, :], match12[0, :])
        # print(matching_line)


        df01 = pd.DataFrame(match01.T, columns=['hit1', 'hit2'])
        df12 = pd.DataFrame(match12.T, columns=['hit2', 'hit3'])
        df23 = pd.DataFrame(match23.T, columns=['hit3', 'hit4'])

        df012 = pd.merge(df01, df12, on='hit2', how='inner')
        df0123 = pd.merge(df012, df23, on='hit3', how='inner')
        print(df0123)

        return df0123



            # time.sleep(100000)


        # coff = np.zeros((np.unique(edge[0, :]).shape[0] + 1, np.unique(edge[1, :]).shape[0] + 1))
        # print(layer_links)
        # coff[edge[0, :], edge[1, :]] = layer_links.cpu().numpy()
        # # coff[edge[1, :], edge[0, :]] = layer_links.cpu().numpy()
        # self.select_best_match(coff, np.unique(edge[0, :]), np.unique(edge[1, :]))

    def select_best_match(self, coffes):

        row_ind, col_ind = linear_sum_assignment(-coffes)

        print(row_ind.shape, col_ind.shape)

        best_match = np.zeros((row_ind.shape[0], 2))

        best_match[:, 0] = row_ind
        best_match[:, 1] = col_ind



        # # 输出最佳配对及其概率
        # for i, j in zip(row_ind, col_ind):
        #     print(f"点 {hit1[i]} 配对 点 {hit2[j]}，链接概率为 {coffes[i, j]}")

        return best_match.T.astype(int)

    def cal_accuracy(self, edge2track):
        pass

    def draw_AUC(self, pred, true):
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fpr, tpr, _ = roc_curve(true.cpu(), pred.cpu())
        auc = roc_auc_score(true.cpu(), pred.cpu())
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.show()
        pass

    def restore(self):
        best_track = self.find_best_track_seed(self.Hitsample)
        accuracy = self.cal_accuracy(best_track)
        return best_track, accuracy
