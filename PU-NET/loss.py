import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace as brk

from scipy.spatial import distance

import numpy as np


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
'''
class RepulsionLoss(nn.Module):

    def __init__(self):
        super(RepulsionLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        #self.h = h
        #self.eps = eps

    def forward(self, pred):
        repulsion_loss = []
        for i in range(pred.shape[0]):
            #print(pred[i,:,:].shape)
            D = distance.squareform(distance.pdist(pred[i,:,:]))
            #print(np.round(D, 1))
            closest = np.argsort(D, axis=1)
            k = 5  # For each point, find the 3 closest points

            knn_values = pred[i,closest[:, 1:k+1]]

            print(knn_values.shape)
            print(pred[i,:,:].shape)

            l2 = 0
            for j in range(knn_values.shape[1]):
                l2 = l2 + torch.sum(np.power((knn_values[:, j, :] - pred[i, :, :]), 2))
                
            dist2 = torch.max(l2, torch.tensor(self.eps).cuda())
            dist = torch.sqrt(dist2)
            weight = torch.exp(- dist2 / self.h ** 2)

            # uniform_loss = torch.mean((self.radius - dist) * weight)
            uniform_loss = torch.mean(- dist * weight) # punet

            repulsion_loss.append(uniform_loss)
            
        return torch.Tensor(repulsion_loss)'''


def test():
    torch.cuda.is_available()
    img_channels = 3
    img_size = 1024
    x = torch.randn((1, 2048, 3))
    y = 18 * x

    print(ChamferLoss())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0




if __name__ == "__main__":
    test()

