import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""

    def __init__(self, cfg):
        super(CNNEncoder, self).__init__()
        self.cfg = cfg
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        """x: bs*3*84*84, out"""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out  # out: bs*3*


class RelationNetwork(nn.Module):
    """Graph construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x):
        x = x.view(-1, 64, 5, 5)
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs *1
        return out


class Prototypical(nn.Module):
    """Main Module for prototypical networks"""

    def __init__(self, cfg):
        super(Prototypical, self).__init__()
        self.cfg = cfg
        self.im_width, self.im_height, self.channels = list(map(int, self.cfg.x_dim.split(',')))
        self.h_dim, self.z_dim = self.cfg.h_dim, self.cfg.z_dim

        self.encoder = CNNEncoder(cfg)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        support, s_labels, query, q_labels = inputs
        num_classes = self.cfg.n_way
        num_support = self.cfg.n_way * self.cfg.k_shot
        num_query = self.cfg.n_way * self.cfg.k_query

        inp = torch.cat((support, query), 0)  # (n*k1+n*k2, 3, w, h)
        emb = self.encoder(inp)  # (n*k1+n*k2, 64, 5, 5)
        emb_s, emb_q = torch.split(emb, [num_support, num_query],
                                   0)  # (n*k1, 64, 5, 5) (n*k2, 64, 5, 5)
        emb_s = emb_s.view(num_classes, self.cfg.k_shot, -1).mean(1)  # (n, 64*5*5)
        emb_q = emb_q.view(num_query, -1)  # (n*k2, 64*5*5)
        emb_s = torch.unsqueeze(emb_s, 0)  # 1xnxD
        emb_q = torch.unsqueeze(emb_q, 1)  # n*k2x1xD
        dist = ((emb_q - emb_s) ** 2).mean(2)  # (n*k2)xnxD -> (n*k2)xn

        ce = nn.CrossEntropyLoss().cuda(0)
        loss = ce(-dist, torch.argmax(q_labels, 1))

        # acc
        pred = torch.argmax(-dist, 1)
        gt = torch.argmax(q_labels, 1)
        correct = (pred == gt).sum()
        acc = 1.0 * correct.float() / float(num_query)

        return loss, acc


class LabelPropagation(nn.Module):
    """label propagation"""

    def __init__(self, cfg):
        super(LabelPropagation, self).__init__()
        self.cfg = cfg
        self.im_width, self.im_height, self.channels = list(map(int, self.cfg.x_dim.split(',')))
        self.h_dim, self.z_dim = self.cfg.h_dim, self.cfg.z_dim

        self.encoder = CNNEncoder(cfg)
        self.relation = RelationNetwork()

        if self.cfg.rn == 300:  # learned sigma, fixed alpha
            self.alpha = torch.tensor([self.cfg.alpha], requires_grad=False).cuda(0)
        else:
            self.alpha = nn.Parameter(torch.tensor([self.cfg.alpha]).cuda(0), requires_grad=True)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        eps = np.finfo(float).eps  # minimal positive number
        support, s_labels, query, q_labels = inputs
        num_classes = self.cfg.n_way
        num_support = self.cfg.n_way * self.cfg.k_shot
        num_query = self.cfg.n_way * self.cfg.k_query

        # Step1: Embedding
        inp = torch.cat((support, query), 0)
        emb_all = self.encoder(inp).view(num_support + num_query, -1)  # (n*k1+n*k2, 64*5*5)
        N, d = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        # sigma
        if self.cfg.rn in [30, 300]:
            self.sigma = self.relation(emb_all)

            # W
            emb_all = emb_all / (self.sigma + eps)  # (n*k1+n*k2)*(64*5*5)
            emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.exp(-W / 2)

        # keep top-k values
        if self.cfg.k > 0:
            topk, indices = torch.topk(W, self.cfg.k)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W = W * mask

        ## normalize
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(num_query, num_classes).cuda(0)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)  # num_query+num_support, num_classes N * k
        F = torch.matmul(torch.inverse(torch.eye(N).cuda(0) - self.alpha * S + eps), y)  # N * k
        Fq = F[num_support:, :]  # query predictions

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = num_query
        acc = 1.0 * correct.float() / float(total)

        return loss, acc
