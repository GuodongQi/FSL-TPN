import os
import time

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import configs
from core.model import LabelPropagation
from datasets.data_loader import data_loader

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    configs.epochs = 600
    configs.iters = 22

    configs.n_way = 5
    configs.k_shot = 1
    configs.k_query = 20
    configs.rn = 30
    configs.num_workers = 8
    print(repr(configs))

    pwd = os.getcwd()
    save_path = os.path.join(pwd, configs.save_path,
                             "%d_way_%d_shot_%d_rn" % (configs.n_way, configs.k_shot, configs.rn))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model_path = os.path.join(save_path, "model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # init dataloader
    print("init data loader")
    train_db = data_loader(configs, split="train")
    val_db = data_loader(configs, split="val")

    # init neural networks
    net = LabelPropagation(configs).to(device)

    if configs.iters:
        net.load_state_dict(torch.load(os.path.join(model_path, "%d_model.pkl" % configs.iters)))
        print('Loading Parameters from %d_model.pkl' % configs.iters)

    # optimizer
    if configs.train_optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    elif configs.train_optim == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=configs.lr, weight_decay=configs.weight_decay, momentum=configs.momentum)
    elif configs.train_optim == 'rmsprop':
        optimizer = optim.RMSprop(
            net.parameters(), lr=configs.lr, weight_decay=configs.weight_decay, momentum=configs.momentum,
            alpha=0.9, centered=True)
    else:
        raise Exception("error optimizer")

    # learning rate decay policy
    if configs.lr_policy == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=list(map(int, configs.milestones.split(','))),
                                gamma=configs.lr_gama)
    elif configs.lr_policy == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=configs.lr_gamma)
    elif configs.lr_policy == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=configs.lr_gama, patience=10, verbose=True)
    else:
        raise Exception('error lr decay policy')

    # Train and validation
    best_acc = 0.0
    best_loss = np.inf
    wait = 0

    writer = SummaryWriter(os.path.join(save_path, "logs", "%s" % time.strftime('%Y-%m-%d-%H-%M')))
    for ep in range(configs.iters, configs.epochs):
        loss_tr = []
        ce_list = []

        acc_tr = []
        loss_val = []
        acc_val = []
        train_loss_item = 0
        train_acc_item = 0

        train_pbar = tqdm(train_db)
        for step, train_data in enumerate(train_pbar):
            train_pbar.set_description(
                'train_epoc:{}, loss:{:.4f}, acc:{:.4f}'.format(ep, train_loss_item, train_acc_item))
            # start to train
            net.train()

            support_x, support_y, query_x, query_y = train_data
            # convert label to one-hot
            support_y = torch.zeros(configs.n_way * configs.k_shot, configs.n_way).scatter(1, support_y.view(-1, 1), 1)
            query_y = torch.zeros(configs.n_way * configs.k_query, configs.n_way).scatter(1, query_y.view(-1, 1), 1)

            inputs = [support_x[0].to(device), support_y.to(device), query_x[0].to(device), query_y.to(device)]
            loss, acc = net(inputs)
            train_loss_item = loss.item()
            train_acc_item = acc.item()
            loss_tr.append(train_loss_item)
            acc_tr.append(train_acc_item)

            net.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 4.0)
            optimizer.step()

        # for valid
        valid_loss_item = 0
        valid_acc_item = 0
        valid_pbar = tqdm(val_db)
        for step, train_data in enumerate(valid_pbar):
            valid_pbar.set_description(
                'valid_epoc:{}, loss:{:.4f}, acc:{:.4f}'.format(ep, valid_loss_item, valid_acc_item))
            # start to valid
            net.eval()

            support_x, support_y, query_x, query_y = train_data
            # convert label to one-hot
            support_y = torch.zeros(configs.n_way * configs.k_shot, configs.n_way).scatter(1, support_y.view(-1, 1), 1)
            query_y = torch.zeros(configs.n_way * configs.k_query, configs.n_way).scatter(1, query_y.view(-1, 1), 1)
            inputs = [support_x[0].to(device), support_y.to(device), query_x[0].to(device), query_y.to(device)]
            with torch.no_grad():
                loss, acc = net(inputs)
            valid_loss_item = loss.item()
            valid_acc_item = acc.item()
            loss_val.append(valid_loss_item)
            acc_val.append(valid_acc_item)

        scheduler.step(np.mean(loss_val))
        print('epoch:{}, loss_tr:{:.5f}, acc_tr:{:.5f}, loss_val:{:.5f}, acc_val:{:.5f}, lr:{:.6f}'
              .format(ep, np.mean(loss_tr), np.mean(acc_tr), np.mean(loss_val), np.mean(acc_val),
                      optimizer.param_groups[0]['lr']))

        # tensorboard
        # writer.add_graph(net, (inputs,))
        writer.add_scalar('Loss/train', np.mean(loss_tr), ep)
        writer.add_scalar('Loss/val', np.mean(loss_val), ep)
        writer.add_scalar('Accuracy/train', np.mean(acc_tr), ep)
        writer.add_scalar('Accuracy/val', np.mean(acc_val), ep)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)

        # Model Save and Stop Criterion
        cond1 = (np.mean(acc_val) > best_acc)
        cond2 = (np.mean(loss_val) < best_loss)

        if cond1 or cond2:
            best_acc = np.mean(acc_val)
            best_loss = np.mean(loss_val)
            print('best val loss:{:.5f}, acc:{:.5f}'.format(best_loss, best_acc))

            # save model

            torch.save(net.state_dict(), os.path.join(save_path, "model", '%d_model.pkl' % (ep + 1)))
            wait = 0

        else:
            wait += 1
            if ep % 100 == 0:
                torch.save(net.state_dict(), os.path.join(save_path, "model", '%d_model.pkl' % (ep + 1)))

        if wait > configs.patience:
            break


if __name__ == '__main__':
    main()
