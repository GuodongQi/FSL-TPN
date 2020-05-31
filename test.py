import os

import numpy as np
import torch
from tqdm import tqdm

from configs.config import configs
from core.model import LabelPropagation
from datasets.data_loader import data_loader

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    configs.iters = 0  # load which weights

    configs.n_way = 5
    configs.k_shot = 1
    configs.k_query = 20
    configs.rn = 30
    print(repr(configs))

    pwd = os.getcwd()
    save_path = os.path.join(pwd, configs.save_path,
                             "%d_way_%d_shot_%d_rn" % (configs.n_way, configs.k_shot, configs.rn))
    model_path = os.path.join(save_path, "model")
    if not os.path.exists(model_path):
        raise Exception("weight files are not found!")

    # init dataloader
    print("init data loader")
    test_db = data_loader(configs, split="test")

    # init neural networks
    net = LabelPropagation(configs).to(device)
    # test
    best_acc = 0
    for ep in tqdm(range(configs.iters, configs.epochs)):
        # load weights
        weights_file = os.path.join(model_path, "%d_model.pkl" % configs.iters)
        if not os.path.exists(weights_file):
            continue
        net.load_state_dict(torch.load(weights_file))
        # print('Loading Parameters from %d_model.pkl' % configs.iters)

        acc_test = []
        test_acc_item = 0

        test_pbar = tqdm(test_db)
        for step, test_data in enumerate(test_pbar):
            test_pbar.set_description('acc:{:.4f}'.format(test_acc_item))
            # start to train
            net.eval()

            support_x, support_y, query_x, query_y = test_data
            # convert label to one-hot
            support_y = torch.zeros(configs.n_way * configs.k_shot, configs.n_way).scatter(1, support_y.view(-1, 1), 1)
            query_y = torch.zeros(configs.n_way * configs.k_query, configs.n_way).scatter(1, query_y.view(-1, 1), 1)

            inputs = [support_x[0].to(device), support_y.to(device), query_x[0].to(device), query_y.to(device)]
            with torch.no_grad():
                loss, acc = net(inputs)

            test_acc_item = acc.item()
            acc_test.append(test_acc_item)

        cur_acc = np.mean(acc_test)
        print('weight file:{}, acc:{:.4f}'.format(weights_file, cur_acc))
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_weight = weights_file
            print('best weight file:{}, best acc:{:.4f}'.format(best_weight, best_acc))


if __name__ == '__main__':
    main()
