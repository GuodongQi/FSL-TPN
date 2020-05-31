import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import configs
from datasets.miniImagenet import MiniImagenet
from datasets.tierImagesnet import TierImageNet


def data_loader(cfg, num_workers=0, split='train'):
    if cfg.dataset == "miniImagenet":
        train_data = MiniImagenet(cfg.data_path, cfg.n_way, cfg.k_shot, cfg.k_query, cfg.x_dim, split)
    elif cfg.dataset == 'tierImagenet':
        train_data = TierImageNet(cfg.data_path, cfg.n_way, cfg.k_shot, cfg.k_query, cfg.x_dim, split)
    else:
        raise Exception("check your spelling of dataset")
    train_db = DataLoader(train_data, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_db


if __name__ == '__main__':
    print(configs.__repr__())
    db = data_loader(configs, "val")
    for i in tqdm(db):
        # support_y = i[1]
        # support_y = torch.zeros(configs.n_way * configs.k_shot, configs.n_way).scatter(1, support_y.view(-1, 1), 1)
        pass
