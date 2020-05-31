import argparse

configs = argparse.ArgumentParser()

# dataset
configs.add_argument('--data_path', type=str, default="F:\\dataset\\miniImagenet", help="dataset path root")
configs.add_argument('--dataset', type=str, default='miniImagenet', help="miniImagenet or tierImagenet")
# configs.add_argument('--data_path', type=str, default="F:\\dataset\\tiered-imagenet", help="dataset path root")
# configs.add_argument('--dataset', type=str, default='tierImagenet', help="miniImagenet or tierImagenet")
configs.add_argument('--num_workers', type=int, default=8, help="dataloader num_works")

# GPU

# FSL setting
configs.add_argument('--n_way', type=int, default='5', help="N-way")
configs.add_argument('--k_shot', type=int, default='2', help="K-shot")
configs.add_argument('--k_query', type=int, default='2', help="K-query")
configs.add_argument('--fetch_global', type=bool, default=False, help="fetch global label or one hot label")

configs.add_argument('--epochs', type=int, default='400', help="epoch")


# network setting
configs.add_argument('--x_dim', type=str, default="84,84,3", metavar='XDIM',
                     help='input image dims')
configs.add_argument('--h_dim', type=int, default=64, metavar='HDIM',
                     help="dimensionality of hidden layers (default: 64)")
configs.add_argument('--z_dim', type=int, default=64, metavar='ZDIM',
                     help="dimensionality of output channels (default: 64)")

# optimization params
configs.add_argument('--train_optim', type=str, default="adam",
                     help="optimizer: adam,sgd,rmsprop")
configs.add_argument('--lr', type=float, default=0.001, metavar='LR',
                     help="base learning rate")
configs.add_argument('--weight_decay', type=float, default=0.0005,
                     help="weight_decay")
configs.add_argument('--momentum', type=float, default=0.9,
                     help="momentum")

# learning rate decay policy
configs.add_argument('--lr_policy', type=str, default="plateau",
                     help="lr decay policy: multi_step, exponentialLR, Plateau")
configs.add_argument('--milestones', type=str, default="100,200,300",
                     help="milestone learning rate decay")
configs.add_argument('--step_size', type=int, default=10000, metavar='STEPSIZE',
                     help="lr decay step size")
configs.add_argument('--lr_gama', type=float, default=0.5, metavar='GAMMA',
                     help="decay rate")
configs.add_argument('--patience', type=int, default=200, metavar='PATIENCE',
                     help="train patience until stop")

# label propagation params
configs.add_argument('--k', type=int, default=20, metavar='K',
                     help="top k in constructing the graph W")
configs.add_argument('--sigma', type=float, default=0.25, metavar='SIGMA',
                     help="Initial sigma in label propagation")
configs.add_argument('--alpha', type=float, default=0.99, metavar='ALPHA',
                     help="Initial alpha in label propagation")
configs.add_argument('--rn', type=int, default=300, metavar='RN',
                     help="graph construction types: "
                          "300: sigma is learned, alpha is fixed" +
                          "30:  both sigma and alpha learned")

# save and restore params
configs.add_argument('--seed', type=int, default=1000, metavar='SEED',
                     help="random seed for code and data sample")
configs.add_argument('--iters', type=int, default=47, metavar='ITERS',
                     help="iteration to restore params")
configs.add_argument('--save_path', type=str, default="checkpoints")

configs = configs.parse_args()
