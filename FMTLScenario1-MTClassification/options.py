
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Federated arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of rounds of training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random Seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use cuDNN deterministic mode (slower, reproducible)')
    parser.add_argument('--num_users',type=int, default=3,
                        help = 'number of users: K')
    parser.add_argument('--frac',type=float,default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep',type=int,default=10,
                        help='the number of local epochs: E')
    parser.add_argument('--local_bs',type=int,default=100,
                        help='local batch size: B')
    parser.add_argument('--lr', type = float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # Model arguments (Updated for auto-encoder)
    parser.add_argument('--model', type=str, default='MultiTaskCNN', choices=['CNN','AutoEncoder','MultiTaskCNN'],
                        help='Choose between CNN, AutoEncoder or MultiTaskCNN')
    parser.add_argument('--kernel_num',type=int,default=9,
                        help='Number of each kind of kernel')
    parser.add_argument('--kernel_sizes',type=str,default='3,4,5',
                        help='Comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels',type=int, default=1,
                        help='Number of channels of images')
    parser.add_argument('--norm',type=str, default='batch_norm',
                        help='batch_norm, layer_norm or None')
    parser.add_argument('--num_filters', type=int, default=32,
                        help="Number of filters for conv nets --32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                             strided convolutions")
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent Space dimension for AutoEncoder')

    # Other Arguments
    parser.add_argument('--dataset',type=str, default='mnist',
                        help='Name of dataset')
    parser.add_argument('--num_classes',type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--gpu',default=None, help=" To use cuda,set \
                                                    to specific GPU Id. Default set to use CPU ")
    parser.add_argument('--optimizer',type=str, default='sgd', choices= ['sgd','Adam'],
                        help='Type of Optimizer')
    # parser.add_argument('--iid',type=int,default=1,
    #                     help='Default set to IID. Set to 0 for non-IID')
    parser.add_argument('--unequal',type=int,default=0,
                        help="Default set to 0 for equal data-splits. \
                             Set to 1 for unequal data-splits")
    parser.add_argument('--partition_type', type=str, default='custom_skewed',
                        choices=['iid', 'noniid', 'custom_skewed'],
                        help="Data partitioning method: 'iid', 'noniid', or 'custom_skewed'")

    parser.add_argument('--stopping_rounds',type=int, default=10,
                        help='Rounds of early stopping')
    parser.add_argument('--verbose',type=str,default=1,
                        help='Verbose')
    # parser.add_argument('--seed',type=int,default=1,
    #                     help='Random Seed')

    args = parser.parse_args()

    return args


