import copy # Import Python's in-built copy function which can be used to create
            # deep copy (completely independent copy) or shallow-copy
            # (copies outer object, maintains reference for inner object)

import torch
from torchvision import datasets, transforms # Loads standard datasets like MNIST, CIFAR10, etc.
                                             # And loads the module transforms used for pre-processing image data

from sampling import mnist_iid, mnist_non_iid, mnist_noniid_unequal, custom_skewed_partition


def get_dataset(args):
    '''
    Returns train and test datasets and a user group which is a dict
    where the keys are the user index and the values are the corresponding
    data for each of those users.
    :param args:
    :return: train and test datsets
    '''

    if args.dataset == 'mnist':
        data_dir = '../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir,train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir,train=False,download=True,
                                      transform=apply_transform)

        # Define task-specific label mapping only if MultiTaskCNN is selected
        label_mappings = {}

        # if args.model == 'MultiTaskCNN':
        #     label_mappings = {
        #         0: lambda y: 0 if y == 2 else 1,  # Client 0: [2, not-2]
        #         1: lambda y: 0 if y == 1 else 1 if y == 2 else 2 if y == 3 else 3,  # Client 1: [1, 2, 3, none]
        #         2: lambda y: 0 if y == 6 else 1  # Client 2: [6, not-6]
        #     }

        if args.model == 'MultiTaskCNN':
            def map_client_0(y):  # 2 vs not-2
                return 0 if y == 2 else 1

            def map_client_1(y):  # 1, 2, 3, none
                if y == 1:
                    return 0
                elif y == 2:
                    return 1
                elif y == 3:
                    return 2
                else:
                    return 3

            def map_client_2(y):  # 6 vs not-6
                return 0 if y == 6 else 1

            label_mappings = {
                0: map_client_0,
                1: map_client_1,
                2: map_client_2
            }

            # 🔍 Debug: check that all labels map correctly for each client
            for client_id, map_func in label_mappings.items():
                for y in range(10):
                    print(f"Client {client_id}, Label {y} -> Mapped: {map_func(y)}")

        # Mode of data sampling amongst users. This part controls
        # the data heterogeneity

        if args.partition_type=='iid':
            # Sample iid user data from MNIST
            user_groups = mnist_iid(train_dataset,args.num_users)
            test_user_groups = mnist_iid(test_dataset, args.num_users)

        elif args.partition_type=='noniid':
            # Sample non-IID user data from MNIST
            if args.unequal:
                # Choose unequal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset,args.num_users)
                test_user_groups = mnist_noniid_unequal(test_dataset, args.num_users)

            else:
                user_groups = mnist_non_iid(train_dataset,args.num_users)
                test_user_groups = mnist_non_iid(test_dataset, args.num_users)

        elif args.partition_type=='custom_skewed':
            try:
                user_groups = custom_skewed_partition(train_dataset, args.num_users, seed=args.seed) # Training data partition
                test_user_groups = custom_skewed_partition(test_dataset, args.num_users, seed = args.seed + 1 ) # Testing data partition
            except Exception as e:
                print("[Partition Error]", repr(e))
                raise


        else:
            raise ValueError("Invalid partition_type. Choose from: 'iid', 'noniid', 'custom_skewed'")

    else:
        raise ValueError('Invalid dataset')

    return train_dataset, test_dataset, user_groups, test_user_groups, label_mappings

# The function below is FedAvg equivalent
def average_weights(w):
    '''
    Return the average of the weights.
    :return: average_weights
    '''

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[key] += w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

def exp_details(args):
    print('\nExperimental Details:')
    print(f'    Model   : {args.model}')
    print(f'    Optimizer   : {args.optimizer}')
    print(f'    Learning    : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('     Federated parameters:')
    # if args.iid:
    #     print('     IID')
    #
    # else:
    #     print('     Non-IID')

    print(f'     Partition Type     : {args.partition_type}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch Size    : {args.local_bs}')
    print(f'    Local Epochs    : {args.local_ep}\n')

    return

