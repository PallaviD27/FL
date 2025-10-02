

# Note - This is still to be checked if this includes everything for auto-encoder
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import test_inference
from update import LocalUpdate
from model import CNNMNIST,AutoEncoder
from utils import get_dataset, average_weights, exp_details
from model import SharedEncoder, TaskHead
from update import test_inference_multitask
from collections import defaultdict

import os
os.makedirs('../save', exist_ok=True)

if __name__ == '__main__':
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('...')

    args = args_parser()
    import random, numpy as np, torch


    def set_seed(seed: int, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Full determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    set_seed(args.seed, args.deterministic)

    if args.partition_type != 'iid':
        args.iid = 0
    else:
        args.iid = 1

    exp_details(args)


    if args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda'if torch.cuda.is_available() and args.gpu is not None else 'cpu'


    # Local dataset and user groups

    train_dataset, test_dataset, user_groups, test_user_groups, label_mappings = get_dataset(args)

    logger = SummaryWriter('../logs')

    try:

        from sampling import print_label_distribution
        print_label_distribution(user_groups, train_dataset)

        # Build model
        # If convolutional neural network
        if args.model == 'CNN':
            if args.dataset =='mnist':
                global_model = CNNMNIST(args= args).to(device)
            else:
                exit('Invalid dataset selected')

        elif args.model == 'AutoEncoder':
            if args.dataset == 'mnist':
                # global_model = AutoEncoder(args=args).to(device) #Commented coz not using auto-encoder arguments
                global_model = AutoEncoder().to(device)

        elif args.model == 'MultiTaskCNN':
            if args.dataset == 'mnist':
                # Shared encoder for all clients
                shared_encoder = SharedEncoder().to(device)

                # Define client-specific output sizes
                client_output_dims = {
                    0: 2,  # Client 0: [2, not-2]
                    1: 4,  # Client 1: [1, 2, 3, none]
                    2: 2  # Client 2: [6, not-6]
                }

                # Client-specific heads
                client_heads = {
                    cid: TaskHead(input_dim=320, output_dim=out_dim).to(device)
                    for cid, out_dim in client_output_dims.items()
                }

                for idx in client_heads:
                    print(f"Client {idx} - Head Output Dim: {client_heads[idx].fc[-1].out_features}")


            else:
                raise ValueError('Invalid dataset selected')

        else:
            raise ValueError('Invalid model type')

        # Set the model to train and set it to device
        # Set models to train
        if args.model != 'MultiTaskCNN':
            global_model.to(device)
            global_model.train()
            print(global_model)
        else:
            shared_encoder.train()
            for head in client_heads.values():
                head.train()


        print(f"\nTotal users: {args.num_users}")
        for uid in user_groups:
            print(f"User {uid} has {len(user_groups[uid])} samples")

        # Added to track loss per client
        client_losses = {i: [] for i in range(args.num_users)}
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0

        # Simulation of distributed users and multiple communication rounds

        # Initialization of histories and other dictionaries (outside the loop)
        # Per-client TRAIN histories (length will become == args.epochs)
        client_train_acc = defaultdict(list)  # values in %
        client_train_loss = defaultdict(list)

        # Per-client TEST histories (filled by test_inference_multitask each round)
        client_test_accuracies = defaultdict(list)  # %
        client_test_losses = defaultdict(list)

        # Overall histories (one scalar per round)
        overall_test_acc_history = []
        overall_test_loss_history = []

        # Server averages per round (optional overall training curves)
        server_avg_train_loss = []
        server_avg_train_acc = []

        # Also keep a simple per-round training loss list if you like
        train_loss = []  # one value per round (same as server_avg_train_loss)

        test_acc, test_loss = None, None

        for epoch in tqdm(range(args.epochs)):
            local_weights = []
            print(f'\n | Global Training Round: {epoch+1} |\n')

            if args.model == 'MultiTaskCNN':
                shared_encoder.train()
                for head in client_heads.values():
                    head.train()
            else:
                global_model.train()

            m = max(int(args.frac * args.num_users),1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)


            participated = set()

            # Pre-append accuracy and loss so every client has exactly args.epochs elements
            for cid in range(args.num_users):
                client_train_acc[cid].append(np.nan)
                client_train_loss[cid].append(np.nan)

            # Client-wise training (Local Training)
            for idx in idxs_users:
                print(f"Training Client {idx} with task head output size: {client_heads[idx].fc[-1].out_features}")

                local_encoder = copy.deepcopy(shared_encoder).to(device)

                local_model = LocalUpdate(
                    args=args,
                    dataset=train_dataset,
                    idxs=user_groups[idx],
                    logger=logger,
                    encoder=local_encoder, # DUpdated from encoder = shared_encoder coz this isnt copying the model locally
                    head=client_heads[idx],  # Use client-specific head
                    client_id=idx,
                    label_mappings=label_mappings
                )
                # Step 2 FL - Local weights per-client found/updated
                w, loss = local_model.update_weights(global_round=epoch)

                client_train_loss[idx][-1] = float(loss)

                acc_train_frac, _ = local_model.inference()  # accuracy on this client's TRAIN data
                client_train_acc[idx][-1] = float(acc_train_frac) * 100.0  # store as %


                local_weights.append(copy.deepcopy(w))
                participated.add(idx)


            if args.model == 'MultiTaskCNN':
                # Average only the shared encoder weights
                # Step 3 FL - local weights of individual clients averaged
                encoder_weights = average_weights(local_weights)
                # Step 4 FL - Shared encoder updated with encoder (averaged) weights
                shared_encoder.load_state_dict(encoder_weights)
            else:
                global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)

            # per-round server averages (ignore NaNs for non-participants)
            server_avg_train_loss.append(float(np.nanmean([client_train_loss[c][-1] for c in client_train_loss])))
            server_avg_train_acc.append(float(np.nanmean([client_train_acc[c][-1] for c in client_train_acc])))

            # (optional) keep the same value in train_loss for plotting
            train_loss.append(server_avg_train_loss[-1])

            # Calculate avg training accuracy over all users at every epoch
            # list_acc, list_loss = [], []
            if args.model == 'MultiTaskCNN':
                shared_encoder.eval()
                for head in client_heads.values():
                    head.eval()

                test_acc, test_loss, client_test_accuracies, client_test_losses = test_inference_multitask(
                    args, shared_encoder, client_heads, test_dataset, test_user_groups,
                    client_test_accuracies, client_test_losses, label_mappings=label_mappings
                )
                overall_test_acc_history.append(float(test_acc))
                overall_test_loss_history.append(float(test_loss))

            else:
                global_model.eval()


            # Print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f'\nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss: {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(server_avg_train_acc[-1])) # No *100


        print(f'\n Results after {args.epochs} global rounds of training:')
        print('|---- Avg Train Accuracy: {:.2f}%'.format(server_avg_train_acc[-1])) # No *100
        if client_test_accuracies is not None:
            print('|---- Test Accuracy: {:.2f}%'.format(overall_test_acc_history[-1]))
            print('|---- Test Loss: {:.4f}'.format(overall_test_loss_history[-1]))
        else:
            print('|---- Test Accuracy: Not Applicable (e.g., in case of an AutoEncoder')

        # Saving the objects train_loss and train_accuracy
        file_name = '../FMTLScenario1-MTClassification/Results/save/objects/{}_{}_{}_C[{}]_partition[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.partition_type, args.local_ep, args.local_bs)

        save_dir = '../FMTLScenario1-MTClassification/Results/save'
        os.makedirs(save_dir, exist_ok=True)

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump([server_avg_train_loss, server_avg_train_acc], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # Plotting
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plotting curves

        rounds = list(range(1, args.epochs + 1))

        # 1) Training loss (server-average per round)
        plt.figure()
        plt.title('Training Loss vs Communication Rounds')
        plt.plot(rounds, server_avg_train_loss)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Training Loss')
        plt.xticks(rounds)
        plt.savefig('../FMTLScenario1-MTClassification/Results/save/server_avg_train_loss_per_round.png')

        # 2) Training accuracy (server-average per round)
        plt.figure()
        plt.title('Average Train Accuracy vs Communication Rounds')
        plt.plot(rounds, server_avg_train_acc)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Average Train Accuracy (%)')
        plt.xticks(rounds)
        plt.savefig('../FMTLScenario1-MTClassification/Results/save/server_avg_train_accuracy_per_round.png')

        # 3) Test accuracy (overall system)
        plt.figure()
        plt.title('Overall TEST Accuracy vs Communication Rounds')
        plt.plot(rounds, overall_test_acc_history)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(rounds)
        plt.savefig('../FMTLScenario1-MTClassification/Results/save/overall_test_accuracy_per_round.png')


        # 4) Per-client TRAIN accuracy
        def dict_to_df(d, rounds):
            import pandas as pd
            cols = sorted(d.keys())
            data = {f'client_{cid}': d[cid] for cid in cols}
            df = pd.DataFrame(data, index=rounds)
            df.index.name = 'round'
            return df


        import pandas as pd

        save_dir = '../FMTLScenario1-MTClassification/Results/save'

        df_train_acc = dict_to_df(client_train_acc, rounds)
        df_train_acc.to_csv(f'{save_dir}/client_train_accuracy_per_round.csv')

        plt.figure()
        plt.title('Client-wise TRAIN Accuracy per Round')
        for cid in sorted(client_train_acc.keys()):
            plt.plot(rounds, client_train_acc[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Train Accuracy (%)')
        plt.xticks(rounds);
        plt.legend();
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_train_accuracy_per_round.png')

        # 5) Per-client TRAIN loss
        df_train_loss = dict_to_df(client_train_loss, rounds)
        df_train_loss.to_csv(f'{save_dir}/client_train_loss_per_round.csv')

        plt.figure()
        plt.title('Client-wise TRAIN Loss per Round')
        for cid in sorted(client_train_loss.keys()):
            plt.plot(rounds, client_train_loss[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Train Loss')
        plt.xticks(rounds);
        plt.legend();
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_train_loss_per_round.png')

        # 6) Per-client TEST accuracy
        df_test_acc = dict_to_df(client_test_accuracies, rounds)
        df_test_acc.to_csv(f'{save_dir}/client_test_accuracy_per_round.csv')
        plt.figure()
        plt.title('Client-wise TEST Accuracy per Round')
        for cid in sorted(client_test_accuracies.keys()):
            plt.plot(rounds, client_test_accuracies[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(rounds);
        plt.legend();
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_test_accuracy_per_round.png')

        # 7) Per-client TEST loss
        df_test_loss = dict_to_df(client_test_losses, rounds)
        df_test_loss.to_csv(f'{save_dir}/client_test_loss_per_round.csv')
        plt.figure()
        plt.title('Client-wise TEST Loss per Round')
        for cid in sorted(client_test_losses.keys()):
            plt.plot(rounds, client_test_losses[cid], label=f'Client {cid}')
        plt.xlabel('Communication Rounds');
        plt.ylabel('Test Loss')
        plt.xticks(rounds);
        plt.legend();
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_test_loss_per_round.png')

        # 8) Overall test loss
        plt.figure()
        plt.title('Overall TEST Loss vs Communication Rounds')
        plt.plot(rounds, overall_test_loss_history)
        plt.xlabel('Communication Rounds');
        plt.ylabel('Test Loss')
        plt.xticks(rounds)
        plt.savefig(f'{save_dir}/overall_test_loss_per_round.png')

    finally:
        logger.close()


