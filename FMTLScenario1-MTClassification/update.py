# Local training of the clients happen in this

import torch
from torch import nn  # nn is for neural neural network modules like loss functions and layers.
# Import Dataloader and Dataset to handle the data batching and custom dataset structures.
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class DatasetSplit(Dataset):
    '''
    This class wraps around a Dataset and restricts it to only the subset of indices idxs.
    '''
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        # Store the list of indexes that a particular user/client will use.
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        '''
        This function returns the number of samples in a dataset, based on the number of assigned indexes.
        '''

        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return image.clone().detach(), label.clone().detach()

# Local training on each client; each client train privately
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, encoder=None, head=None, client_id=None, label_mappings=None):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'
        self.client_id = client_id
        base_seed = int(getattr(self.args, "seed", 0) or 0)
        cid = int(self.client_id) if self.client_id is not None else 0
        g = torch.Generator()  # CPU generator (what DataLoader expects)
        g.manual_seed(base_seed + cid)  # unique but repeatable per client
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader = DataLoader(Subset(dataset, list(idxs)), batch_size=self.args.local_bs, shuffle=True, generator=g)



        if args.model == 'CNN':
            # Default criterion set to NLL loss function
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        elif args.model == 'AutoEncoder':
            # When AutoEncoder structure in models, MSE loss to be used
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        elif args.model == 'MultiTaskCNN':
            self.encoder = encoder.to(self.device)
            self.head = head.to(self.device)
            self.label_mappings = label_mappings

            # Combine encoder and head into one model for optimization
            self.model = nn.Sequential(self.encoder, self.head)
            self.criterion = nn.CrossEntropyLoss().to(self.device)  # or CrossEntropyLoss() if output isn't log_softmax
            self.optimizer = None # Optimizer not being built here but where it is needed i.e. update_weights

    def train_val_test(self, dataset,idxs ):
        '''
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        '''

        # Split indexes for train, validation and test (80,10,10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset,idxs_train),
                                 batch_size=self.args.local_bs, shuffle = True)

        validloader = DataLoader(DatasetSplit(dataset,idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)

        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=False)

        return trainloader, validloader, testloader


    def update_weights(self, global_round):
        self.encoder.to(self.device)
        self.head.to(self.device)
        self.encoder.train()
        self.head.train()
        # self.model.train()
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(list(self.encoder.parameters()) + list(self.head.parameters()),
                                         lr=self.args.lr, momentum=self.args.momentum)

            # One-time sanity check; prints once per training call (Felt cute, might delete later)
            ids_opt = {id(p) for g in self.optimizer.param_groups for p in g['params']}
            ids_mod = {id(p) for p in self.encoder.parameters()} | {id(p) for p in self.head.parameters()}
            print(f"[Client {self.client_id}] Optimizer has all params:", ids_mod <= ids_opt)
            # One-time sanity check (Felt cute, might delete later)


        epoch_loss = []

        print(f"--> Starting training for Client {self.client_id}")

        for epoch in range(self.args.local_ep):
            print(f"    Epoch {epoch + 1}/{self.args.local_ep}")
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                print(f"        Batch {batch_idx + 1}")
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long() #casting dtype

                # Apply label mapping if applicable i.e. if the client has one
                if self.label_mappings is not None and self.client_id in self.label_mappings:
                    map_fn = self.label_mappings[self.client_id]
                    labels = torch.tensor([map_fn(y.item()) for y in labels],
                        device=self.device, dtype=torch.long)

                # Checking bounds before loss; assumes the head's last Linear is in a lit/Sequential like head.fc[-1]
                if hasattr(self.head, "fc"):
                    out_dim = self.head.fc[-1].out_features
                elif isinstance(self.head, torch.nn.Sequential):
                    out_dim = list(self.head.children())[-1].out_features
                else:
                    # generic fallback: last Linear in the head
                    last_linear = [m for m in self.head.modules() if isinstance(m, torch.nn.Linear)][-1]
                    out_dim = last_linear.out_features

                if labels.min() < 0 or labels.max() >= out_dim:
                    raise ValueError(
                        f"[Train][Client {self.client_id}] Mapped labels out of range: "
                        f"min={labels.min().item()} max={labels.max().item()} vs head_out={out_dim}"
                    )

                self.optimizer.zero_grad()
                outputs = self.head(self.encoder(images))
                loss = self.criterion(outputs, labels) # FL Loss - Client's batch loss; Cross-Entropy
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item()) #  FL Loss - Storing it for "this" batch

            epoch_loss.append(sum(batch_loss) / len(batch_loss)) #  FL Loss - epoch loss is average of the batch losses in that epoch

        avg_loss = sum(epoch_loss) / len(epoch_loss) # FL Loss - Average loss over local epochs = loss for a communication round

        # (this is optional but we are doing it here) switch back to eval when done
        self.encoder.eval()
        self.head.eval()

        #  Step xx FL - Return only encoder weights if MultiTaskCNN, else full model ( used in Fed Avg)
        if self.args.model == 'MultiTaskCNN':
            return self.encoder.state_dict(), avg_loss
        else:
            return self.model.state_dict(), avg_loss

    def inference(self):
        self.encoder.eval()
        self.head.eval()
        self.model.eval()

        loss_sum, count = 0.0, 0
        total, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long() # Specifiying dtype explicitly

                # Apply label mapping once
                if self.label_mappings is not None and self.client_id in self.label_mappings:
                    map_func = self.label_mappings[self.client_id]
                    labels = torch.tensor([map_func(label.item()) for label in labels],
                                          device=self.device, dtype=torch.long)



                # Optional sanity check (same logic as in update_weights)
                if hasattr(self.head, "fc"):
                    out_dim = self.head.fc[-1].out_features
                elif isinstance(self.head, torch.nn.Sequential):
                    out_dim = list(self.head.children())[-1].out_features
                else:
                    last_linear = [m for m in self.head.modules() if isinstance(m, torch.nn.Linear)][-1]
                    out_dim = last_linear.out_features

                if labels.min() < 0 or labels.max() >= out_dim:
                    raise ValueError(f"[Infer][Client {self.client_id}] Mapped labels out of range "
                                     f"(min={labels.min().item()}, max={labels.max().item()}, out_dim={out_dim})")

                # Optional sanity check

                # outputs = self.model(images)
                # batch_loss = self.criterion(outputs, labels)
                # loss += batch_loss.item()

                # Same bound check already used in training

                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels)  # mean over batch
                loss_sum += batch_loss.item() * labels.size(0)  # accumulate by sample
                count += labels.size(0)


                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0 # Client accuracy based on correct predictions from total predictions
        avg_loss = loss_sum /count if count > 0 else 0.0
        return accuracy, avg_loss


def test_inference(args, model, test_dataset):
    '''
    Returns the accuracy and the loss
    :param model:
    :param test_dataset:
    :return: accuracy, loss
    '''

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    if args.model=='CNN':
        criterion = nn.CrossEntropyLoss().to(device)

    elif args.model=='AutoEncoder':
        # MSE loss for auto-encoder; one of the criterion is to be commented out
        criterion = nn.CrossEntropyLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        if args.model == 'AutoEncoder':
            batch_loss = criterion(outputs, images)
            loss += batch_loss.item()
            total += images.size(0)
        else:
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs,1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    if args.model == 'AutoEncoder':
        return None, loss/total # For autoencoder only loss is returned, accuracy can be None or 0

    else:
        accuracy = correct/total
        return accuracy, loss / total

from torch.utils.data import DataLoader, Subset
import torch.nn as nn

def test_inference_multitask(
    args, encoder, heads, test_dataset, user_groups,
    client_test_accuracies=None, client_test_losses=None,
    label_mappings=None
):
    """
    Returns:
        avg_acc (overall %), avg_loss (overall per-sample),
        client_test_accuracies: dict[int, list[float]]  # % per client per round
        client_test_losses:     dict[int, list[float]]  # per-sample loss per client per round
    """
    device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'

    # init dicts if needed
    if client_test_accuracies is None:
        client_test_accuracies = {}
    if client_test_losses is None:
        client_test_losses = {}

    # convenience: last Linear out_features of a head
    def head_out_dim(h):
        if hasattr(h, "fc"):
            return h.fc[-1].out_features
        elif isinstance(h, torch.nn.Sequential):
            return list(h.children())[-1].out_features
        else:
            last_linear = [m for m in h.modules() if isinstance(m, torch.nn.Linear)][-1]
            return last_linear.out_features

    total_correct, total_samples = 0, 0
    total_loss_sum = 0.0

    # ensure encoder is on correct device (heads are moved when wrapped into model)
    encoder = encoder.to(device)

    for client_id in range(args.num_users):
        head = heads[client_id]
        model = nn.Sequential(encoder, head).to(device)
        model.eval()

        test_loader = DataLoader(
            Subset(test_dataset, list(user_groups[client_id])),
            batch_size=64, shuffle=False
        )
        criterion = nn.CrossEntropyLoss().to(device)

        correct, samples = 0, 0
        loss_sum = 0.0  # accumulate **per-sample** loss
        out_dim = head_out_dim(head)

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).long()   # CE needs Long target

                # client-specific label remap (if provided)
                if label_mappings and client_id in label_mappings:
                    map_fn = label_mappings[client_id]
                    mapped = [map_fn(int(y.item())) for y in labels]
                    labels = torch.tensor(mapped, dtype=torch.long, device=device)

                # bound check once (avoids IndexError in CE)
                if labels.numel() > 0 and (labels.min() < 0 or labels.max() >= out_dim):
                    raise ValueError(
                        f"[Client {client_id}] Mapped label(s) out of range for head with {out_dim} outputs. "
                        f"min={labels.min().item()}, max={labels.max().item()}"
                    )

                outputs = model(images)

                # loss returned is mean over the batch -> convert to sum then to per-sample
                bs = labels.size(0)
                batch_loss = criterion(outputs, labels)   # mean over batch
                loss_sum   += batch_loss.item() * bs
                samples    += bs

                _, predicted = torch.max(outputs, 1)
                correct    += (predicted == labels).sum().item()

        # per-client stats for this round
        acc_client = 100.0 * correct / samples if samples > 0 else 0.0
        loss_client = loss_sum / samples if samples > 0 else 0.0

        # stash into dicts
        if client_id not in client_test_accuracies:
            client_test_accuracies[client_id] = []
            client_test_losses[client_id]     = []
        client_test_accuracies[client_id].append(acc_client)
        client_test_losses[client_id].append(loss_client)

        # accumulate overall
        total_correct  += correct
        total_loss_sum += loss_sum
        total_samples  += samples

    avg_acc  = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0.0
    return avg_acc, avg_loss, client_test_accuracies, client_test_losses
