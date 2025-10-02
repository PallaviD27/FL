# Loss for AutoEncoder still to be included
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from model import CNNMNIST,AutoEncoder
import os

if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # Load dataset
    train_dataset, test_dataset, _ = get_dataset(args)

    # Build model
    if args.model == 'CNN':
        global_model = CNNMNIST(args=args)
        criterion = torch.nn.CrossEntropyLoss().to(device)

    elif args.model == 'AutoEncoder':
        global_model = AutoEncoder() # latent_dim=args.latent_dim removed
        criterion = torch.nn.CrossEntropyLoss().to(device)

    else:
        raise ValueError(f"Invalid model type: {args.model}. Choose 'CNN' or 'AutoEncoder'.")

    # Set the model to train and send it to the device

    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr = args.lr, momentum=0.5)

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr = args.lr, weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # criterion = torch.nn.NLLLoss().to(device)
    #  # MSE to be put here

    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)

            if args.model == 'AutoEncoder':
                loss = criterion(outputs,images)

            else:
                loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            if batch_idx % 50 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.
                      format(epoch+1,batch_idx*len(images), len(trainloader.dataset),
                        100.*batch_idx / len(trainloader), loss.item()))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\n Train Loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss

    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    os.makedirs('../save', exist_ok=True)
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,args.epochs))

    # Test

    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')

    if test_acc is not None:
        print('Test Accuracy: {:.2f}%'.format(100*test_acc))

    else:
        print('No classification accuracy for AutoEncoder')

