"Here implementing both CNN for MNIST and auto-encoder structure"

from torch import nn
import torch.nn.functional as F

class CNNMNIST(nn.Module):
    def __init__(self,args):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=25)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

# "For AutoEncoder"
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28)),
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# For multi-classification

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared encoder used by all clients
class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # MNIST images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.has_printed_shape = False

    def forward(self, x):
        x = self.conv(x)
        if not self.has_printed_shape:
            print(f"Encoder output shape before flattening: {x.shape}")
            self.has_printed_shape=True
        x = x.view(x.size(0), -1)  # flatten
        return x  # shared latent representation

# Client-specific task head (classification)
class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaskHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Combined model: encoder is shared, head is client-specific
class ClientModel(nn.Module):
    def __init__(self, shared_encoder, output_dim):
        super(ClientModel, self).__init__()
        self.encoder = shared_encoder
        self.head = TaskHead(input_dim=320, output_dim=output_dim)  # 320 is output size from encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
