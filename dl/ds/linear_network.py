import torch
from torch import nn, optim


class LinearNetwork(nn.Module):

    # Assumes that dims contain one more entry than the required no. of layers
    def __init__(self, dims, last_layer_activation=None, lr=0.0001):
        super(LinearNetwork, self).__init__()

        self.lr = lr

        # Creating the model
        layers = []
        n_dims = len(dims)
        for idx in range(1, n_dims - 1):
            layers.append(nn.Linear(dims[idx - 1], dims[idx]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(dims[n_dims - 2], dims[n_dims - 1]))
        if last_layer_activation:
            if last_layer_activation == 'relu':
                layers.append(nn.ReLU(True))
            elif last_layer_activation == 'sigmoid':
                layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

        # Optimization function
        self.optimizer = optim.Adam(self.main.parameters(), self.lr)

    def forward(self, x):
        return self.main(x)

    def unfreeze_params(self):
        for p in self.main.parameters():
            p.requires_grad = True

    def freeze_params(self):
        for p in self.main.parameters():
            p.requires_grad = False

    def optimization_step(self):
        self.optimizer.step()
