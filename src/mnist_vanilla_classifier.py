import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class VanillaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VanillaClassifier, self).__init__()
        self.denselayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16,num_classes),
            nn.ReLU(),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.denselayers(x)
        return x


def train(model, train_loader, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    loss_info = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for data in tqdm(train_loader):
            img, label = data
            recon = model(img)
            l = criterion(recon, label)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(float(l))
        loss_info.append(epoch_loss)
        # print("Epoch:{}, Loss:{:.4f}".format(epoch + 1, float(l)))
        # outputs.append((epoch, img, recon),)
    return loss_info
