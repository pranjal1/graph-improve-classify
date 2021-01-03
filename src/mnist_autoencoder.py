import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        self.encoding_dimension = 64

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, train_loader, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    ) 
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch:{}, Loss:{:.4f}".format(epoch + 1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs

if __name__ == "__main__":
    model = Autoencoder()
    max_epochs = 20
    outputs = train(model, num_epochs=max_epochs)
    save_path = os.path.join(os.path.dirname(__file__),"../tmp/mnist_autoencoder.bin")
    torch.save(model.state_dict(), save_path)

    """
    model = Autoencoder()
    model.load_state_dict(torch.load("tmp/mnist_autoencoder.bin"))
    model.eval()
    """
