from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv


class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, index):
        return (
            self.embeddings[index],
            self.labels[index],
        )

    def __len__(self):
        return self.embeddings.shape[0]


class Net(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        encoder_model,
        sample_dataset,
        train_dataset,
        num_epochs=5,
        learning_rate=1e-3,
        seed=None,
    ):
        super(Net, self).__init__()
        self.encoder_model = encoder_model
        self.sample_loader = self.get_encoding(sample_dataset, batch_size=64)
        self.train_loader = self.get_encoding(train_dataset, batch_size=1, shuffle=True)
        # make a model that predicts the edges between the nodes

        self.edge_linear = torch.nn.Linear(
            encoder_model.encoding_dimension, encoder_model.encoding_dimension
        )

        self.gconv = GCNConv(
            in_channels=encoder_model.encoding_dimension,
            out_channels=encoder_model.encoding_dimension,
        )

        self.denselayers = nn.Sequential(
            # nn.Flatten(), #already flattened
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.ReLU(),
            nn.Softmax(dim=0),
        )
        self.reset_loader()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()

        self.trainables = (
            self.edge_linear.parameters()
            + self.gconv.parameters()
            + self.denselayers.parameters()
        )

        self.optimizer = torch.optim.Adam(
            self.trainables, lr=learning_rate, weight_decay=1e-5
        )
        self.seed = seed

    def get_encoding(self, ds, batch_size, shuffle=False):
        if not isinstance(ds, Dataset):
            raise Exception(
                f"The Dataset object is not valid. Got object of class {ds.__class__}."
            )
        # intermediate dl for getting embeddings, batch size does not need user input
        _dl = DataLoader(ds, batch_size=64)
        _all_enc_embeddings = []
        _all_enc_labels = []
        for data, label in _dl:
            embedding = self.encoder_model.encoder(data).flatten(
                start_dim=1
            )  # check if flatten will work
            # found that the Graph only takes 1D tensor as node feature
            _all_enc_embeddings.append(embedding)
            _all_enc_labels.append(label)
        _all_enc_embeddings = torch.cat(_all_enc_embeddings).detach()
        _all_enc_labels = torch.cat(_all_enc_labels).detach()
        # dataset from the new embeddings
        ds = CustomDataset(_all_enc_embeddings, _all_enc_labels)
        # dataloader with user defined batch size
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,)
        return dl

    def reset_loader(self):
        self.sample_iterator = iter(self.sample_loader)

    def forward(self, x):
        try:
            sample_xs, _ = self.sample_iterator.__next__()
        except StopIteration:
            self.reset_loader()
            sample_xs, _ = self.sample_iterator.__next__()

        sample_xs = self.edge_linear(sample_xs)
        x = self.edge_linear(x)
        edge_attr = F.softmax(torch.mul(sample_xs, x), dim=0)
        node_features = torch.cat([sample_xs, x])
        num_nodes = len(edge_attr)  # count start from 0
        edges = torch.tensor(
            [
                [_ for _ in range(len(edge_attr))],
                [num_nodes for _ in range(len(edge_attr))],
            ],
            dtype=torch.long,
        )
        # make graph
        g = Data(x=node_features, edge_index=edges, edge_attr=edge_attr)
        aggregated_x = F.elu(
            self.gconv(x=g.x, edge_index=g.edge_index, edge_weight=g.edge_attr)
        )
        # isolate x's feature from the graph
        aggregated_x = aggregated_x[-1, :]
        pred = self.denselayers(aggregated_x)
        return pred

    def train(self):
        if self.seed:
            torch.manual_seed(self.seed)
        loss_info = []
        for epoch in range(self.num_epochs):
            epoch_loss = []
            for data in tqdm(self.train_loader):
                img, label = data
                recon = self.forward(img)
                l = self.criterion(recon, label)
                l.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss.append(float(l))
            loss_info.append(epoch_loss)
            # print("Epoch:{}, Loss:{:.4f}".format(epoch + 1, float(l)))
            # outputs.append((epoch, img, recon),)
        return loss_info
