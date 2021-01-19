from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import networkx as nx
from matplotlib import pyplot as plt

from torch_geometric import utils
from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import InMemoryDataset, Data

from .cifar_autoencoder import CifarDataSet


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


class Net:
    def __init__(
        self,
        num_classes,
        encoder_model,
        sample_dataset,
        train_dataset,
        keep_prob=0.7,
        num_epochs=5,
        learning_rate=1e-3,
        train_batch_size=64,
        samples_for_graph=32,
        use_graph=True,
        seed=None,
    ):
        self.encoder_model = encoder_model
        self.keep_prob = keep_prob
        self.train_batch_size = train_batch_size
        self.samples_for_graph = samples_for_graph
        self.use_graph = use_graph
        logger.info("Computing the embedding for the train dataset")
        self.train_loader = self.get_encoding(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

        if self.use_graph:
            logger.info("Will use graph to augment features")
            logger.info("Computing the embedding for the sample dataset")
            self.sample_loader = self.get_encoding(
                sample_dataset,
                batch_size=self.train_batch_size * self.samples_for_graph,
            )
            # make a model that predicts the edges between the nodes
            self.edge_linear = torch.nn.Linear(
                encoder_model.encoding_dimension, encoder_model.encoding_dimension
            )

            self.gconv = GCNConv(
                in_channels=encoder_model.encoding_dimension,
                out_channels=encoder_model.encoding_dimension,
            )
            self.reset_loader()
        else:
            logger.info("Will not use graph to augment features")

        self.denselayers = nn.Sequential(
            # nn.Flatten(), #already flattened
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(in_features=1024, out_features=10),
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()

        if not self.use_graph:
            self.trainables = self.denselayers.parameters()
        else:
            self.trainables = [
                {"params": self.edge_linear.parameters()},
                {"params": self.gconv.parameters()},
                {"params": self.denselayers.parameters()},
            ]

        self.optimizer = torch.optim.Adam(
            self.trainables, lr=learning_rate, weight_decay=1e-5
        )
        self.seed = seed

    def get_encoding(self, ds, batch_size, shuffle=False):
        if not ds.__class__.__name__ == "CifarDataSet":
            raise Exception(
                f"The Dataset object is not valid. Got object of class {ds.__class__}."
            )
        # intermediate dl for getting embeddings, batch size does not need user input
        _dl = DataLoader(ds, batch_size=64)
        _all_enc_embeddings = []
        _all_enc_labels = []
        for data, label in tqdm(_dl):
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

    def forward(self, x, is_train=True):
        if not is_train:
            self.denselayers.eval()
            if self.use_graph:
                self.gconv.eval()
                self.edge_linear.eval()
        else:
            self.denselayers.train()
            if self.use_graph:
                self.gconv.train()
                self.edge_linear.train()
        if self.use_graph:
            try:
                sample_xs, l_ = self.sample_iterator.__next__()
            except StopIteration:
                self.reset_loader()
                sample_xs, l_ = self.sample_iterator.__next__()
            sample_xs = self.edge_linear(sample_xs)
            x = self.edge_linear(x)
            all_graphs = []
            for i in range(x.shape[0]):
                samples_per_x = sample_xs[
                    i * self.samples_for_graph : (i + 1) * self.samples_for_graph
                ]
                one_x = x[i : i + 1]
                edge_attr = F.sigmoid(
                    torch.sum(torch.mul(samples_per_x, one_x), axis=1)
                )
                node_features = torch.cat([samples_per_x, one_x])
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
                all_graphs.append(g)
            graph_data, slices = InMemoryDataset.collate(all_graphs)
            aggregated_x = F.elu(
                self.gconv(
                    x=graph_data.x,
                    edge_index=graph_data.edge_index,
                    edge_weight=graph_data.edge_attr,
                )
            )
            # isolate x's feature from the graph
            x_loc = [_ - 1 for _ in slices["x"][1:]]
            aggregated_x = aggregated_x[x_loc, :]
            pred = self.denselayers(aggregated_x)
        else:
            pred = self.denselayers(x)
        return pred

    def test_forward_run(self, x):
        try:
            sample_xs, l_ = self.sample_iterator.__next__()
        except StopIteration:
            self.reset_loader()
            sample_xs, l_ = self.sample_iterator.__next__()
        assert x.shape[0] == 1
        logger.info(f"x.shape = {x.shape}")
        logger.info(f"x[:,:10] = {x[:,:10]}")
        logger.info(f"sample_xs.shape = {sample_xs.shape}")
        logger.info(f"sample_xs[:5,:10] = {sample_xs[:5,:10]}")
        sample_xs = self.edge_linear(sample_xs)
        x = self.edge_linear(x)
        logger.info(f"Applying Linear Transformation")
        logger.info(f"x[:,:10] = {x[:,:10]}")
        logger.info(f"sample_xs[:5,:10] = {sample_xs[:5,:10]}")
        all_graphs = []
        for i in range(x.shape[0]):
            samples_per_x = sample_xs[
                i * self.samples_for_graph : (i + 1) * self.samples_for_graph
            ]
            one_x = x[i : i + 1]
            logger.info(f"one_x.shape = {one_x.shape}")
            logger.info(f"samples_per_x.shape = {samples_per_x.shape}")
            logger.info(f"samples_per_x[:10] for current x = {samples_per_x[:,:10]}")
            logger.info(
                f"labels for samples_per_x = {l_[i * self.samples_for_graph : (i + 1) * self.samples_for_graph]}"
            )
            edge_attr = F.sigmoid(torch.sum(torch.mul(samples_per_x, one_x), axis=1))
            logger.info(
                f"shape of edge attribute from one_x and sample_for_x = {edge_attr.shape}"
            )
            logger.info(f"edge attribute from one_x and sample_for_x = {edge_attr}")
            node_features = torch.cat([samples_per_x, one_x])
            logger.info(
                f"node feature shape from concat of one_x and sample_for_x = {node_features.shape}"
            )
            logger.info(
                f"node feature from concat of one_x and sample_for_x = {node_features}"
            )
            num_nodes = len(edge_attr)  # count start from 0
            edges = torch.tensor(
                [
                    [_ for _ in range(len(edge_attr))],
                    [num_nodes for _ in range(len(edge_attr))],
                ],
                dtype=torch.long,
            )
            logger.info(
                f"shape of edges directed from sample_for_x to one_x = {edges.shape}"
            )
            logger.info(f"edges directed from sample_for_xs to one_x = {edges}")
            # make graph
            g = Data(x=node_features, edge_index=edges, edge_attr=edge_attr)
            plt.figure(figsize=(10, 10))
            g_nx = utils.to_networkx(g)
            nx.draw_kamada_kawai(g_nx, with_labels=True)
            logger.info("Graph created from one_x and sample_for_x")
            logger.info(f"graph.x.shape = {g.x.shape}")
            logger.info(f"graph.edge_index.shape = {g.edge_index.shape}")
            logger.info(f"graph.edge_attr.shape = {g.edge_attr.shape}")
            all_graphs.append(g)
        logger.info(
            "Colleting graphs for multiple xs. Only one is used for sanity test."
        )
        graph_data, slices = InMemoryDataset.collate(all_graphs)
        logger.info(f"Shape of collated graphs = {graph_data.x.shape}")
        logger.info(f"Position of xs in collated graph = {slices}")
        logger.info("Performing 1 graph conv followed by ReLU")
        aggregated_x = F.elu(
            self.gconv(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_weight=graph_data.edge_attr,
            )
        )
        logger.info(f"Shape of graph xs after conv = {aggregated_x.shape}")
        logger.info(f"graph xs after conv = {aggregated_x}")
        # isolate x's feature from the graph
        x_loc = [_ - 1 for _ in slices["x"][1:]]
        logger.info("Isolating xs after graph conv")
        aggregated_x = aggregated_x[x_loc, :]
        logger.info(f"Shape of augmented x = {aggregated_x.shape}")
        logger.info(f"augmented x = {aggregated_x}")
        logger.info("Passing the augmented x through the dense layers")
        self.denselayers.eval()
        pred = self.denselayers(aggregated_x)
        logger.info(f"Shape of prediction after dense layers = {pred.shape}")
        logger.info(f"Prediction after dense layers = {pred}")

    def train(self):
        if self.seed:
            torch.manual_seed(self.seed)
        loss_info = []
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1} of {self.num_epochs}")
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

    def sanity_test(self):
        loss_info = []
        imgs_batch, labels_batch = self.train_loader.__iter__().__next__()
        img, label = imgs_batch[:1], labels_batch[:1]
        logger.info(f"label of current x = {label}")
        recon = self.test_forward_run(img)
