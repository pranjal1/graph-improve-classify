import os
import pickle
from PIL import Image

import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch_geometric.data import InMemoryDataset, Data

from .interface import EmbeddingInterface


class ResNetLoader(EmbeddingInterface):
    def __init__(self, config):
        super().__init__()
        self.is_cuda_available = torch.cuda.is_available()
        self.allowed_variants = [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]
        self.load_model(config)
        self.load_transform_pipeline()

    def load_transform_pipeline(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_model(self, config):
        if config.variant not in self.allowed_variants:
            raise Exception(
                "Unknown resnet variant. Please choose one from {}".format(
                    ", ".join(self.allowed_variants)
                )
            )
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", config.variant, pretrained=True
        )
        modules = list(self.model.children())[: -config.layer_from_end]
        self.model = nn.Sequential(*modules)
        for p in self.model.parameters():
            p.requires_grad = False

    def transform(self, image, is_single=False):
        if isinstance(image, str):
            image = Image.open(image)
        input_batch = self.preprocess(image)
        input_batch = input_batch.unsqueeze(0)
        return input_batch

    def get_embedding(self, input_batch):
        if self.is_cuda_available:
            input_batch = input_batch.to("cuda")
            self.model.to("cuda")

        return self.model(input_batch)


class CifarDataset(InMemoryDataset):
    def __init__(
        self, root, config, transform=None, pre_transform=None, pre_filter=None,
    ):
        self.config = config
        self.embedding = ResNetLoader(self.config)
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices, self.num_embeddings = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_file_names(self):
        logger.info("Required files check!")
        return self.config.raw_files

    @property
    def processed_file_names(self):
        return [
            os.path.join(os.path.dirname("__path__"), "../", "tmp/cifar_processed.dat"),
        ]

    def download(self):
        logger.error(
            f"Can not find files {self.raw_paths}. Please check and place the files in correct path!"
        )
        raise FileNotFoundError

    def unpickle(self, f):
        with open(f, "rb") as fo:
            dct = pickle.load(fo, encoding="bytes")
        img_all = dct.get(b"data").reshape((-1, 3, 32, 32))
        return (
            [
                Image.fromarray(np.uint8(np.transpose(x, axes=(1, 2, 0)))).convert(
                    "RGB"
                )
                for x in img_all
            ],
            dct.get(b"labels"),
        )

    def process(self):
        data_list = []

        logger.info("Processing training dataset...")
        lst_imgs, lst_labels = [], []
        # return the 1st n-1 batches for training
        for fi_path in self.config.raw_files:
            i, l = self.unpickle(fi_path)
            i = [
                self.embedding.get_embedding(self.embedding.transform(x))
                for x in tqdm(i[:100])
            ]
            lst_imgs.append(i)
            lst_labels.extend(l)
        data = np.row_stack(lst_imgs)
        data, slices = self.collate(data)
        torch.save((data, slices, lst_labels), self.processed_paths[0])
        logger.info(f"Processed files saved as {self.processed_paths[0]}")


if __name__ == "__main__":
    o = CifarDataset(root="../tmp/")
    print(o.data)
