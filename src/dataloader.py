from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from .interface import DataloaderInterface


class ResNetLoader(DataloaderInterface):
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
        if is_single:
            input_batch = input_batch.unsqueeze(0)

        return input_batch

    def get_embedding(self, input_batch):
        if self.is_cuda_available:
            input_batch = input_batch.to("cuda")
            self.model.to("cuda")

        return self.model(input_batch)

    def train_test_dataloader(self):
        pass
