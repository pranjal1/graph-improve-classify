import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet18", pretrained=True
        )
        self.model.eval()
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    def _preprocess_batch(self, x):
        x_transformed = x.numpy().reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        x_pil = [Image.fromarray(np.uint8(x)).convert("RGB") for x in x_transformed]
        x_pil_processed = [self.preprocess(x) for x in x_pil]
        return torch.stack(x_pil_processed)

    def class_prediction(self, x):
        return self.model(self._preprocess_batch(x))

    def encoder(self, x):
        return self.features(self._preprocess_batch(x)).flatten(start_dim=1)
