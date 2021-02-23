import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


class ResnetEncoder(nn.Module):
    def __init__(self, layer_from_last: int):
        super(ResnetEncoder, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet18", pretrained=True
        )
        self.model.eval()
        self.features = nn.Sequential(*list(self.model.children())[:-layer_from_last])
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
        self.encoding_dimension = self._get_encoding_dimension()

    def _get_encoding_dimension(self,):
        _dummy_batch = torch.randint(0, 255, (1, 32, 32, 3)).float()
        _op = self.encoder(_dummy_batch)
        return _op.shape[-1]

    def _preprocess_batch(self, x, device):
        x_transformed = x.cpu().numpy().reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        x_pil = [Image.fromarray(np.uint8(x)).convert("RGB") for x in x_transformed]
        x_pil_processed = [self.preprocess(x) for x in x_pil]
        return torch.stack(x_pil_processed).to(device)

    def class_prediction(self, x):
        return self.model(self._preprocess_batch(x))

    def encoder(self, x, device="cpu"):
        try:
            return self.features(self._preprocess_batch(x, device)).flatten(start_dim=1)
        except:
            print(get_gpu_memory_map())
            assert 1 == 2
