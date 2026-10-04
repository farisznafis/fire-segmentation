"""Model definition shared by training and ONNX export."""

import segmentation_models_pytorch as smp
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FireSegmenter(nn.Module):
    """U-Net with an ImageNet-pretrained encoder.

    Input: RGB float tensor (N, 3, H, W) in [0, 1]; normalisation happens inside the
    model so the exported ONNX graph needs no extra preprocessing.
    Output: logits (N, 1, H, W).
    """

    def __init__(self, encoder: str = "efficientnet-b0", pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


class ProbabilityHead(nn.Module):
    """Wraps a FireSegmenter so it outputs probabilities (used for export)."""

    def __init__(self, model: FireSegmenter):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))
