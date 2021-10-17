import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self, in_channels=64, out_channels=1, init_features=32):
        super(Unet, self).__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                                    'unet',
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    init_features=init_features,
                                    pretrained=False)

        self.classifier_layer = nn.Sequential(
            nn.Linear(256 ** 2, 2)
        )

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.flatten(start_dim=1)
        x = self.classifier_layer(x)
        return x[:, 0].unsqueeze(1)
