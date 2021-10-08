import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                                    'unet',
                                    in_channels=64,
                                    out_channels=1,
                                    init_features=32,
                                    pretrained=False)

        self.classifier_layer = nn.Sequential(
            # nn.Linear(1280, 512),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.2),
            # nn.Linear(512, 256),
            nn.Linear(256**2, 2)
        )

    def forward(self, inputs):
        x = self.model(inputs)

        # # Pooling and final linear layer
        # x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        # x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x[:, 0].unsqueeze(1)
