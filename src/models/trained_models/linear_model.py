import torch
import torch.nn as nn

class TrafficSignClassifierLinearModel(nn.Module):
    def __init__(self, img_channel=3, img_size=32, num_classes=43):
        super(TrafficSignClassifierLinearModel, self).__init__()
        input_dim = img_size * img_size * img_channel
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
