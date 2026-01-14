import torch.nn as nn
from torchvision import models

def get_mobilenet_model(num_classes=43, pretrained=True):
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
    else:
        weights = None

    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
