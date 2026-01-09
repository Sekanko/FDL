import timm
import torch.nn as nn

def get_resnet_model(num_classes=43):
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    return model
