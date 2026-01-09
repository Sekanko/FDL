import timm
import torch.nn as nn

def get_cifar_model(num_classes=43):
    model = timm.create_model("resnet20", pretrained=True, num_classes=num_classes)
    return model
