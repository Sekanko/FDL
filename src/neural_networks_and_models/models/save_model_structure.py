
from dataclasses import dataclass
from enum import Enum

from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from neural_networks_and_models.classifier_linear_nn import TrafficSignClassifierLinearNN
from neural_networks_and_models.traffic_sign_recognizer import TrafficSignRecognizer
from timm.models import resnet18
from ultralytics import YOLO


@dataclass
class ModelInfo:
    model_class: any
    folder_name: str
    prefix: str
    extension: str


class ModelRegistry(Enum):
    LINEAR = ModelInfo(
        model_class=TrafficSignClassifierLinearNN,
        folder_name='linear_nn_model',
        prefix='LNN_model',
        extension='pth'
    )
    CONV = ModelInfo(
        model_class=TrafficSignClassifierConvNN,
        folder_name='conv_nn_model',
        prefix='CNN_model',
        extension='pth'
    )
    RESNET = ModelInfo(
        model_class=resnet18,
        folder_name='resnet_model',
        prefix='ResNet_model',
        extension='pth'
    )
    TRAFFIC_SIGN_RECOGNIZER = ModelInfo(
        model_class=TrafficSignRecognizer,
        folder_name='recognizer_model',
        prefix='TSR_model',
        extension='pth'
    )
    YOLO = ModelInfo(
        model_class=YOLO,
        folder_name='yolo_model',
        prefix='YOLO_model',
        extension='pt'
    )

    @property
    def info(self):
        return self.value



