import torch
import os
from neural_networks_and_models.classifier_linear_nn import (
    TrafficSignClassifierLinearNN,
)
from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from timm.models.resnet import ResNet
from neural_networks_and_models.traffic_sign_recognizer import TrafficSignRecognizer
from ultralytics import YOLO


def save_model(model):
    match model:
        case TrafficSignClassifierLinearNN():
            save_path = get_save_path('linear_nn_model', 'LNN_model')
            torch.save(model.state_dict(), save_path)
            print("Linear model saved")
        case TrafficSignClassifierConvNN():
            save_path = get_save_path('conv_nn_model', 'CNN_model')
            torch.save(model.state_dict(), save_path)
            print("Convolutional model saved")
        case ResNet():
            save_path = get_save_path('resnet_model', 'ResNet_model')
            torch.save(model.state_dict(), save_path)
            print("ResNet model saved")
        case TrafficSignRecognizer():
            save_path = get_save_path('recognizer_model', 'TSR_model')
            torch.save(model.state_dict(), save_path)
            print("Traffic Sign Recognizer model saved")
        case YOLO():
            save_path = get_save_path('yolo_model', 'YOLO_model', extension='pt')
            model.save(save_path)
            print("YOLO model saved")
        case _:
            raise TypeError("Unsupported model type for saving.")

def get_save_folder_path(name):
    base_dir = os.path.dirname(__file__)
    save_model_path = os.path.join(base_dir, 'trained_models', name)

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    return save_model_path

def get_save_path(folder_name, model_name, extension='pth'):
    save_folder_path = get_save_folder_path(folder_name)
    model_number = 0

    full_path = os.path.join(save_folder_path, f"{model_name}_v{model_number}.{extension}")

    while os.path.exists(full_path):
        model_number += 1
        full_path = os.path.join(save_folder_path, f"{model_name}_v{model_number}.{extension}")

    return full_path