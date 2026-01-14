import torch
import os

from ultralytics import YOLO
from src.neural_networks_and_models.models.save_model_structure import ModelRegistry


def save_model(model, registry_member: ModelRegistry):
    info = registry_member.info
    save_path = get_save_path(info.folder_name, info.prefix, info.extension)

    if isinstance(model, YOLO):
        model.save(save_path)
    else:
        torch.save(model.state_dict(), save_path)

def get_save_folder_path(name):
    base_dir = os.path.dirname(__file__)
    save_model_path = os.path.join(base_dir, 'trained_models', name)

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    return save_model_path

def get_save_path(folder_name, model_name, extension):
    save_folder_path = get_save_folder_path(folder_name)
    model_number = 0

    full_path = os.path.join(save_folder_path, f"{model_name}_v{model_number}.{extension}")

    while os.path.exists(full_path):
        model_number += 1
        full_path = os.path.join(save_folder_path, f"{model_name}_v{model_number}.{extension}")

    return full_path