import torch
import os
from ultralytics import YOLO
import re

import neural_networks_and_models.models.save_model_struckture as ModelRegistry

def load_model(model_registry: ModelRegistry, version=None):
    model_info = model_registry.info

    base_dir = os.path.dirname(__file__)
    model_folder_path = os.path.join(base_dir, 'trained_models', model_info.folder_name)

    if version is None:
        version = _get_latest_version(model_folder_path, model_info.prefix, model_info.extension)

    model_path = os.path.join(model_folder_path, f"{model_info.prefix}_v{version}.{model_info.extension}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    if model_info.model_class == YOLO:
        model = YOLO(model_path)
    else:
        model = model_info.model_class()
        model.load_state_dict(torch.load(model_path))
        model = torch.load(model_path)

    return model


def _get_latest_version(folder_path, prefix, extension):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(extension)]

    if not files:
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    versions = [int(re.search(r'_v(\d+)', f).group(1)) for f in files]

    if not versions:
        raise FileNotFoundError(f"No max version found for model {prefix} in {folder_path}.")
    
    print(f"Latest version for model {prefix} is v{max(versions)}")
    return max(versions)