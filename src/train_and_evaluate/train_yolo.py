import os
import yaml
from ultralytics import YOLO
import torch

def train_yolo_model(path, model, epochs=10, img_size=640, batch_size=16):
    yaml_filename = "data.yaml"

    data_config = {
        "path": os.path.abspath(path),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["traffic_sign"],
    }

    with open(yaml_filename, "w") as f:
        yaml.dump(data_config, f)

    model.train(
        data=yaml_filename,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device="cpu" if not torch.cuda.is_available() else "cuda",

    )

    return model