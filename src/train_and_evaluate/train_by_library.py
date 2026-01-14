
import torch
import os

from train_and_evaluate.train_model import train_model
from train_and_evaluate.train_yolo import train_yolo_model
from data.prepare_yolo_data import prepare_yolo_dataset
from data.fetchers import download_polish_dataset
from data.ensure import get_whole_data

def torch_training(model):
    epochs = 10
    lr = 0.001
    batch_size = 32

    print("Training Setup")
    print("Do you want to specify training inputs? (y/n)")
    if input().lower() == "y":
        epochs = int(input(f"Number of epochs (default {epochs}): ") or epochs)
        lr = float(input(f"Learning rate (default {lr}): ") or lr)
        batch_size = int(input(f"Batch size (default {batch_size}): ") or batch_size)

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training: {epochs} epochs, lr={lr}...")

    train_loader, val_loader, _ = get_whole_data(batch_size)

    trained_model = train_model(
        model=model,
        dataloader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
    )

    return trained_model

def YOLO_training(model):
    print("Training Setup for YOLO")
    raw_path = download_polish_dataset()
    detection_path = os.path.join(raw_path, "detection")

    target_path = prepare_yolo_dataset(detection_path)

    y_epochs = 10
    y_img_size = 640
    y_batch_size = 16

    print("Do you want to specify YOLO training inputs? (y/n)")
    if input().lower() == "y":
        y_epochs = int(input(f"Epochs (default {y_epochs}): ") or y_epochs)
        y_img_size = int(input(f"Image size (default {y_img_size}): ") or y_img_size)
        y_batch_size = int(
            input(f"Batch size (default {y_batch_size}): ") or y_batch_size
        )

    print(f"Starting YOLO training on {target_path}...")
    trained_model = train_yolo_model(
        path=target_path,
        model=model,
        epochs=y_epochs,
        img_size=y_img_size,
        batch_size=y_batch_size,
    )

    return trained_model