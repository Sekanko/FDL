from data.ensure import get_whole_data
from train_and_evaluate.evaluate_model import evaluate_model
from mappers.map_classes import get_classes_to_names
import torch

def torch_evaluation(model):
    batch_size = 32

    print("Evaluation Setup")
    print("Do you want to specify evaluation batch size? (y/n)")
    if input().lower() == "y":
        batch_size = int(input(f"Batch size (default {batch_size}): ") or batch_size)

    _, _, test_loader = get_whole_data(batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Starting evaluation: batch_size={batch_size}...")

    classes = get_classes_to_names()
  
    evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        classes=classes,
    )

def YOLO_evaluation(model):
    print("Starting YOLO Validation...")
    model.val()
