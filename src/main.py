import os
from neural_networks_and_models.traffic_sign_recognizer import TrafficSignRecognizer
from neural_networks_and_models.yolo_model import load_yolo_model
import torch
from train_and_evaluate.train_model import train_model
from data.SignDataset import create_dataloaders
from data.ensure import german_data_as_df, polish_data_as_df, belgium_data_as_df
from data.merge import merge_dataframes
from neural_networks_and_models.classifier_linear_nn import (
    TrafficSignClassifierLinearNN,
)
from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from neural_networks_and_models.resnet_model import get_resnet_model
from train_and_evaluate.evaluate_model import evaluate_model
from torch.nn import CrossEntropyLoss
import os
import pandas as pd
from mappers.map_classes import get_classes_to_names
import neural_networks_and_models.models.save_model_struckture as ModelRegistry
from neural_networks_and_models.models.load_model import load_model
import time
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
from PIL import Image


def create_model():
    while True:
        print("Select model type to create:")
        print("1. Linear Neural Network")
        print("2. Convolutional Neural Network")
        print("3. ResNet Model")
        print("4. YOLO Model")
        print("5. Back to main menu")
        choice = input("Enter your choice (1/2/3/4/5): ")

        match choice:
            case "1":
                print("Creating Linear Neural Network...")
                # Add logic for creating Linear NN
            case "2":
                print("Creating Convolutional Neural Network...")
                # Add logic for creating Conv NN
            case "3":
                print("Creating ResNet Model...")
                # Add logic for creating ResNet Model
            case "4":
                print("Creating YOLO Model...")
                # Add logic for creating YOLO Model
            case "5":
                break
            case _:
                print("Invalid choice. Please try again.")


def load_model_procedure():
    print("Choose model type to load:")
    print("1. Linear Neural Network")
    print("2. Convolutional Neural Network")
    print("3. ResNet Model")
    print("4. YOLO Model")
    print("Anything else to cancel.")
    choice = input("Enter your choice (1/2/3/4): ")

    print("Choose version to load (or press Enter for latest):")
    version_input = input("Version: ")
    version = int(version_input) if version_input else None

    try:
        match choice:
            case "1":
                model_registry = ModelRegistry.ModelRegistry.LINEAR
                model = load_model(model_registry, version)
            case "2":
                model_registry = ModelRegistry.ModelRegistry.CONV
                model = load_model(model_registry, version)
            case "3":
                model_registry = ModelRegistry.ModelRegistry.RESNET
                model = load_model(model_registry, version)
            case "4":
                model_registry = ModelRegistry.ModelRegistry.YOLO
                model = load_model(model_registry, version)
            case _:
                print("Loading cancelled.")
                return None, None

        return model, model_registry
    except FileNotFoundError:
        print("Such model does not exist.")
        return None, None


def load_img():
    print("Enter relative to src/main.py path to image:")
    path = input("Path: ")

    img = cv2.imread(path)
    if img is None:
        print("Image not found.")
        return None

    return img


def torch_prediction(model, model_registry, img):
    print(f"--- Processing with PyTorch Model ({model_registry.name}) ---")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocess = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(img_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    print(f"Prediction: Class ID {prediction}")
    return prediction


def YOLO_detection(model, img):
    conf_threshold = 0.7
    selected_classes = [0]
    is_verbose = False

    print("YOLO Prediction Settings")
    print("Do you want to customize prediction settings? (y/n): ")
    choice = input().lower()

    if choice == "y":
        conf_input = input("Confidence (default 0.7): ")
        conf_threshold = float(conf_input) if conf_input else 0.7

        classes_entered = input(
            "Classes IDs to look for, comma separated (e.g. 0,1,5) or Enter for all: "
        )
        if classes_entered:
            selected_classes = [int(x.strip()) for x in classes_entered.split(",")]

        verb_choice = input("Show full YOLO logs? (y/n): ").lower()
        is_verbose = True if verb_choice == "y" else False

    results = model.predict(
        source=img,
        conf=conf_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=is_verbose,
        classes=selected_classes,
    )

    for r in results:
        print(f"Found {len(r.boxes)} objects.")
        for box in r.boxes:
            c = int(box.cls)
            prob = box.conf[0].item()
            print(f" - Class: {c}, Confidence: {prob:.2f}")

    return results


def predictions(model, model_registry, img):
    if model_registry != ModelRegistry.ModelRegistry.YOLO:
        return torch_prediction(model, model_registry, img)
    else:
        return YOLO_detection(model, img)


def use_model():
    model, model_registry = load_model_procedure()
    if model is None:
        time.sleep(2)
        return

    while True:
        print("Model loaded successfully.")
        print("1. Use model")
        print("2. Back to main menu")
        choice = input("Enter your choice (1/2): ")
        match choice:
            case "1":
                img = load_img()
                if img is not None:
                    predictions(model,model_registry, img)
                else:
                    print("Failed to load image. Try again.")
            case "2":
                break
            case _:
                print("Invalid choice. Please try again.")


def starting_app():
    print("Starting the application...")
    print("Should program use GPU if available? (y/n): ")
    gpu_choice = input().lower()

    if gpu_choice == "n":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    starting_app()

    while True:
        print("Select an option:")
        print("1. Create model")
        print("2. Load model")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        match choice:
            case "1":
                create_model()
            case "2":
                print("Use an existing model...")
                use_model()
            case "3":
                print("Exiting the program.")
                break
            case _:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":

    main()
