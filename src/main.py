import os
import time

import cv2
import torch
import torchvision.transforms as T
import kagglehub

from neural_networks_and_models.models.save_model_struckture import ModelRegistry
from data.SignDataset import create_dataloaders
from data.ensure import belgium_data_as_df, german_data_as_df, polish_data_as_df
from data.merge import merge_dataframes
from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from neural_networks_and_models.classifier_linear_nn import (
    TrafficSignClassifierLinearNN,
)
from neural_networks_and_models.models.load_model import load_model
from neural_networks_and_models.models.save_model import save_model
from neural_networks_and_models.resnet_model import get_resnet_model
from neural_networks_and_models.traffic_sign_recognizer import TrafficSignRecognizer
from neural_networks_and_models.yolo_model import load_yolo_model
from train_and_evaluate.evaluate_model import evaluate_model
from train_and_evaluate.train_model import train_model
from train_and_evaluate.train_yolo import train_yolo_model
from data.prepare_yolo_data import prepare_yolo_dataset


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
                model_registry = ModelRegistry.LINEAR
                model = load_model(model_registry, version)
            case "2":
                model_registry = ModelRegistry.CONV
                model = load_model(model_registry, version)
            case "3":
                model_registry = ModelRegistry.RESNET
                model = load_model(model_registry, version)
            case "4":
                model_registry = ModelRegistry.YOLO
                model = load_model(model_registry, version)
            case "5":
                model_registry = ModelRegistry.TRAFFIC_SIGN_RECOGNIZER
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
    path = input("Path: ").strip().replace('"', "").replace("'", "")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)

    img = cv2.imread(full_path)
    if img is None:
        print("Image not found.")
        return None

    return img


def torch_prediction(model, model_registry, img):
    print(f"Processing with PyTorch Model ({model_registry.name})")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocess = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((32, 32)),
            T.ToTensor(),
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
    selected_classes = None
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


def recognizer_prediction(model, img):
    class_ids = model(img, conf=0.25, classes=None)

    if not class_ids:
        print("No signs detected.")
        return []

    for idx, cid in enumerate(class_ids):
        # TODO
        # Maper na nazwę

        print(f"Sign {idx+1}: Recognized Class ID {cid}")

    return class_ids


def predictions(model, model_registry, img):
    if model_registry == ModelRegistry.TRAFFIC_SIGN_RECOGNIZER:
        recognizer_prediction(model, img)
    elif model_registry == ModelRegistry.YOLO:
        return YOLO_detection(model, img)
    else:
        return torch_prediction(model, model_registry, img)


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
                    predictions(model, model_registry, img)
                else:
                    print("Failed to load image. Try again.")
            case "2":
                break
            case _:
                print("Invalid choice. Please try again.")


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

    df = merge_dataframes(
        [german_data_as_df(), polish_data_as_df(), belgium_data_as_df()]
    )
    train_loader, val_loader, _ = create_dataloaders(df, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training: {epochs} epochs, lr={lr}...")
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
    # TODO
    # Omówić z damianem pozyskiwanie ściezki z tego datasetu
    raw_path = kagglehub.dataset_download("chriskjm/polish-traffic-signs-dataset")
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


def start_training(model, model_registry):
    if model_registry == ModelRegistry.TRAFFIC_SIGN_RECOGNIZER:
        print("Hybrid Model Training Menu:")
        print("1. Train Detector (YOLO)")
        print("2. Train Classifier (PyTorch)")
        print("3. Train Both")
        hybrid_choice = input("Choice: ")

        if hybrid_choice in ["1", "3"]:
            model.detector = YOLO_training(model.detector)

        if hybrid_choice in ["2", "3"]:
            model.classifier = torch_training(model.classifier)

        trained_model = model
    elif model_registry == ModelRegistry.YOLO:
        trained_model = YOLO_training(model)
    else:
        trained_model = torch_training(model)

    print("Training completed successfully.")
    return trained_model


def manage_model_workflow(model, model_registry):
    while True:
        print(f"\n--- Model Management: {model_registry.name} ---")
        print("1. Train (Szkolenie z parametrami)")
        print("2. Use (Inference test na obrazku)")
        print("3. Evaluate (Pełny test na zbiorze testowym)")
        print("4. Save (Zapisz model do /trained_models)")
        print("5. Back to menu")
        choice = input("Enter choice (1-5): ")

        match choice:
            case "1":
                model = start_training(model, model_registry)
            case "2":
                img = load_img()
                if img is not None:
                    predictions(model, model_registry, img)
            case "3":
                # TODO
                pass
            case "4":
                save_model(model, model_registry)
            case "5":
                break


def ask_for_model(recognizer_included=True):
    print("\n--- Model Creation Menu ---")
    print("1. Linear Neural Network")
    print("2. Convolutional Neural Network")
    print("3. ResNet Model")

    if not recognizer_included:
        return input("Enter your choice (1-3): ")

    print("4. YOLO Model")
    print("5. Traffic Sign Recognizer (YOLO + Classifier)")
    print("6. Back to main menu")
    return input("Enter your choice (1-6): ")


def create_classifier_instance(choice):
    match choice:
        case "1":
            return TrafficSignClassifierLinearNN(), ModelRegistry.LINEAR
        case "2":
            return TrafficSignClassifierConvNN(), ModelRegistry.CONV
        case "3":
            return get_resnet_model(), ModelRegistry.RESNET
        case _:
            return None, None


def create_model():
    model = None
    model_registry = None

    while model is None:
        choice = ask_for_model()

        if choice == "6":
            return

        if choice in ["1", "2", "3"]:
            model, model_registry = create_classifier_instance(choice)

        elif choice == "4":
            print("Creating YOLO Model...")
            model = load_yolo_model()
            model_registry = ModelRegistry.YOLO

        elif choice == "5":
            print("Creating Traffic Sign Recognizer (Hybrid System)...")
            detector = load_yolo_model()

            classifier = None
            while classifier is None:
                print("\nSelect classifier for the Hybrid System:")
                sub_choice = ask_for_model(recognizer_included=False)
                classifier, _ = create_classifier_instance(sub_choice)
                if classifier is None:
                    print("Please select a correct option (1-3).")
                    time.sleep(1)

            model = TrafficSignRecognizer(detector, classifier)
            model_registry = ModelRegistry.TRAFFIC_SIGN_RECOGNIZER

        else:
            print("Invalid choice. Please try again.")

    manage_model_workflow(model, model_registry)


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
