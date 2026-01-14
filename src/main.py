import os
import time

import cv2

from neural_networks_and_models.models.save_model_structure import ModelRegistry

from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from neural_networks_and_models.classifier_linear_nn import (
    TrafficSignClassifierLinearNN,
)
from neural_networks_and_models.models.load_model import load_model
from neural_networks_and_models.models.save_model import save_model
from neural_networks_and_models.resnet_model import get_resnet_model
from neural_networks_and_models.mobilenet_model import get_mobilenet_model
from neural_networks_and_models.traffic_sign_recognizer import TrafficSignRecognizer
from neural_networks_and_models.yolo_model import load_yolo_model
from train_and_evaluate.train_by_library import torch_training, YOLO_training
from neural_networks_and_models.models.predict_by_library import (
    torch_prediction,
    YOLO_detection,
    recognizer_prediction,
)
from train_and_evaluate.evaluate_by_library import torch_evaluation, YOLO_evaluation


def load_model_procedure():
    print("Choose model type to load:")
    print("1. Linear Neural Network")
    print("2. Convolutional Neural Network")
    print("3. MobileNet Model")
    print("4. ResNet Model")
    print("5. YOLO Model")
    print("Anything else to cancel.")
    choice = input("Enter your choice (1/2/3/4/5): ")

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
                model_registry = ModelRegistry.MOBILENET
                model = load_model(model_registry, version)
            case "4":
                model_registry = ModelRegistry.RESNET
                model = load_model(model_registry, version)
            case "5":
                model_registry = ModelRegistry.YOLO
                model = load_model(model_registry, version)
            case "6":
                model_registry = ModelRegistry.TRAFFIC_SIGN_RECOGNIZER
                model = load_model(model_registry, version)
            case _:
                print("Loading cancelled.")
                return None, None

        return model, model_registry
    except FileNotFoundError as e:
        print(f"Error: {e}")
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

    manage_model_workflow(model, model_registry)


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


def handle_eval(model, model_registry):
    if model_registry == ModelRegistry.TRAFFIC_SIGN_RECOGNIZER:
        print("Hybrid Model Evaluation Menu:")
        print("1. Evaluate Detector (YOLO)")
        print("2. Evaluate Classifier (PyTorch)")
        print("3. Evaluate Both")
        hybrid_choice = input("Choice: ")

        if hybrid_choice in ["1", "3"]:
            YOLO_evaluation(model.detector)

        if hybrid_choice in ["2", "3"]:
            torch_evaluation(model.classifier)

    elif model_registry == ModelRegistry.YOLO:
        YOLO_evaluation(model)

    else:
        torch_evaluation(model)

    print("Evaluation completed successfully.")


def manage_model_workflow(model, model_registry):
    while True:
        print(f"\n--- Model Management: {model_registry.name} ---")
        print("1. Train")
        print("2. Evaluate")
        print("3. Use")
        print("4. Save")
        print("5. Back to menu")
        choice = input("Enter choice (1-5): ")

        match choice:
            case "1":
                model = start_training(model, model_registry)
            case "2":
                handle_eval(model, model_registry)
            case "3":
                img = load_img()
                if img is not None:
                    predictions(model, model_registry, img)
            case "4":
                save_model(model, model_registry)
                time.sleep(1)
            case "5":
                break


def ask_for_model(recognizer_included=True):
    print("\n--- Model Creation Menu ---")
    print("1. Linear Neural Network")
    print("2. Convolutional Neural Network")
    print("3. MobileNet Model")
    print("4. ResNet Model")

    if not recognizer_included:
        return input("Enter your choice (1-4): ")

    print("5. YOLO Model")
    print("6. Traffic Sign Recognizer (YOLO + Classifier)")
    print("7. Back to main menu")
    return input("Enter your choice (1-6): ")


def create_classifier_instance(choice):
    match choice:
        case "1":
            return TrafficSignClassifierLinearNN(), ModelRegistry.LINEAR
        case "2":
            return TrafficSignClassifierConvNN(), ModelRegistry.CONV
        case "3":
            return get_mobilenet_model(), ModelRegistry.MOBILENET
        case "4":
            return get_resnet_model(), ModelRegistry.RESNET
        case _:
            return None, None


def create_model():
    model = None
    model_registry = None

    while model is None:
        choice = ask_for_model()

        if choice == "7":
            return

        if choice in ["1", "2", "3", "4"]:
            model, model_registry = create_classifier_instance(choice)

        elif choice == "5":
            print("Creating YOLO Model...")
            model = load_yolo_model()
            model_registry = ModelRegistry.YOLO

        elif choice == "6":
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
