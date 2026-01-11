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

# Póki co main do testów czy wszytsko działa poprawnie

# torch.cuda.is_available = lambda: False


def test_classification_model(train_df, val_df, test_df, meta_df, size=(224, 224)):
    print("Tworzenie dataloaderów...")

    train_loader = create_dataloaders(train_df, batch_size=32, size=size)
    val_loader = create_dataloaders(val_df, batch_size=32, size=size)
    print(f"Ilość batchy treningowych: {len(train_loader)}\n")

    print("Inicjalizacja modelu...")
    # model = TrafficSignClassifierLinearNN()
    model = TrafficSignClassifierConvNN()
    #model = get_resnet_model()

    print(f"Model:\n{model}\n")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Rozpoczęcie trenowania")

    trained_model = train_model(
        model=model,
        val_loader=val_loader,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,
    )

    test_loader = create_dataloaders(test_df, batch_size=32, size=size)
    evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        criterion=criterion,
    )
    
    return model


def main():
    print("=== Pobieranie danych ===")
    # train_df, val_df, test_df, meta_df = ensure_data()


    # print("=== klasyfikacja ===")
    # yolo = load_yolo_model()

    # recognizer = TrafficSignRecognizer(detector=yolo, classifier=model)

    # img_path = os.path.join(os.getcwd(), 'image.png')

    # with torch.no_grad():
    #     results = recognizer(img_path)

    # if not results:
    #     print("Nie wykryto żadnych znaków.")
    # else:
    #     for i, prediction in enumerate(results):
    #         # prediction to tensor [1, liczba_klas]
            
    #         # 1. Wybieramy indeks klasy z najwyższym wynikiem
    #         class_id = torch.argmax(prediction, dim=1).item()
            
    #         # 2. (Opcjonalnie) Obliczamy pewność w %
    #         prob = torch.softmax(prediction, dim=1).max().item()
            
    #         print(f"Obiekt {i+1}:")
    #         print(f"  -> Klasa ID: {class_id}")
    #         print(f"  -> Pewność: {prob:.2%}")
    #         print("-" * 20)


    x, y, z = german_data_as_df()
    print(x)
    d, _, _ = polish_data_as_df()
    print(d)
    f, _, _ = belgium_data_as_df()
    print(f)
    merged = merge_dataframes([x, d, f])
    print(merged)

if __name__ == "__main__":
    main()
