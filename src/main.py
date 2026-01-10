import torch
from train_and_evaluate.train_model import train_model
from data.gtsrb_dataset import ensure_data
from data.SignDataset import create_dataloaders
from data.belgium_classification_ds import download_belgium_ds
from mappers.map_ppm_to_png import map_ppm_to_png
from neural_networks_and_models.classifier_linear_nn import (
    TrafficSignClassifierLinearNN,
)
from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from neural_networks_and_models.resnet_model import get_resnet_model
from train_and_evaluate.evaluate_model import evaluate_model
from torch.nn import CrossEntropyLoss

# Póki co main do testów czy wszytsko działa poprawnie

#torch.cuda.is_available = lambda: False


def test_classification_model(train_df, val_df, test_df, meta_df):
    print("Tworzenie dataloaderów...")

    train_loader = create_dataloaders(train_df, batch_size=32, size=(224, 224))
    val_loader = create_dataloaders(val_df, batch_size=32, size=(224, 224))

    print(f"Ilość batchy treningowych: {len(train_loader)}\n")

    print("Inicjalizacja modelu...")
    # model = TrafficSignClassifierLinearNN()
    #model = TrafficSignClassifierConvNN()
    model = get_resnet_model()

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

    test_loader = create_dataloaders(test_df, batch_size=32, size=(224, 224))
    evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        criterion=criterion,
    )
    
    return model


def main():
    print("=== Pobieranie danych ===")
    train_df, val_df, test_df, meta_df = ensure_data()
    print(f"Dane treningowe: {len(train_df)} próbek")
    print(f"Dane walidacyjne: {len(val_df)} próbek\n")

    print("=== klasyfikacja ===")
    #model = test_classification_model(train_df, val_df, test_df, meta_df)
    
    print("test mappera")
    ds = download_belgium_ds()
    map_ppm_to_png(ds)



if __name__ == "__main__":
    main()
