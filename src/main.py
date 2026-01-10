import torch
from train_and_evaluate.train_model import train_model
from data.gtsrb_dataset import ensure_data
from data.SignDataset import create_dataloaders
from neural_networks_and_models.classifier_linear_nn import TrafficSignClassifierLinearNN
from neural_networks_and_models.classifier_conv_nn import TrafficSignClassifierConvNN
from neural_networks_and_models.resnet_model import get_resnet_model
from neural_networks_and_models.detector_conv_nn import TrafficSignDetectorConvNN
from neural_networks_and_models.detector_linear_nn import TrafficSignDetectorLinearNN
from train_and_evaluate.evaluate_model import evaluate_model
from torch.nn import CrossEntropyLoss

# Póki co main do testów czy wszytsko działa poprawnie

torch.cuda.is_available = lambda: False

def test_classification_model(train_df, val_df, test_df, meta_df):
    print("Tworzenie dataloaderów...")

    train_loader = create_dataloaders(train_df, batch_size=32, size=(32, 32))
    val_loader = create_dataloaders(val_df, batch_size=32, size=(32, 32))

    print(f"Ilość batchy treningowych: {len(train_loader)}\n")
    
    print("Inicjalizacja modelu...")
    # model = TrafficSignClassifierLinearNN()
    model = TrafficSignClassifierConvNN()
    # model = get_resnet_model()

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
        is_classification=True,
    )

    test_loader = create_dataloaders(test_df, batch_size=32, size=(32, 32))
    evaluate_model(model=trained_model, test_loader=test_loader, criterion=criterion, is_classification=True)

    

def test_detection_model(train_df, val_df, test_df, meta_df):
    print("Tworzenie dataloaderów dla detekcji...")
    
    train_loader = create_dataloaders(train_df, batch_size=32, size=(224, 224), mode="detection")
    val_loader = create_dataloaders(val_df, batch_size=32, size=(224, 224), mode="detection")
    
    print(f"Ilość batchy treningowych: {len(train_loader)}\n")
    
    print("Inicjalizacja modelu detektora...")
    
    model = TrafficSignDetectorConvNN()
    # model = TrafficSignDetectorLinearNN()
    
    print(f"Model detektora:\n{model}\n")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Rozpoczęcie trenowania detektora...")
    
    trained_detector = train_model(
        model=model,
        dataloader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        is_classification=False,
    )
 

def main():
    print("=== Pobieranie danych ===")
    train_df, val_df, test_df, meta_df = ensure_data()
    print(f"Dane treningowe: {len(train_df)} próbek")
    print(f"Dane walidacyjne: {len(val_df)} próbek\n")
    
    print("=== klasyfikacja ===")
    test_classification_model(train_df, val_df, test_df, meta_df)
    
    # print("\n=== detekcja ===")
    # trained_detector = test_detection_model(train_df, val_df, test_df, meta_df)

if __name__ == "__main__":
    main()

    

   
