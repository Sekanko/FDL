import torch
from train.train_model import train_model
from data.gtsrb_dataset import ensure_data
from data.SignDataset import create_dataloaders
from models.linear_model import TrafficSignClassifierLinearModel
from models.convolutional_model import TrafficSignClassifierConvolutionalModel
from models.resnet_model import get_resnet_model
# Póki co main do testów czy wszytsko działa poprawnie

#torch.cuda.is_available = lambda: False

if __name__ == "__main__":
    print("Testowanie modelu liniowego\n")
    
    print("Pobieranie danych...")
    train_df, val_df, test_df, meta_df = ensure_data()
    print(f"Dane treningowe: {len(train_df)} próbek")
    print(f"Dane walidacyjne: {len(val_df)} próbek\n")

    print("Tworzenie dataloaderów...")
    train_loader = create_dataloaders(train_df, batch_size=32, size=(224,224))
    val_loader = create_dataloaders(val_df, batch_size=32, size=(224,224))
    print(f"Ilość batchy treningowych: {len(train_loader)}\n")
    
    print("Inicjalizacja modelu liniowego...")
    # model = TrafficSignClassifierLinearModel()
    # model = TrafficSignClassifierConvolutionalModel()
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
        num_epochs=5
    )
    
    print("\n=== Trening zakończony ===")
    print("Model został pomyślnie wytrenowany!")
