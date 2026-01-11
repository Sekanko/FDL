import torch
import matplotlib.pyplot as plt


def train_model(model, dataloader, val_loader, criterion, optimizer, num_epochs, model_name="model"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_metric = 0.0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_metric += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        epoch_score = running_metric.double() / len(dataloader.dataset)

        model.eval()
        val_loss = 0.0
        val_metrics = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                v_loss = criterion(outputs, labels)

                val_loss += v_loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                val_metrics += torch.sum(preds == labels.data)

        v_epoch_loss = val_loss / len(val_loader.dataset)
        v_epoch_score = val_metrics.double() / len(val_loader.dataset)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_score.item())
        val_losses.append(v_epoch_loss)
        val_accs.append(v_epoch_score.item())

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} {'Acc'}: {epoch_score:.4f}")
        print(f"Val   Loss: {v_epoch_loss:.4f} {'Acc'}: {v_epoch_score:.4f}")
        print("-" * 25)

    visualize_metrics(train_losses, train_accs, val_losses, val_accs, model_name)

    return model


def visualize_metrics(train_losses, train_accs, val_losses, val_accs, model_name):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 6))

    # --- Wykres 1: Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title(f'Accuracy: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Wykres 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'Loss: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{model_name}_loss_acc.png"
    plt.savefig(filename)
    plt.close()
