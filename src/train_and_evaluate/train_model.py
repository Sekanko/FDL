import torch


def train_model(
    model, dataloader, val_loader, criterion, optimizer, num_epochs, is_classification
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

            if is_classification:
                _, preds = torch.max(outputs, 1)
                running_metric += torch.sum(preds == labels.data)
            else:
                running_metric += torch.sum(torch.abs(outputs - labels.data))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        if is_classification:
            epoch_score = running_metric.double() / len(dataloader.dataset)
        else:
            epoch_score = running_metric / (len(dataloader.dataset) * 4)

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

                if is_classification:
                    _, preds = torch.max(outputs, 1)
                    val_metrics += torch.sum(preds == labels.data)
                else:
                    val_metrics += torch.sum(torch.abs(outputs - labels.data))

        v_epoch_loss = val_loss / len(val_loader.dataset)
        if is_classification:
            v_epoch_score = val_metrics.double() / len(val_loader.dataset)
        else:
            v_epoch_score = val_metrics / (len(val_loader.dataset) * 4)

        score_name = "Acc" if is_classification else "MAE"

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} {score_name}: {epoch_score:.4f}")
        print(f"Val   Loss: {v_epoch_loss:.4f} {score_name}: {v_epoch_score:.4f}")
        print("-" * 25)

    return model
