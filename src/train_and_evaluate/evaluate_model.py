import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader, criterion, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_name = model.__class__.__name__

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    get_confusion_matrix(all_labels, all_preds, classes)
    print_classification_report(all_labels, all_preds)


def get_confusion_matrix(y_true, y_pred, classes, print_on_console=True, save_as_plot=True, model_name="model"):
    cm = confusion_matrix(y_true, y_pred)
    if print_on_console:
        print(cm)
    if save_as_plot:
        fig, ax = plt.subplots(figsize=(15, 15))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap="Blues", ax=ax, values_format='d', xticks_rotation="vertical")
        ax.set_title("Macierz pomyłek")
        try:
            file_name = f"{model_name}_cf.png"
            plt.savefig(file_name, bbox_inches='tight', dpi=300)
            print(f"Saved confusion matrix as: {file_name}")
        except Exception as e:
            print(f"Failed to save confusion matrix")
        finally:
            plt.close(fig)




def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
