import torch
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, test_loader, criterion, is_classification=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
            
            if is_classification:
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    get_confusion_matrix(all_labels, all_preds)
    print_classification_report(all_labels, all_preds)



def get_confusion_matrix(y_true, y_pred, print_on_console=True, save_as_plot=False):
    cm = confusion_matrix(y_true, y_pred)
    if print_on_console:
        print(cm)
    if save_as_plot:
        pass
    
def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

