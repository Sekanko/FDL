import cv2
import torch
import torchvision.transforms as T
from mappers.map_classes import get_classes_to_names


def torch_prediction(model, model_registry, img, display=True):
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

    if display:
        prediction_display([prediction])

    return prediction

def prediction_display(class_ids):
    class_map = get_classes_to_names()

    for id in class_ids:
        print(f"\n Classified sign \"{class_map[id]}\" with id: {id} \n")

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

    prediction_display(class_ids)

    return class_ids
