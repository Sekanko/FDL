from ultralytics import YOLO


def load_yolo_model():
    model = YOLO("yolo11n.pt")
    return model