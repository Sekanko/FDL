from ultralytics import YOLO


def load_yolo_model():
    model = YOLO("yolov8n-oiv7.pt")
    return model