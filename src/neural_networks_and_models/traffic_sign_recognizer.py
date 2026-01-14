import torch
import torch.nn as nn
from mappers.map_yolo_to_classifire import map_yolo_to_classifier
import cv2


class TrafficSignRecognizer(nn.Module):
    def __init__(self, detector, classifier, target_size=(32, 32)):
        super(TrafficSignRecognizer, self).__init__()
        self.detector = detector
        self.classifier = classifier
        self.target_size = target_size

    def forward(self, image, conf=0.7, classes=None):
        detections = self.detector.predict(
            source=image,
            conf=conf,
            verbose=False,
            device=next(self.parameters()).device,
            classes=classes, 
        )

        results = []

        for r in detections:
            crops = map_yolo_to_classifier(r, r.orig_img, self.target_size)

            for crop in crops:
                crop_batch = crop.unsqueeze(0).to(
                    next(self.classifier.parameters()).device
                )

                self.classifier.eval() 
                with torch.no_grad():
                    logits = self.classifier(crop_batch)
                    prediction = torch.argmax(logits, dim=1).item()
                
                results.append(prediction)

        return results
