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

    def forward(self, x):
        image = cv2.imread(x)

        detections = self.detector.predict(
            source=image,
            conf=0.15,  # Trzeba dawać bardzo niskie. Należy wrócić do pomysłu z doszkalaniem modelu lub zmienić z YOLO
            verbose=False,
            device="cpu" if not torch.cuda.is_available() else "cuda",
            classes=[495, 549],  # Kolejno klasa anku stop i znaku drogowego
        )

        results = []

        for r in detections:
            crops = map_yolo_to_classifier(r, r.orig_img, self.target_size)

            for crop in crops:
                crop_batch = crop.unsqueeze(0).to(
                    next(self.classifier.parameters()).device
                )

                prediction = self.classifier(crop_batch)
                results.append(prediction)

        return results
