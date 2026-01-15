import cv2
from torchvision import transforms


def map_yolo_to_classifier(result, photo, target_size=(224, 224)):
    image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size), 
            transforms.ToTensor()
    ])

    crops = []
    
    for box in result.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)

        crop = photo[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        

        tensor_crop = image_transform(crop_rgb)
        crops.append(tensor_crop)

    return crops
