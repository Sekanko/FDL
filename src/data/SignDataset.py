import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler


class SignDataset(Dataset):
    def __init__(self, df, size, mode, transform):
        self.paths = df["Path"].values
        self.labels = df["ClassId"].values
        self.rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].values
        self.size = size
        self.mode = mode
        self.transform = transform
            
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = self.rois[idx]

        if self.mode == "detection":
            w, h = image.size
            target = torch.tensor([
                x1 / w,
                y1 / h, 
                x2 / w,
                y2 / h
            ], dtype=torch.float32)
            image_tensor = self.transform(image)
            return image_tensor, target

        elif self.mode == "classification":
            if x1 != -1:
                image = image.crop((x1, y1, x2, y2))
   
            image_tensor = self.transform(image)
            label = self.labels[idx]

            return image_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(data, batch_size=32, size=(32,32), mode="classification", use_aug=False):
    modes = ["classification", "detection"]
    if mode not in modes:
        raise ValueError(f"Mode {mode} not in {modes}")

    normalizator = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    base_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalizator
    ])

    aug_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ColorJitter(0.3, 0.3), 
        transforms.RandomAffine(30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalizator
    ])

    selected_transform = aug_transform if use_aug else base_transform
    dataset = SignDataset(data, size, mode, transform=selected_transform)

    if use_aug and mode == "classification":
        class_counts = data['ClassId'].value_counts().to_dict()
        weights = [1.0 / class_counts[c] for c in data['ClassId']]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

