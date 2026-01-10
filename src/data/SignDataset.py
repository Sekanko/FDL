import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class SignDataset(Dataset):
    def __init__(self, df, size, mode):
        self.paths = df["Path"].values
        self.labels = df["ClassId"].values
        self.rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].values
        self.size = size
        self.mode = mode

        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
        
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
            image_tensor = self.image_transform(image)
            return image_tensor, target

        elif self.mode == "classification":
            if x1 == -1:
                pass
            else:
                image = image.crop((x1, y1, x2, y2))
            image_tensor = self.image_transform(image)
            label = self.labels[idx]

            return image_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(data, batch_size=32, size=(32,32), mode="classification"):
    modes = ["classification", "detection"]
    if mode not in modes:
        raise ValueError(f"Function create_dataloaders() does not contain mode {mode}. Must be: {modes}")
    dataset = SignDataset(data, size, mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
