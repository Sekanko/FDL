import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class SignDataset(Dataset):
    def __init__(self, df):
        self.paths = df["Path"].values
        self.labels = df["ClassId"].values
        self.rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        x1, y1, x2, y2 = self.rois[idx]
        
        image = image.crop((x1, y1, x2, y2))
        
        image_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        image_tensor = image_transform(image)
        label = self.labels[idx]

        return image_tensor, torch.tensor(label, dtype=torch.long)

def create_dataloaders(data, batch_size=32):
    dataset = SignDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader