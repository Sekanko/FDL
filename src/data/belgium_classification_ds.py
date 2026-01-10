import kagglehub
import os
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/damian/FDL/src/")
from mappers.map_classes import get_belgium_mapping
from tqdm import tqdm
from PIL import Image

def download_belgium_ds():
    dataset_path = kagglehub.dataset_download("mahadevkonar/belgiumts-dataset")
    return dataset_path

def find_target_folder(base_path, target_name):
    if os.path.isdir(os.path.join(base_path, target_name)):
        return os.path.join(base_path, target_name)

    for root, dirs, files in os.walk(base_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
            
    raise FileNotFoundError(f"Nie znaleziono folderu końcowego '{target_name}' wewnątrz {base_path}")

def map_to_german_standard_df(path, target_data="Training"):
    mapping = get_belgium_mapping()
    
    path = find_target_folder(path, target_data)

    data = []
    
    class_folders = sorted(os.listdir(path))
    
    for name in tqdm(class_folders, desc=f"Mapowanie {target_data}"):
        folder_path = os.path.join(path, name)
        
        if not os.path.isdir(folder_path):
            continue
            
        try:
            belgium_id = int(name)
        except ValueError:
            continue
            
        if belgium_id in mapping:
            german_id = mapping[belgium_id]
            
            for file in os.listdir(folder_path):
                if file.endswith(('.ppm', '.png')):
                    full_path = os.path.join(folder_path, file)

                    with Image.open(full_path) as img:
                        w, h = img.size
                    
                    data.append({
                        "Width": w,
                        "Height": h,
                        "Roi.X1": -1,
                        "Roi.Y1": -1,
                        "Roi.X1": -1,
                        "Roi.Y2": -1,
                        "ClassId": german_id,
                        "Path": full_path
                    })
    
    return pd.DataFrame(data)

def main():
    path = download_belgium_ds()
    df = map_to_german_standard_df(path)
    print(df)
    

if __name__=="__main__":
    main()
