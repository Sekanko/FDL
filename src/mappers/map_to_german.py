import os
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/damian/FDL/src/")
from mappers.map_classes import get_belgium_mapping, get_polish_mapping
from tqdm import tqdm
from PIL import Image

import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

def find_target_folder(base_path, target_name):
    """
    Rekurencyjnie szuka folderu o nazwie target_name (np. 'Training').
    """
    if os.path.isdir(os.path.join(base_path, target_name)):
        return os.path.join(base_path, target_name)

    for root, dirs, files in os.walk(base_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
            
    raise FileNotFoundError(f"Nie znaleziono folderu '{target_name}' wewnątrz {base_path}")

def map_dataset_to_german_standard(path, mapping_dict, subset_name="Training"):
    """
    Uniwersalna funkcja mapująca dowolny dataset do formatu GTSRB.
    
    Args:
        path (str): Ścieżka do datasetu.
        mapping_dict (dict): Słownik { 'OryginalnaNazwa': GermanID }.
        subset_name (str): Nazwa podfolderu z ktorego wyciagamy dane.
    """

    real_path = find_target_folder(path, subset_name)

    data = []
    class_folders = sorted(os.listdir(real_path))
    
    for folder_name in tqdm(class_folders, desc=f"Mapowanie {subset_name}"):
        folder_full_path = os.path.join(real_path, folder_name)
        
        if not os.path.isdir(folder_full_path):
            continue

        german_id = None

        if folder_name in mapping_dict:
            german_id = mapping_dict[folder_name]

        else:
            try:
                folder_id_int = int(folder_name)
                if folder_id_int in mapping_dict:
                    german_id = mapping_dict[folder_id_int]
            except ValueError:
                pass 

        if german_id is None:
            continue

        for file in os.listdir(folder_full_path):
            if file.lower().endswith(('.ppm', '.png', '.jpg')):
                img_path = os.path.join(folder_full_path, file)

                with Image.open(img_path) as img:
                        w, h = img.size
                
                data.append({
                    "Width": w,
                    "Height": h,
                    "Roi.X1": -1,
                    "Roi.Y1": -1,
                    "Roi.X2": -1,
                    "Roi.Y2": -1,
                    "ClassId": german_id,
                    "Path": img_path,
                })

    
    return pd.DataFrame(data)
