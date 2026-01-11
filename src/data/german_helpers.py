import os
from .fetchers import download_german_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_files(dataset_dir):
    """Laduje pliki csv i zwraca dataframe"""

    train_csv = os.path.join(dataset_dir, "Train.csv")
    test_csv = os.path.join(dataset_dir, "Test.csv")
    meta_csv = os.path.join(dataset_dir, "Meta.csv")

    for path in [train_csv, test_csv, meta_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku {path}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    meta_df = pd.read_csv(meta_csv)

    return train_df, test_df, meta_df


def convert_df_img_paths_to_absolute_paths(dataset_path, df):
    """Funkcja pomocnicza, zamienia sciezki do obrazow na sciezki bezwzgledne."""

    df["Path"] = df["Path"].apply(lambda row: os.path.join(dataset_path, row))
    return df
