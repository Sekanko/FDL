import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

def download_gtsb_dataset(): 
    """ Pobiera dataset, jesli aktualnie nie ma go w kagglehub cache.
        Zwraca sciezke do folderu z datasetem.
    """
    dataset_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    return dataset_path

def load_csv_files(dataset_dir):
    """ Laduje pliki csv i zwraca dataframe """

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
    """ Funkcja pomocnicza, zamienia sciezki do obrazow na sciezki bezwzgledne.
    """

    df['Path'] = df['Path'].apply(
        lambda row:
        os.path.join(dataset_path, row)
    )
    return df

def ensure_data():
    path = download_gtsb_dataset()
    train_df, test_df, meta_df = load_csv_files(path)
    train_df, test_df, meta_df = [convert_df_img_paths_to_absolute_paths(path, df) for df in [train_df, test_df, meta_df]]
    train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=50, stratify=train_df['ClassId'])
    return train_split, val_split, test_df, meta_df
        
def main():
    train, val, test, meta = ensure_data()
    print(train.head(5))
    

if __name__=="__main__":
    main()