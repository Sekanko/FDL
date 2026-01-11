from sklearn.model_selection import train_test_split
from . import fetchers
from . import german_helpers
from mappers.normalize import to_german_standard
from mappers import map_classes
from mappers.img_format import map_ppm_to_png

def german_data_as_df():
    path = fetchers.download_german_dataset()
    train_df, test_df, meta_df = german_helpers.load_csv_files(path)
    train_df, test_df, meta_df = [
        german_helpers.convert_df_img_paths_to_absolute_paths(path, df)
        for df in [train_df, test_df, meta_df]
    ]
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=50, stratify=train_df["ClassId"]
    )
    return train_split, val_split, test_df

def polish_data_as_df():
    path = fetchers.download_polish_dataset()
    df = to_german_standard(path, map_classes.get_polish_mapping(), "classification")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=50, stratify=df["ClassId"])
    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=50, stratify=train_df["ClassId"])
    return train_split, val_split, test_df

def belgium_data_as_df():
    path = fetchers.download_belgium_dataset()
    data = map_ppm_to_png(path)
    train_df = to_german_standard(data, map_classes.get_belgium_mapping(), "Training")
    test_df = to_german_standard(data, map_classes.get_belgium_mapping(), "Testing")
    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=50, stratify=train_df["ClassId"])
    return train_split, val_split, test_df