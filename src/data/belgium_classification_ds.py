import kagglehub


def download_belgium_ds():
    dataset_path = kagglehub.dataset_download("mahadevkonar/belgiumts-dataset")
    return dataset_path
