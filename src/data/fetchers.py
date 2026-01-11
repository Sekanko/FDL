import kagglehub

def download_german_dataset():
    dataset_path = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    )
    return dataset_path

def download_polish_dataset():
    dataset_path = kagglehub.dataset_download(
        "chriskjm/polish-traffic-signs-dataset"
        )
    return dataset_path

def download_belgium_dataset():
    dataset_path = kagglehub.dataset_download(
        "mahadevkonar/belgiumts-dataset"
        )
    return dataset_path