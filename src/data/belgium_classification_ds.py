import kagglehub
import os

def download_belgium_ds():
    dataset_path = kagglehub.dataset_download("mahadevkonar/belgiumts-dataset")
    return dataset_path

def main():
    path = download_belgium_ds()
    map_ppm_to_png(path)

if __name__=="__main__":
    main()

