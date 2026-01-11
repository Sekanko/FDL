import kagglehub

def download_polish_ds():
    path = kagglehub.dataset_download("chriskjm/polish-traffic-signs-dataset")
    return path

if __name__=="__main__":
    download_polish_ds()