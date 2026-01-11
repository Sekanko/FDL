import os
from PIL import Image

def map_ppm_to_png(src_root_path, dst_root_path="png_dataset"):
    """
    Konwertuje wszystkie pliki .ppm w dataset i zapisuje je jako .png w nowym katalogu.
    Zachowuje strukturę katalogów.
    
    Args:
        src_root_path (str): ścieżka do pobranego datasetu (ppm)
        dst_root_path (str): folder, w którym zostaną zapisane PNG (domyślnie 'png_dataset')
    
    Returns:
        str: ścieżka do katalogu z PNG
    """
    for dirpath, dirnames, filenames in os.walk(src_root_path):
        relative_path = os.path.relpath(dirpath, src_root_path)
        target_dir = os.path.join(dst_root_path, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in filenames:
            if file.lower().endswith(".ppm"):
                src_file = os.path.join(dirpath, file)
                dst_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".png")
                img = Image.open(src_file)
                img.save(dst_file)
    
    return dst_root_path

if __name__ == "__main__":
    dataset_path = input("Podaj ścieżkę do datasetu PPM: ")
    output_path = map_ppm_to_png(dataset_path)
    print(f"Konwersja zakończona! PNG w: {output_path}")

