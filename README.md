# Detekcja i klasyfikacja znaków drogowych
Pojekt przygotowywany w ramach przedmiotu FDL (Future of Deep Learning)
## O projekcie

Celem projektu jest stworzenie modelu, który potrafi poprawnie zaklasyfikować zdjęcie znaku drogowego do jednej z 43 kategorii (zgodnych ze standardem GTSRB). Projekt bada wpływ łączenia różnych zbiorów danych oraz porównuje skuteczność własnych architektur CNN z gotowymi, bardziej zaawansowanymi rozwiązaniami.

### Główne funkcjonalności:
* **Integracja danych:** Pobieranie, pre-processing i opcjonalne łączenie zbiorów danych z Niemiec (GTSRB), Polski i Belgii.
* **Architektury:** Implementacja i porównanie kilku modeli:
    * Prosta sieć liniowa.
    * Własna Konwolucyjna Sieć Neuronowa.
    * MobileNetV2 (Fine Tuning).
* **Wizualizacja:** Automatyczne generowanie wykresów uczenia (Loss/Accuracy) oraz macierzy pomyłek (Confusion Matrix).
* **Ewaluacja:** Szczegółowe raporty klasyfikacji (Precision, Recall, F1-Score).
  
## Uruchamianie
1. Upewnij się, że masz pobranego [pythona](https://www.python.org/downloads/) (wersja 3.10+)
2. Sklonuj repozytorium za pomocą `git clone git@github.com:Sekanko/FDL.git`
3. W konsoli przjedź do folderu z programem `cd fdl`
4. Przygotuj środowisko python:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
5. Po zakończeczniu instalacji wykonaj komendę `python src/main.py` i podążaj za instrukcjami w programie
