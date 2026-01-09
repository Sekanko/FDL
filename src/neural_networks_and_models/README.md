# Informacje o sieciach

Docelowo chcemy mieć modele do dwóch celów:
- Klasyfikacji znaków drogowych
- Wykrywaniu na zdjęciach znaków drogowych

Do każdego z nich chcemy mieć po 3 wersje:
- Naszą liniową
- Naszą konwolucyjną
- Istniejącą dowolną inną niż powyższe

Nasze sieci mają oddzielne implemntacje dla klasyfikacji i wykrywania, aby móc oddzielnie zmieniać ich parametry w celu optymalizacji pod konkretne zadanie (podejście dwuetapowe).