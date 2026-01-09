# Informacje o sieciach
Wszystkie sieci dotyczą problemu wykrywania i rozpoznawania znaków drogowych. Postanowiliśmy zastosować podejście dwuetapowe, aby móc oddzielnie zmieniać parametry sieci neuronowych w celu optymalizacji pod konkretne zadanie, dlatego docelowo chcemy mieć oddzielne **klasyfikację** i **wykrywanie**.

Do każdego z tych celów chcemy mieć po trzy modele:
- Nasz liniowy
- Nasz konwolucyjną
- Dowolny istniejący, inny niż powyższe
