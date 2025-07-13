# Ballistic Trajectory Simulator

Projekt w Pythonie służący do symulacji lotu pocisku z uwzględnieniem oporu powietrza, warunków atmosferycznych i wyznaczania kąta strzału. Pozwala na:

- Dynamiczne obliczanie kąta wystrzału w celu trafienia na zadaną odległość
- Generowanie trajektorii z milimetrową dokładnością
- Uwzględnienie temperatury, ciśnienia i wilgotności przy wyliczaniu gęstości powietrza
- Interpolację pozycji pocisku dokładnie w punkcie celu oraz na 100 m (jeśli dystans > 100 m)
- Eksport trajektorii do pliku CSV z uporządkowanymi danymi czasowymi

---
## Wymagania

- Python
- NumPy
- Matplotlib

## Instalacja
1. Sklonuj repozytorium:
```bash
git clone https://github.com/Daldek/Proca.git
cd Proca
```
Instalacja zależności:
```bash
pip install numpy matplotlib
```

---
## Uruchomienie

```bash
python bullet_simulator.py
```

Po uruchomieniu użytkownik zostanie poproszony o dane wejściowe:

- Masa pocisku (w grainach)
- Prędkość początkowa (m/s)
- Odległość do celu (m)
- Wysokość celu (m)
- Temperatura (°C)
- Ciśnienie atmosferyczne (hPa)
- Wilgotność (%)

Wyniki:
- Wyliczony kąt strzału w konsoli
- Interpolowane trafienie w punkt celu
- Opcjonalna interpolacja dla odległości 100 m (jeśli dystans > 100 m)
- Plik `trajectory.csv` z pełną trajektorią i punktami interpolowanymi
- Wykres trajektorii z oznaczonym punktem celu

---
## Struktura CSV

| Time [s] | X [m]    | Y [m]    |
|----------|----------|----------|
| 0.000    | 0.000    | 0.000    |
| ...      | ...      | ...      |
| 0.1260   | 99.000   | 0.244    | ← interpolowane trafienie
| 0.1260   | 100.000  | 0.244    | ← interpolacja na 100 m (jeśli dotyczy)

---
## Wizualizacja

Trajektoria rysowana w matplotlib, z oznaczeniem celu jako czerwony „X”.

---
## Problemy i wsparcie
Jeśli napotkasz problemy, zgłoś je w sekcji [Issues](https://github.com/Daldek/Proca/issues).

---
## Licencja
Projekt jest udostępniony na liencji MIT. Szczegóły znajdziesz w pliku ``License``.

---
## Autorzy
- [Piotr de Bever](https://www.linkedin.com/in/piotr-de-bever/) [@LinkedIn](https://www.linkedin.com/in/piotr-de-bever/)