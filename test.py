import pandas as pd

# Lade die CSV-Datei
file_path = 'data/cleaned_data/cleaned_data.csv'
data = pd.read_csv(file_path)

# Gebe die ersten 5 Zeilen aus
print("Erste 5 Zeilen des Datensatzes:")
print(data.head())

# Gebe Informationen zu den Spalten aus
print("\nSpalteninformationen:")
print(data.info())