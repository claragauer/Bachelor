import pysubgroup as ps
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_models import measure_memory_usage
import time

# Konfiguration für Subgruppenentdeckung
TARGET_VALUE = 1
RESULT_SET_SIZE = 10
SEARCH_DEPTH = 2
THRESHOLD_ABWASSER = 1e13  # Schwellenwert für "hohe Abwasserbelastung"
THRESHOLD_CASES = 50       # Schwellenwert für "hohe Fallzahlen"

def load_and_preprocess_data(file_path):
    """
    Daten laden und bereinigen.
    """
    try:
        df = pd.read_csv(file_path, delimiter=';', decimal='.')
        
        # Datum konvertieren
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        
        # Fehlende Werte auffüllen
        df['7d-Median SARS-CoV-2 Abwasser'] = pd.to_numeric(df['7d-Median SARS-CoV-2 Abwasser'], errors='coerce').fillna(0)
        df['7d-Median SARS-CoV-2-Fälle'] = pd.to_numeric(df['7d-Median SARS-CoV-2-Fälle'], errors='coerce').fillna(0)

        print("Daten nach der Bereinigung:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Fehler bei der Datenvorverarbeitung: {e}")
        raise

def define_target(df):
    """
    Zielvariable für Subgruppenentdeckung definieren.
    """
    # Definieren von "hoher Abwasserbelastung" oder "hoher Fallzahl" als Zielvariable
    df['High_Abwasser_or_Cases'] = ((df['7d-Median SARS-CoV-2 Abwasser'] > THRESHOLD_ABWASSER) |
                                    (df['7d-Median SARS-CoV-2-Fälle'] > THRESHOLD_CASES)).astype(int)
    target = ps.BinaryTarget('High_Abwasser_or_Cases', TARGET_VALUE)
    return target

def create_search_space(df):
    """
    Suchraum für Subgruppenentdeckung basierend auf verfügbaren Spalten erstellen.
    """
    search_space = []

    # Selektoren für Abwasserbelastung und Fallzahlen in bestimmten Bereichen
    abwasser_ranges = [(0, 5e12), (5e12, 1e13), (1e13, 2e13), (2e13, 5e13)]
    search_space += [ps.IntervalSelector('7d-Median SARS-CoV-2 Abwasser', start, end) for start, end in abwasser_ranges]

    cases_ranges = [(0, 10), (10, 50), (50, 100), (100, 200)]
    search_space += [ps.IntervalSelector('7d-Median SARS-CoV-2-Fälle', start, end) for start, end in cases_ranges]

    return search_space

def run_subgroup_discovery(df, target, search_space):
    """
    Apriori-Algorithmus zur Subgruppenentdeckung ausführen.
    """
    quality_function = ps.WRAccQF()
    apriori = ps.Apriori()

    task = ps.SubgroupDiscoveryTask(
        df, 
        target, 
        search_space, 
        result_set_size=RESULT_SET_SIZE, 
        depth=SEARCH_DEPTH, 
        qf=quality_function
    )

    result = apriori.execute(task)
    result_df = result.to_dataframe()
    print(result_df)

    return result_df

def display_results(df, result_df):
    """
    Ergebnisse der Subgruppenentdeckung anzeigen und visualisieren.
    """
    for i, row in result_df.iterrows():
        print(f"Details für Subgruppe {i}:")
        print(row)
        print(f"Bedingungen der Subgruppe: {row['subgroup']}")
        print("-" * 40)

        subgroup_condition = row['subgroup']
        positive_cases = df[subgroup_condition.covers(df)]

        # Darstellung der "7d-Median SARS-CoV-2 Abwasser" für alle Daten und die positive Subgruppe
        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df['7d-Median SARS-CoV-2 Abwasser'], color='gray', alpha=0.5, label='Alle Daten')
        plt.scatter(positive_cases.index, positive_cases['7d-Median SARS-CoV-2 Abwasser'], color='blue', label='Hohe Abwasser-Subgruppe')
        plt.title(f'Subgruppe {i}: Abwasserbelastung nach Eintrag')
        plt.xlabel('Eintragsindex')
        plt.ylabel('7d-Median SARS-CoV-2 Abwasser')
        plt.legend()
        plt.show()

def main(file_path):
    """
    Hauptfunktion, um den PySubgroup-Algorithmus auf die CSV-Datei anzuwenden.
    """
    try:
        def run_pipeline(file_path):
            start_time = time.time()

            df = load_and_preprocess_data(file_path)
            target = define_target(df)
            search_space = create_search_space(df)
            result_df = run_subgroup_discovery(df, target, search_space)

            # Ergebnisse anzeigen und visualisieren
            display_results(df, result_df)

            end_time = time.time()
            print(f"Gesamtlaufzeit: {end_time - start_time:.2f} Sekunden")

        _, peak_memory = measure_memory_usage(run_pipeline, file_path)
        print(f"Maximaler Speicherverbrauch während der Pipeline: {peak_memory:.2f} MB")

    except Exception as e:
        print(f"Ein Fehler trat während der Hauptausführung auf: {e}")

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/covid19wasteWater.csv"
    main(file_path)
