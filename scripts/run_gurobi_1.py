import pandas as pd
import gurobipy as gp
import numpy as np 
from gurobipy import GRB
import matplotlib.pyplot as plt

THETA_DC = 3            # Max. Anzahl Selektoren für mehr Spezifität
THETA_CC = 10           # Min. Anzahl von Fällen in der Subgruppe
THETA_MAX_RATIO = 0.05  # Max. Verhältnis der Fälle in der Subgruppe
MAXIMUM_UNIQUE_VALUES = 1000

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=";")
    df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True)
    df['Anzahl_Personen'] = pd.to_numeric(df['Anzahl_Personen'], errors='coerce')
    return df.dropna()

def define_selectors(data):
    selectors = {}  # Initialize an empty dictionary to store selectors

    for column in data.columns:
        # Proceed only if the column is numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            # Calculate 25% and 75% quantiles for the column
            threshold_low = data[column].quantile(0.25)
            threshold_high = data[column].quantile(0.75)
            
            # Create high and low selectors based on quantiles
            selectors[f"High_{column}"] = (data[column] > threshold_high).astype(int)
            selectors[f"Low_{column}"] = (data[column] < threshold_low).astype(int)
        else:
            print(f"Skipping column '{column}' because it is not numeric.")

    return selectors

def setup_model(data, selectors):
    model = gp.Model("WRAcc")
    T = model.addVars(len(data), vtype=GRB.BINARY, name="T")
    D = model.addVars(len(selectors), vtype=GRB.BINARY, name="D")
    model.update()
    # Dynamischen Schwellenwert auf Basis des oberen Quartils festlegen
    threshold = np.percentile(data['Anzahl_Personen'], 75)

    # Bedingung für positive Fälle aufstellen, z.B. Anzahl der Personen über dem Schwellenwert und spezifische Altersklasse
    positives = ((data['Anzahl_Personen'] > threshold) & (data['Altersklasse'].isin(['20 - 29', '30 - 39']))).astype(int).tolist()
    positive_share = sum(positives) / len(data)
    subgroup_size = T.sum()
    subgroup_positives = gp.quicksum(T[i] * positives[i] for i in range(len(data)))
    wracc = subgroup_positives / len(data) - subgroup_size * positive_share / len(data)
    model.setObjective(wracc, GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) <= THETA_DC)
    model.addConstr(subgroup_size >= THETA_CC)
    model.addConstr(subgroup_size <= int(THETA_MAX_RATIO * len(data)))
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) >= 1)
    
    for i, selector_values in enumerate(selectors.values()):
        for c in range(len(data)):
            model.addConstr(T[c] >= D[i] * selector_values.iloc[c])
    
    return model, T

def run_optimization(model, T, data):
    model.optimize()
    if model.status == GRB.OPTIMAL:
        selected_cases = [i for i in range(len(data)) if T[i].x > 0.5]
        print("WRAcc-Wert:", model.objVal)
        print("Anzahl ausgewählter Fälle:", len(selected_cases))
        print("Indizes der ausgewählten Fälle:", selected_cases)
        return data.iloc[selected_cases]
    else:
        print("Keine optimale Lösung gefunden.")
        return pd.DataFrame()

def plot_results(data, subgroup):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Datum'], data['Anzahl_Personen'], color='gray', alpha=0.5, label='All Data')
    plt.scatter(subgroup['Datum'], subgroup['Anzahl_Personen'], color='blue', label='High Anzahl_Personen')
    plt.title('Anzahl_Personen over Time')
    plt.xlabel('Date')
    plt.ylabel('Anzahl_Personen')
    plt.legend()
    plt.show()

def main(file_path):
    data = load_and_preprocess_data(file_path)
    selectors = define_selectors(data)
    model, T = setup_model(data, selectors)
    subgroup = run_optimization(model, T, data)
    if not subgroup.empty:
        plot_results(data, subgroup)

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/covid19casesByGender.csv"
    main(file_path)
