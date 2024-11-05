import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from preprocess_data import load_data, handle_missing_values, handle_outliers, encode_categorical_columns, balance_data
from evaluate_models import measure_memory_usage

# Constants for constraints to avoid using floating-point numbers directly in the code
THETA_DC = 3            # Allow up to 3 selectors for more specificity
THETA_CC = 10           # Require at least 10 cases for the subgroup
THETA_MAX_RATIO = 0.05 # Maximum ratio of cases that can be included in the subgroup

MAXIMUM_UNIQUE_VALUES = 1000 # Maximum number of unique numbers allowed in order to create selectors.

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=";")
        df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True)  # Convert to datetime
        df['Anzahl_Personen'] = pd.to_numeric(df['Anzahl_Personen'], errors='coerce')  # Ensure numeric
        df.dropna(inplace=True)  # Drop rows with missing values if any
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

def define_selectors(data):
    selectors = {}

    # Highly specific combined conditions
    selectors["Altersklasse_30_39_Geschlecht_FE_Anzahl_Personen_50+"] = (
        (data['Altersklasse'] == '30 - 39') & (data['Geschlecht'] == 'FE') & (data['Anzahl_Personen'] > 50)
    ).astype(int)
    selectors["Altersklasse_40_49_Geschlecht_MA_Anzahl_Personen_50+"] = (
        (data['Altersklasse'] == '40 - 49') & (data['Geschlecht'] == 'MA') & (data['Anzahl_Personen'] > 50)
    ).astype(int)
    selectors["Altersklasse_50_59_Geschlecht_FE_Anzahl_Personen_50+"] = (
        (data['Altersklasse'] == '50 - 59') & (data['Geschlecht'] == 'FE') & (data['Anzahl_Personen'] > 50)
    ).astype(int)

    return selectors

def setup_model(n_cases, selectors):
    model = gp.Model("Subgroup_Discovery")
    model.setParam('OutputFlag', 1)

    T = model.addVars(n_cases, vtype=GRB.BINARY, name="T")
    D = model.addVars(len(selectors), vtype=GRB.BINARY, name="D")
    PosRatio = model.addVar(vtype=GRB.CONTINUOUS, name="PosRatio")
    model.update()

    for i, (selector_name, selector_values) in enumerate(selectors.items()):
        for c in range(n_cases):
            model.addConstr(T[c] >= D[i] * selector_values.iloc[c]) 

    add_constraints(model, n_cases, selectors, T, D)

    return model, T, D, PosRatio

def add_constraints(model, n_cases, selectors, T, D):
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) <= THETA_DC, "MaximumNumberSelectors")
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= THETA_CC, "MinimumCasesSubgroup")
    theta_max = int(THETA_MAX_RATIO * n_cases)
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) <= theta_max, "MaximumSizeSubgroup")
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= 1, "NonEmptySubset")
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) >= 1, "AtLeastOneSelector")

def set_objective(model, T, PosRatio, data, n_cases):
    try:
        model.update()
        
        # Calculate the target proportion of positives in the entire dataset
        positives_dataset = (data['Anzahl_Personen'] > 100).astype(int)
        target_share_dataset = positives_dataset.sum() / n_cases

        # Auxiliary variables for subgroup size and positive cases in subgroup
        SD_size = model.addVar(name="SD_size", vtype=GRB.CONTINUOUS)
        SD_positives = model.addVar(name="SD_positives", vtype=GRB.CONTINUOUS)

        # Define SD_size as the sum of T[c] to represent the subgroup size
        model.addConstr(SD_size == gp.quicksum(T[c] for c in range(n_cases)), "SD_size_Definition")

        # Define SD_positives as the sum of positive cases in the subgroup
        model.addConstr(SD_positives == gp.quicksum(positives_dataset.iloc[c] * T[c] for c in range(n_cases)), "SD_positives_Definition")

        # Define PosRatio as the proportion of positives in the subgroup, avoiding direct division
        model.addConstr(PosRatio * SD_size == SD_positives, "PosRatio_Definition")
        model.addConstr(SD_size <= int(0.1 * n_cases), "MaxSubgroupSize")  # Limiting subgroup size to 10% of total cases

        # Reformulate WRAcc as an objective without division by SD_size
        Q_SD = (SD_positives - target_share_dataset * SD_size) / n_cases
        model.setObjective(Q_SD, GRB.MAXIMIZE)
    except Exception as e:
        print(f"Error in set_objective (WRAcc): {e}")

def run_optimization(model):
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible_model.ilp")
    elif model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
    else:
        print(f"Optimization was unsuccessful. Gurobi status code: {model.status}")

def main(file_path):
    try:
        def run_optimization_pipeline(file_path):
            print("Starting optimization pipeline.")
            data = load_and_preprocess_data(file_path)
            selectors = define_selectors(data)
            n_cases = len(data)

            model, T, D, PosRatio = setup_model(n_cases, selectors)
            set_objective(model, T, PosRatio, data, n_cases)
            run_optimization(model)

            # Define subgroup
            high_cases = data[data['Anzahl_Personen'] > 2]

            # Print subgroup
            print("Subgroup with high number of people:")
            print(high_cases)

            # Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(data['Datum'], data['Anzahl_Personen'], color='gray', alpha=0.5, label='All Data')
            plt.scatter(high_cases['Datum'], high_cases['Anzahl_Personen'], color='blue', label='High Anzahl_Personen')
            plt.title('Anzahl_Personen over Time')
            plt.xlabel('Date')
            plt.ylabel('Anzahl_Personen')
            plt.legend()
            plt.show()

        _, peak_memory = measure_memory_usage(run_optimization_pipeline, file_path)
        print(f"The peak memory usage during the optimization for {file_path} was: {peak_memory:.2f} MB")
    except Exception as e:
        print(f"An error occurred during the optimization process: {e}")

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/covid19casesByGender.csv" 
    main(file_path)
