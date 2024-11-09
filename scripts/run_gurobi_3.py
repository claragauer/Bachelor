import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import sys
import time
import os
import matplotlib.pyplot as plt
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'Bachelor' directory to the sys.path
sys.path.append(os.path.join(current_dir, 'scripts'))

from sklearn.preprocessing import LabelEncoder
from preprocess_data import load_data, handle_missing_values, handle_outliers, encode_categorical_columns, balance_data
from evaluate_models import measure_memory_usage

# Constants for constraints to avoid using floating-point numbers directly in the code
THETA_DC = 2           # Maximum number of selectors allowed
THETA_CC = 2           # Minimum coverage required for the subgroup
THETA_MAX_RATIO = 0.5  # Maximum ratio of cases that can be included in the subgroup

MAXIMUM_UNIQUE_VALUES = 1000 # Maximum number of unique numbers allowed in order to create selectors.

def load_and_preprocess_data(file_path):
    """
    Load and preprocess dataset by handling missing values and handling outliers.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of LabelEncoders used for encoding categorical columns.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()  # Optional: Drop rows with missing values
        print(df.dtypes)

        # Versuche, die mittlere Spalte in numerischen Typ zu konvertieren
        #df['7d-Median SARS-CoV-2 Abwasser'] = pd.to_numeric(df['7d-Median SARS-CoV-2 Abwasser'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def define_selectors(data):
    """
    Define selectors as binary vectors based on dataset attributes. Selectors are used
    to determine which data points satisfy specific conditions related to subgroup characteristics.
    
    Parameters:
        data (pd.DataFrame): The preprocessed DataFrame.
        conditions (dict): Dictionary with column names as keys and lists of values as conditions.
        
    Returns:
        dict: Dictionary of binary selectors, where each key is a condition and each value is a binary vector.
    """

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


def setup_model(n_cases, selectors):
    """
    Setup Gurobi model for subgroup discovery by defining decision variables and adding constraints.
    
    Parameters:
        n_cases (int): Number of cases in the dataset.
        selectors (dict): Dictionary of selectors indicating conditions to be considered.
        
    Returns:
        gp.Model: Configured Gurobi model.
        gp.tupledict: Decision variables T (binary vector indicating if a case is in the subgroup).
        gp.tupledict: Decision variables D (binary vector indicating if a selector is active).
        gp.Var: PosRatio (variable for the positive ratio in the subgroup).
    """
    # Create Gurobi model
    model = gp.Model("Subgroup_Discovery")
    model.setParam('OutputFlag', 1)

    # Decision Variables
    T = model.addVars(n_cases, vtype=GRB.BINARY, name="T")  # Binary vector indicating subgroup membership
    D = model.addVars(len(selectors), vtype=GRB.BINARY, name="D")  # Binary vector indicating active selectors
    PosRatio = model.addVar(name="PosRatio")  # Positive ratio in the subgroup

    # Constraints to relate T and D based on the dataset
    for i, (selector_name, selector_values) in enumerate(selectors.items()):
        for c in range(n_cases):
            # Ensures that if a selector is active, all matching cases must be in the subgroup
            model.addConstr(T[c] >= D[i] * selector_values.iloc[c]) # C1

    # Add complexity, coverage, and other constraints
    add_constraints(model, n_cases, selectors, T, D)

    return model, T, D, PosRatio

def add_constraints(model, n_cases, selectors, T, D):
    """
    Add constraints to the Gurobi model to enforce conditions like maximum selectors, 
    minimum and maximum subgroup sizes, and subgroup non-triviality.
    
    Parameters:
        model (gp.Model): Gurobi model.
        n_cases (int): Number of cases.
        selectors (dict): Dictionary of selectors.
        T (gp.tupledict): Decision variables T.
        D (gp.tupledict): Decision variables D.
    """
    # Maximum number of selectors used
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) <= THETA_DC, "MaximumNumberSelectors")

    # Minimum number of cases in the subgroup
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= THETA_CC, "MinimumCasesSubgroup")

    # Maximum size of the subgroup (as a proportion of total cases)
    theta_max = int(THETA_MAX_RATIO * n_cases)
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) <= theta_max, "MaximumSizeSubgroup")

    # Ensure subgroup is not empty
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= 1, "NonEmptySubset")

    # Ensure at least one selector is active in the subgroup description
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) >= 1, "AtLeastOneSelector")

    # Angleichen von 

def set_objective(model, T, PosRatio, data, n_cases):
    """
    Define the objective function for the Gurobi model to maximize the Weighted Relative Accuracy (WRAcc) of the subgroup.
    
    Parameters:
        model (gp.Model): Gurobi model.
        T (gp.tupledict): Decision variables T.
        PosRatio (gp.Var): Positive ratio variable.
        data (pd.DataFrame): Preprocessed DataFrame.
        n_cases (int): Number of cases.
    """
    # Calculate the proportion of positive cases in the dataset
    # NOTEX TO SELF: Der Typ hat ein Problem mit Leerzeichen, wird da immer ein Fehler herausgeben 
    #positives_dataset = (data['7d-Median SARS-CoV-2 Abwasser'] > 1e13).astype(int)
    #target_share_dataset = sum(positives_dataset) / n_cases
    # Introduce auxiliary variables for SD_size and SD_positives becuase the linear approximation needs variables 
    #SD_size = model.addVar(vtype=GRB.CONTINUOUS, name="SD_size")
    #SD_positives = model.addVar(vtype=GRB.CONTINUOUS, name="SD_positives")
    #PosRatio = model.addVar(vtype=GRB.CONTINUOUS, name="PosRatio")
    # Link the SD_size to the sum of selected cases
    #model.addConstr(SD_size == gp.quicksum(T[c] for c in range(n_cases)), "SD_size_Constraint")
    # Link SD_positives to the sum of positive cases in the subgroup
    #model.addConstr(SD_positives == gp.quicksum(positives_dataset[c] * T[c] for c in range(n_cases)), "SD_positives_Constraint")
    # Set PosRatio as the ratio of positives in the subgroup
    #model.addConstr(PosRatio * SD_size == SD_positives, "PosRatio_Definition")
    # Example in Gurobi using piecewise-linear approximation (PWL)
    #breakpoints = [1, 10, 20, 30, 33]  # Define breakpoints for SD_size based on your dataset
    #values = [0, 5, 10, 15, 33]        # Corresponding values for SD_positives at each breakpoint

    # Adding the PWL constraint for linearizing the product
    #model.addGenConstrPWL(SD_size, SD_positives, breakpoints, values, "PosRatioConstraint_PWL")

    # Objective function to maximize WRAcc
    #Q_SD = (SD_size / n_cases) * (PosRatio - target_share_dataset)
    #model.setObjective(Q_SD, GRB.MAXIMIZE)

    #QUADRATIC CONSTRAINT
    try: 
        positives = (data['Insgesamt'] > 5000).astype(int)
        n = len(data)
        positive_share = sum(positives) / n

        # Modell und Variablen
        model = gp.Model("WRAcc")
        T = model.addVars(n, vtype=GRB.BINARY)

        # Zielfunktion WRAcc
        wracc = (gp.quicksum(T[i] * positives[i] for i in range(n)) / n) - (T.sum() * positive_share / n)
        model.setObjective(wracc, GRB.MAXIMIZE)

        # Bedingung und Optimierung
        model.addConstr(T.sum() >= 1)
        model.optimize()

    except Exception as e:
        print(f"Error in set_objective (WRAcc): {e}")

def run_optimization(model):
    """
    Run the optimization process and display the results if an optimal solution is found.
    
    Parameters:
        model (gp.Model): Gurobi model.
    """
    model.optimize()
    model.write("presolved_model.lp")
    model.write("presolved_model.mps")
    model.write("model.lp")
    if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("infeasible_model.ilp")
    elif model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
    else:
        print(f"Optimization was unsuccessful. Gurobi status code: {model.status}")
            # Additional information in case of suboptimal or error status
        if model.status == GRB.INTERRUPTED:
            print("Optimization was interrupted.")
        elif model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded.")
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        elif model.status == GRB.CUTOFF:
            print("Objective cutoff limit reached.")


def main(file_path):
    """
    Main function to run the Gurobi optimization process on the given CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:
        def run_optimization_pipeline(file_path):
            start_time = time.time()
            print("Made it here")
            data = load_and_preprocess_data(file_path)
            print("After loading")
            print(data.columns)
            selectors = define_selectors(data)  # Define selectors based on the sampled data
            print("Selectors")
            n_cases = len(data)  # Get number of cases in the sampled data

            model, T, D, PosRatio = setup_model(n_cases, selectors)
            print("Set up model")
            set_objective(model, T, PosRatio, data, n_cases)
            print("Set objective")
            run_optimization(model)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time for the optimization pipeline: {elapsed_time:.2f} seconds")

            # Definieren der Subgruppe basierend auf hohen Werten in 'Insgesamt' und 'Langzeitarbeitslose'
            high_cases = data[(data['Insgesamt'] > 5000) & (data['Langzeitarbeitslose'] > 1600)]

            # Ausgabe der Subgruppe
            print("Subgroup with high 'Insgesamt' and 'Langzeitarbeitslose':")
            print(high_cases)

            # Visualisierung der Subgruppe
            plt.figure(figsize=(10, 6))

            # Scatterplot für alle Daten (grau)
            plt.scatter(data.index, data['Insgesamt'], color='gray', alpha=0.5, label='Alle Daten')

            # Scatterplot für die hohen Fälle (blau)
            plt.scatter(high_cases.index, high_cases['Insgesamt'], color='blue', label='High Cases')
            # Titel und Achsenbeschriftungen
            plt.title('Gesamtzahl vs. Eintragsindex (Hohe Fälle)')
            plt.xlabel('Eintragsindex')
            plt.ylabel('Gesamtzahl der Personen')

            # Legende
            plt.legend()

            # Plot anzeigen
            plt.show()


        # Measure memory usage during the complete pipeline
        _, peak_memory = measure_memory_usage(run_optimization_pipeline, file_path)

        print(f"The peak memory usage during the optimization for {file_path} was: {peak_memory:.2f} MB")
    except Exception as e:
        print(f"An error occurred during the optimization process: {e}")

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/unemployment.csv" 
    main(file_path)  # Pass the file_path argument to main
    


#if model.status==3:
 #       model.computeIIS()
 #       model.write("infeasible_model.ilp")

 #https://www.gurobi.com/documentation/current/refman/py_model_addconstrs.html
 #https://www.gurobi.com/documentation/current/refman/py_quicksum.html