import gurobipy as gp
from gurobipy import GRB
import time 
import pandas as pd
import sys
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
        df = pd.read_csv(file_path, delimiter=',')
        #df = df.dropna()  # Optional: Drop rows with missing values
        print(df.dtypes)
        df['Total_MWe_Mean'] = pd.to_numeric(df['Total_MWe_Mean'], errors='coerce')
        df['NetUndevelopedRP'] = pd.to_numeric(df['NetUndevelopedRP'], errors='coerce')
        df['Acres_GeothermalField'] = pd.to_numeric(df['Acres_GeothermalField'], errors='coerce')

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
        
    Returns:
        dict: Dictionary of binary selectors, where each key is a condition and each value is a binary vector.
    """
    import pandas as pd

def define_selectors(data):
    """
    Define selectors as binary vectors based on dataset attributes. Selectors are used
    to determine which data points satisfy specific conditions related to subgroup characteristics.
    
    Parameters:
        data (pd.DataFrame): The preprocessed DataFrame.
        
    Returns:
        dict: Dictionary of binary selectors, where each key is a condition and each value is a binary vector.
    """

    data['Total_MWe_Mean'].fillna(0, inplace=True)
    data['NetUndevelopedRP'].fillna(0, inplace=True)
    data['Acres_GeothermalField'].fillna(0, inplace=True)

    # Define thresholds based on data distribution (e.g., median or quantile-based for flexibility)
    THRESHOLD_HIGH_CAPACITY = data['Total_MWe_Mean'].quantile(0.75)  # High capacity as the upper quartile
    THRESHOLD_LOW_CAPACITY = data['Total_MWe_Mean'].quantile(0.25)   # Low capacity as the lower quartile

    THRESHOLD_HIGH_UNDEVELOPED = data['NetUndevelopedRP'].quantile(0.75)  # High potential in upper quartile
    THRESHOLD_LOW_UNDEVELOPED = data['NetUndevelopedRP'].quantile(0.25)   # Low potential in lower quartile

    THRESHOLD_HIGH_ACRES = data['Acres_GeothermalField'].quantile(0.75)   # High acreage for upper quartile
    THRESHOLD_LOW_ACRES = data['Acres_GeothermalField'].quantile(0.25)    # Low acreage for lower quartile

    # Initialize selectors dictionary
    selectors = {}

    # Define selectors based on calculated thresholds
    selectors["High_Capacity"] = (data["Total_MWe_Mean"] > THRESHOLD_HIGH_CAPACITY).astype(int)
    selectors["Low_Capacity"] = (data["Total_MWe_Mean"] < THRESHOLD_LOW_CAPACITY).astype(int)

    selectors["High_UndevelopedRP"] = (data["NetUndevelopedRP"] > THRESHOLD_HIGH_UNDEVELOPED).astype(int)
    selectors["Low_UndevelopedRP"] = (data["NetUndevelopedRP"] < THRESHOLD_LOW_UNDEVELOPED).astype(int)

    selectors["High_Acres"] = (data["Acres_GeothermalField"] > THRESHOLD_HIGH_ACRES).astype(int)
    selectors["Low_Acres"] = (data["Acres_GeothermalField"] < THRESHOLD_LOW_ACRES).astype(int)

    # Display selector thresholds and sample values for validation
    print(f"Thresholds:\n High Capacity: {THRESHOLD_HIGH_CAPACITY}, Low Capacity: {THRESHOLD_LOW_CAPACITY}")
    print(f"High Undeveloped RP: {THRESHOLD_HIGH_UNDEVELOPED}, Low Undeveloped RP: {THRESHOLD_LOW_UNDEVELOPED}")
    print(f"High Acres: {THRESHOLD_HIGH_ACRES}, Low Acres: {THRESHOLD_LOW_ACRES}")

    # Display results of the selectors
    for selector_name, selector_values in selectors.items():
        print(f"{selector_name}: {selector_values.tolist()[:10]}...")  # Display first 10 for brevity

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
    
    try: 
        positives_dataset = (data['Total_MWe_Mean'] > 500).astype(int)
        target_share_dataset = positives_dataset.sum() / n_cases
        SD_size = gp.quicksum(T[c] for c in range(n_cases))
        SD_positives = gp.quicksum(positives_dataset[c] * T[c] for c in range(n_cases))
        model.addConstr(PosRatio * SD_size == SD_positives, "PosRatioConstraint")
        Q_SD = (SD_size / n_cases) * (PosRatio - target_share_dataset)
        model.setObjective(Q_SD, GRB.MAXIMIZE)

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
    """
    Load data, set up and run the Gurobi optimization, and measure the runtime.
    
    Parameters:
        file_path (str): Path to the CSV file.
    """
    print("Made it here")
    
    # Record start time
    start_time = time.time()
    
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
    
    # Run optimization and measure runtime
    run_optimization(model)
    
    # Record end time and calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Optimization runtime: {runtime:.2f} seconds")

    # Continue with analysis and visualization
    high_cases = data[(data['Total_MWe_Mean'] > 100) & (data['NetUndevelopedRP'] > 50)]
    print(data[['Total_MWe_Mean', 'NetUndevelopedRP']].describe())
    
    print("Subgroup with high 'Total_MWe_Mean' and 'NetUndevelopedRP':")
    print(high_cases)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data['Total_MWe_Mean'], color='gray', alpha=0.5, label='All Data')
    plt.scatter(high_cases.index, high_cases['Total_MWe_Mean'], color='blue', label='High Cases')
    plt.title('Total MWe Mean vs. Entry Index (High Cases)')
    plt.xlabel('Entry Index')
    plt.ylabel('Total MWe Mean')
    plt.legend()
    plt.show()
    
    return runtime  # Return runtime if needed

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/geothermal.csv" 
    main(file_path)  # Pass the file_path argument to main
    


#if model.status==3:
 #       model.computeIIS()
 #       model.write("infeasible_model.ilp")

 #https://www.gurobi.com/documentation/current/refman/py_model_addconstrs.html
 #https://www.gurobi.com/documentation/current/refman/py_quicksum.html