import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scripts.preprocess_data import load_data, handle_missing_values, handle_outliers, encode_categorical_columns, balance_data

# Constants for constraints to avoid using floating-point numbers directly in the code
THETA_DC = 2           # Maximum number of selectors allowed
THETA_CC = 2           # Minimum coverage required for the subgroup
THETA_MAX_RATIO = 0.5  # Maximum ratio of cases that can be included in the subgroup

MAXIMUM_UNIQUE_VALUES = 5 # Maximum number of unique numbers allowed in order to create selectors.

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
        df = load_data(file_path)
        df = handle_missing_values(df)
        df = handle_outliers(df)
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

    selectors = {} # Initialize an empty dictionary to store selectors
    
    # Iterate over each column in the data frame to create selectors based on unique values
    for column in data.columns:
        unique_values = data[column].unique() # Get all unique values in the column 
        # Create selectors only for columns with a manageable number of unique values to avoid excessive computation.
        if len(unique_values) <= MAXIMUM_UNIQUE_VALUES:  
            # Iterate over each unique value to create a corresponding binary selector
            for value in unique_values:
                # Create unique selector name by combining the column and the value 
                selector_name = f"{column}_{value}"
                selectors[selector_name] = (data[column] == value).astype(int)
        else:
            print(f"Skipping column '{column}' due to too many unique values ({len(unique_values)}).")

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
            model.addConstr(T[c] >= D[i] * selector_values[c])

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
    positives_dataset = data['Label'].tolist()
    target_share_dataset = sum(positives_dataset) / n_cases

    # Define subgroup size and positive count within the subgroup
    SD_size = gp.quicksum(T[c] for c in range(n_cases))
    SD_positives = gp.quicksum(positives_dataset[c] * T[c] for c in range(n_cases))

    # Constraint to define the positive ratio in the subgroup
    model.addConstr(PosRatio * SD_size == SD_positives, "PosRatioConstraint")

    # Objective function to maximize WRAcc
    Q_SD = (SD_size / n_cases) * (PosRatio - target_share_dataset)
    model.setObjective(Q_SD, GRB.MAXIMIZE)

def run_optimization(model):
    """
    Run the optimization process and display the results if an optimal solution is found.
    
    Parameters:
        model (gp.Model): Gurobi model.
    """
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
    else:
        print("No optimal solution found.")


def main(file_path):
    """
    Main function to run the Gurobi optimization process on the given CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:
        data = load_and_preprocess_data(file_path)
        selectors = define_selectors(data)
        n_cases = len(data)
        model, T, D, PosRatio = setup_model(n_cases, selectors)
        set_objective(model, T, PosRatio, data, n_cases)
        run_optimization(model)
    except Exception as e:
        print(f"An error occurred during the optimization process: {e}")



if __name__ == "__main__":
    main()
