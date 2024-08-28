import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scripts.preprocess_data import load_data, handle_missing_values, handle_outliers, encode_categorical_columns, balance_data

# Constants for constraints to avoid using floating-point numbers directly in the code
THETA_DC = 2           # Maximum number of selectors allowed
THETA_CC = 2           # Minimum coverage required for the subgroup
THETA_MAX_RATIO = 0.5  # Maximum ratio of cases that can be included in the subgroup

def load_and_preprocess_data(file_path):
    """
    Load and preprocess dataset by handling missing values, encoding categorical variables, 
    balancing data, and handling outliers.
    
    Parameters:
        file_path (str): Path to the CSV file.
        categorical_columns (list): List of categorical columns to encode.
        target_column (str): Name of the target column for balancing.
        outlier_columns (list): List of columns to check for outliers.
        balance_method (str): Method for balancing data ('oversample' or 'undersample').
        outlier_method (str): Method to handle outliers ('z-score' or 'iqr').
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of LabelEncoders used for encoding categorical columns.
    """
    # Step 1: Load dataset
    df = load_data(file_path)

    # Step 2: Handle Missing Values
    #df = handle_missing_values(df, method='ffill')

    # Step 3: Encode Categorical Variables
    #df, label_encoders = encode_categorical_columns(df, categorical_columns)

    # Step 4: Handle Outliers
    #df = handle_outliers(df, outlier_columns, method=outlier_method)

    # Step 5: Balance Data
    #df = balance_data(df, target_column, method=balance_method)

    return df


def define_selectors(data):
    """
    Define selectors as binary vectors based on dataset attributes. Selectors are used
    to determine which data points satisfy specific conditions related to subgroup characteristics.
    
    Parameters:
        data (pd.DataFrame): The preprocessed DataFrame.
        
    Returns:
        dict: Dictionary of binary selectors, where each key is a condition and each value is a binary vector.
    """
    selectors = {
        'Color_Red': (data['Color'] == 'Red').astype(int),
        'Color_Blue': (data['Color'] == 'Blue').astype(int),
        'Shape_Circle': (data['Shape'] == 'Circle').astype(int),
        'Shape_Square': (data['Shape'] == 'Square').astype(int)
    }
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
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) <= THETA_DC)

    # Minimum number of cases in the subgroup
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= THETA_CC)

    # Maximum size of the subgroup (as a proportion of total cases)
    theta_max = int(THETA_MAX_RATIO * n_cases)
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) <= theta_max)

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
    # Step 1: Load and preprocess data using functions from preprocess_data.py
    data = load_and_preprocess_data(file_path)
    
    # Step 2: Define selectors
    selectors = define_selectors(data)
    n_cases = len(data)  # Number of cases
    
    # Step 3: Setup Gurobi model
    model, T, D, PosRatio = setup_model(n_cases, selectors)
    
    # Step 4: Set objective function and constraints
    set_objective(model, T, PosRatio, data, n_cases)
    
    # Step 5: Run optimization
    run_optimization(model)



if __name__ == "__main__":
    main()
