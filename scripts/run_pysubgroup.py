import pysubgroup as ps
import pandas as pd
import os
from scripts.preprocess_data import load_data, handle_missing_values, handle_outliers

# Constants
TARGET_VALUE = 1
RESULT_SET_SIZE = 5
SEARCH_DEPTH = 2

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data.

    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of LabelEncoders.
    """
    try:
        df = load_data(file_path)
        df = handle_missing_values(df)
        df = handle_outliers(df)
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def define_target(df):
    """
    Define the target for subgroup discovery.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
    Returns:
        ps.BinaryTarget: The target for subgroup discovery.
    """
    # Define the target variable. 'Label' is the target column with binary values.
    target = ps.BinaryTarget('Label', TARGET_VALUE) # CG: Achtung hardkodiert hier
    return target
    

def create_search_space(columns):
    """
    Create the search space for subgroup discovery based on specified columns and values.
    
    Parameters:
        columns (list): List of tuples containing column names and values to create selectors for.
        
    Returns:
        list: List of possible selectors for subgroup discovery.
    """
    return [ps.EqualitySelector(col, val) for col, val in columns]

def run_subgroup_discovery(df, target, search_space):
    """
    Run the Apriori subgroup discovery algorithm.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        target (ps.BinaryTarget): The target for subgroup discovery.
        search_space (list): List of possible selectors for subgroup discovery.
        
    Returns:
        pd.DataFrame: DataFrame with the results of subgroup discovery.
    """
    # Define the quality function (Weighted Relative Accuracy (WRAcc))
    quality_function = ps.WRAccQF()

    # Create the Apriori algorithm object
    apriori = ps.Apriori()

    # Create the task for subgroup discovery
    task = ps.SubgroupDiscoveryTask(
        df, 
        target, 
        search_space, 
        result_set_size=RESULT_SET_SIZE, 
        depth=SEARCH_DEPTH, 
        qf=quality_function)

    # Execute the algorithm
    result = apriori.execute(task)

    # Convert the result to a DataFrame and display it
    result_df = result.to_dataframe()
    print(result_df)

    return result_df

def display_results(result_df):
    """
    Display detailed results of the subgroup discovery process.
    
    Parameters:
        result_df (pd.DataFrame): DataFrame with the results of subgroup discovery.
    """
    # Display detailed information about each subgroup
    for i, row in result_df.iterrows():
        print(f"Details of Subgroup {i}:")
        print(row)
        print(f"Conditions of Subgroup {i}: {row['subgroup']}")
        print("-" * 40)  # Separator line between subgroups

    # Check for subgroups with zero instances or invalid results and handle them if needed
    for index, row in result_df.iterrows():
        if pd.isna(row['lift']) or row['target_share_dataset'] == 0:
            print(f"Warning: Invalid result in row {index} with subgroup {row['subgroup']}")

def main(file_path):
    """
    Main function to run the PySubgroup algorithm on the given CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:
        df = load_and_preprocess_data(file_path)
        target = define_target(df)
        search_space = create_search_space([('Color', 'Red'), ('Color', 'Blue'), ('Shape', 'Circle'), ('Shape', 'Square')])
        result_df = run_subgroup_discovery(df, target, search_space)
        display_results(result_df)
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

if __name__ == "__main__":
    main()
