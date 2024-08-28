import pysubgroup as ps
import pandas as pd
import os
from scripts.preprocess_data import load_data, handle_missing_values, handle_outliers
# Constants
TARGET_VALUE = 1

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data.

    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of LabelEncoders.
    """
    # Step 1: Load dataset
    df = load_data(file_path)

    # Step 2: Handle Missing Values
    df = handle_missing_values(df, method='ffill')

    # Step 3: Encode Categorical Variables
    #df, label_encoders = encode_categorical_columns(df, categorical_columns)

    # Step 4: Handle Outliers
    df = handle_outliers(df, method='z-score') # evtl methode spezifizieren

    # Step 5: Balance Data
    #df = balance_data(df, target_column, method=balance_method)
    return df


def define_target(df):
    """
    Define the target for subgroup discovery.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
    Returns:
        ps.BinaryTarget: The target for subgroup discovery.
    """
    # Define the target variable. 'Label' is the target column with binary values.
    target = ps.BinaryTarget('Label', TARGET_VALUE)
    print(target)
    return target

def create_search_space():
    """
    Create the search space for subgroup discovery.
    
    Returns:
        list: List of possible selectors for subgroup discovery.
    """
    search_space = [
        ps.EqualitySelector('Color', 'Red'),
        ps.EqualitySelector('Color', 'Blue'),
        ps.EqualitySelector('Shape', 'Circle'),
        ps.EqualitySelector('Shape', 'Square')
    ]
    return search_space

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
        result_set_size=5, 
        depth=2, 
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
    # Load data
    df = load_and_preprocess_data(file_path)

    # Define target for subgroup discovery
    target = define_target(df)
    
    # Create search space for subgroup discovery
    search_space = create_search_space()
    
    # Run subgroup discovery
    result_df = run_subgroup_discovery(df, target, search_space)
    
    # Display results
    display_results(result_df)

if __name__ == "__main__":
    main()
