import pysubgroup as ps
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'Bachelor' directory to the sys.path
sys.path.append(os.path.join(current_dir, 'scripts'))

from preprocess_data import load_data, handle_missing_values, handle_outliers
from evaluate_models import measure_memory_usage
# Constants
TARGET_VALUE = 1
RESULT_SET_SIZE = 3
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
    threshold = 10000
    # Create a binary target based on whether pedestrian count exceeds the threshold
    df['Binary_Pedestrian'] = (df['Fußgänger insgesamt'] > threshold).astype(int)
    
    # Define the binary target using the new column
    target = ps.BinaryTarget('Binary_Pedestrian', 1)  # Target is 1 (true) if pedestrian count > threshold
    
    return target
    

def create_search_space(columns):
    """
    Create the search space for subgroup discovery based on specified columns and values.
    
    Parameters:
        columns (list): List of tuples containing column names and values to create selectors for.
        
    Returns:
        list: List of possible selectors for subgroup discovery.
    """
    # NOTEX TO SELF: es gibt keinen Numeric Selector 
    weather_conditions = ['rain', 'partly-cloudy-day', 'clear-day']
    
    # Create selectors for Wetterlage
    search_space = [ps.EqualitySelector('Wetterlage', condition) for condition in weather_conditions]
    
    # Create selectors for temperature intervals
    temperature_ranges = [(15, 25), (25, 35), (35, 45)]  # Example ranges
    search_space += [ps.IntervalSelector('Temperatur', start, end) for start, end in temperature_ranges]
    
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
            search_space = create_search_space(df)
            result_df = run_subgroup_discovery(df, target, search_space)
             # Anzeige und Visualisierung der Ergebnisse
            for i, row in result_df.iterrows():
                print(f"Details of Subgroup {i}:")
                print(row)
                print(f"Conditions of Subgroup {i}: {row['subgroup']}")
                print("-" * 40)

                # Extrahiere die Bedingung und finde die Fälle, die diese Subgruppe erfüllen
                subgroup_condition = row['subgroup']
                
                # Filter für die Subgruppe anwenden (subgroup_condition.covers(df) gibt die gefilterten Zeilen zurück)
                positive_cases = df[subgroup_condition.covers(df)]
                
                # Visualisierung der Subgruppe
                plt.figure(figsize=(10, 6))

                # Scatterplot für alle Daten (grau)
                plt.scatter(df['Temperatur'], df['Fußgänger insgesamt'], color='gray', alpha=0.5, label='Alle Daten')

                # Scatterplot für die positiven Fälle (blau)
                plt.scatter(positive_cases['Temperatur'], positive_cases['Fußgänger insgesamt'], color='blue', label='Positive Fälle')

                # Titel und Achsenbeschriftungen
                plt.title(f'Subgroup {i}: Fußgängeranzahl vs. Temperatur')
                plt.xlabel('Temperatur')
                plt.ylabel('Fußgänger insgesamt')

                # Legende
                plt.legend()

                # Plot anzeigen
                plt.show()

        # Speicherverbrauch messen
        #_, peak_memory = measure_memory_usage(file_path)
        #print(f"The peak memory usage during the optimization for {file_path} was: {peak_memory:.2f} MB")

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/besucher.csv" 
    main(file_path)  # Pass the file_path argument to main
    

