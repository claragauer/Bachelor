import pysubgroup as ps
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import time

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'Bachelor' directory to the sys.path
sys.path.append(os.path.join(current_dir, 'scripts'))

from preprocess_data import load_data, handle_missing_values, handle_outliers
from evaluate_models import measure_memory_usage
# Constants
TARGET_VALUE = 1
RESULT_SET_SIZE = 10
SEARCH_DEPTH = 2
THRESHOLD_ABWASSER = 1e13  # Schwellenwert für SARS-CoV-2-Konzentration im Abwasser
THRESHOLD_FALLS = 50       # Schwellenwert für gemeldete SARS-CoV-2-Fälle

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
        df = pd.read_csv(file_path)
        
        # Entferne alle Zeilen, die komplett leer sind
        #df.dropna(inplace=True)
        df= df.dropna(axis=1) # drop columns which are empty

        # Optional: Entferne Zeilen, die in bestimmten Spalten NaN-Werte haben
        #df.dropna(subset=['7d-Median SARS-CoV-2 Abwasser', '7d-Median SARS-CoV-2-Fälle'], inplace=True)

        print("Data after dropping NaN rows:")
        print(df.head())
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
    # Datensatz Abwasser Covid
    #df['High_Abwasser'] = (df['7d-Median SARS-CoV-2 Abwasser'] > THRESHOLD_ABWASSER).astype(int)
    #target = ps.BinaryTarget('High_Abwasser', 1)
    # Datensatz 2
    #df['High_Insgesamt'] = (df['Insgesamt'] > 5000).astype(int)
    #target = ps.BinaryTarget('High_Insgesamt', TARGET_VALUE)
    # Datensatz Geothermal Field - California 
    df['High_Capacity'] = (df['Total_MWe_Mean'] > 100).astype(int)
    target = ps.BinaryTarget('High_Capacity', 1)
    return target
    

def create_search_space(columns):
    """
    Create the search space for subgroup discovery based on specified columns and values.
    
    Parameters:
        columns (list): List of tuples containing column names and values to create selectors for.
        
    Returns:
        list: List of possible selectors for subgroup discovery.
    """
    search_space = []
    
    # Datensatz 1
    #concentration_ranges = [(1e12, 2e13), (2e13, 3e13), (3e13, 5e13)]
    #search_space += [ps.IntervalSelector('7d-Median SARS-CoV-2 Abwasser', start, end) for start, end in concentration_ranges]
    #cases_ranges = [(0, 50), (50, 200), (200, 1000)]
    #search_space += [ps.IntervalSelector('7d-Median SARS-CoV-2-Fälle', start, end) for start, end in cases_ranges]

    # Datensatz 2
    #age_groups = ['00 - 09', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 - 89']
    #genders = ['MA', 'FE']
    #search_space += [ps.EqualitySelector('Altersklasse', age_group) for age_group in age_groups]
    #search_space += [ps.EqualitySelector('Geschlecht', gender) for gender in genders]
    #personen_ranges = [(1, 5), (6, 10), (11, 20)]
    #search_space += [ps.IntervalSelector('Anzahl_Personen', start, end) for start, end in personen_ranges]
    
    #Datensatz 3 - Arbeitslosigkeit
    # Selektoren für 'Insgesamt', 'Männer', 'Frauen' und 'Langzeitarbeitslose'
    # Intervalle und Kategorien für die Subgruppen
    #total_ranges = [(4000, 5000), (5000, 6000)]
    #men_ranges = [(2000, 2500), (2500, 3000)]
    #women_ranges = [(2000, 2300), (2300, 2500)]
    #longterm_unemployed_ranges = [(1500, 1600), (1600, 1800)]
    #search_space += [ps.IntervalSelector('Insgesamt', start, end) for start, end in total_ranges]
    #search_space += [ps.IntervalSelector('Männer', start, end) for start, end in men_ranges]
    #search_space += [ps.IntervalSelector('Frauen', start, end) for start, end in women_ranges]
    #search_space += [ps.IntervalSelector('Langzeitarbeitslose', start, end) for start, end in longterm_unemployed_ranges]
    
    # Datensatz 4 - Geothermal Spaces - California
    # Intervals for geothermal power capacity (Total_MWe_Mean)
    capacity_ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000)]
    search_space += [ps.IntervalSelector('Total_MWe_Mean', start, end) for start, end in capacity_ranges]
    undeveloped_ranges = [(0, 50), (50, 100), (100, 500), (500, 1000)]
    search_space += [ps.IntervalSelector('NetUndevelopedRP', start, end) for start, end in undeveloped_ranges]
    area_ranges = [(0, 10000), (10000, 50000), (50000, 100000), (100000, 500000)]
    search_space += [ps.IntervalSelector('Acres_GeothermalField', start, end) for start, end in area_ranges]
    search_space += [ps.EqualitySelector('ProtectedArea_Exclusion', value) for value in [0, 1]]
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
        # Capture start time for runtime measurement
        start_time = time.time()
        
        # Measure memory usage during the optimization pipeline
        def run_optimization_pipeline(file_path):
            df = load_and_preprocess_data(file_path)
            if isinstance(df, pd.Series):
                df = df.to_frame()
            target = define_target(df)
            search_space = create_search_space(df)
            result_df = run_subgroup_discovery(df, target, search_space)

            # Display and visualize results
            for i, row in result_df.iterrows():
                print(f"Details of Subgroup {i}:")
                print(row)
                print(f"Conditions of Subgroup {i}: {row['subgroup']}")
                print("-" * 40)

                # Extract condition and identify cases within this subgroup
                subgroup_condition = row['subgroup']
                positive_cases = df[subgroup_condition.covers(df)]

                # Plotting the results
                plt.figure(figsize=(10, 6))
                plt.scatter(df.index, df['Total_MWe_Mean'], color='gray', alpha=0.5, label='All Data')
                plt.scatter(positive_cases.index, positive_cases['Total_MWe_Mean'], color='blue', label='Positive Cases')
                plt.title(f'Subgroup {i}: Total MWe Mean by Entry Index')
                plt.xlabel('Entry Index')
                plt.ylabel('Total MWe Mean')
                plt.legend()
                plt.show()


        # Run the optimization pipeline with memory measurement
        _, peak_memory = measure_memory_usage(run_optimization_pipeline, file_path)
        
        # Capture end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Output memory and runtime
        print(f"The peak memory usage during the optimization for {file_path} was: {peak_memory:.2f} MB")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/geothermalData.csv" 
    main(file_path)  # Pass the file_path argument to main
    

