import pysubgroup as ps
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from evaluate_models import measure_memory_usage
import time

# Constants for subgroup discovery
TARGET_VALUE = 1
RESULT_SET_SIZE = 10
SEARCH_DEPTH = 2
THRESHOLD_INCIDENCE = 20  # Threshold for Sieben-Tage-Inzidenz to define "high incidence"

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data for subgroup discovery.
    """
    try:
        df = pd.read_csv(file_path, delimiter=';')

        # Convert week-year to a datetime format (assuming start of the week)
        df['Kalenderwoche'] = pd.to_datetime(df['Kalenderwoche'] + '-1', format='%Y-%W-%w', errors='coerce')

        # Ensure numeric data types where needed
        df['Sieben-Tage-Inzidenz'] = pd.to_numeric(df['Sieben-Tage-Inzidenz'], errors='coerce').fillna(0)

        print("Data after preprocessing:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

def define_target(df):
    """
    Define the target for subgroup discovery based on high incidence values.
    """
    # Define high incidence as a binary target
    df['High_Incidence'] = (df['Sieben-Tage-Inzidenz'] > THRESHOLD_INCIDENCE).astype(int)
    target = ps.BinaryTarget('High_Incidence', TARGET_VALUE)
    return target

def create_search_space(df):
    """
    Create the search space for subgroup discovery based on available columns.
    """
    search_space = []

    # Add selectors based on Altersklasse
    age_classes = df['Altersklasse'].unique()
    search_space += [ps.EqualitySelector('Altersklasse', age_class) for age_class in age_classes]

    # Add selectors for the Sieben-Tage-Inzidenz ranges
    incidence_ranges = [(0, 5), (5, 20), (20, 50), (50, 100)]
    search_space += [ps.IntervalSelector('Sieben-Tage-Inzidenz', start, end) for start, end in incidence_ranges]

    return search_space

def run_subgroup_discovery(df, target, search_space):
    """
    Run the Apriori subgroup discovery algorithm.
    """
    # Define the quality function (Weighted Relative Accuracy)
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
        qf=quality_function
    )

    # Execute the algorithm
    result = apriori.execute(task)

    # Convert the result to a DataFrame and display it
    result_df = result.to_dataframe()
    print(result_df)

    return result_df

def display_results(df, result_df):
    """
    Display and visualize the results of the subgroup discovery process.
    """
    for i, row in result_df.iterrows():
        print(f"Subgroup {i} Details:")
        print(row)
        print(f"Subgroup Conditions: {row['subgroup']}")
        print("-" * 40)

        # Extract condition and identify cases within this subgroup
        subgroup_condition = row['subgroup']
        positive_cases = df[subgroup_condition.covers(df)]

        # Plotting the Sieben-Tage-Inzidenz for all data and the positive cases in the subgroup
        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df['Sieben-Tage-Inzidenz'], color='gray', alpha=0.5, label='All Data')
        plt.scatter(positive_cases.index, positive_cases['Sieben-Tage-Inzidenz'], color='blue', label='High Incidence Subgroup')
        plt.title(f'Subgroup {i}: Sieben-Tage-Inzidenz by Entry Index')
        plt.xlabel('Entry Index')
        plt.ylabel('Sieben-Tage-Inzidenz')
        plt.legend()
        plt.show()

def main(file_path):
    """
    Main function to run the PySubgroup algorithm on the given CSV file.
    """
    try:
        def run_optimization_pipeline(file_path):
            start_time = time.time()
            
            # Load, preprocess, and run the subgroup discovery pipeline
            df = load_and_preprocess_data(file_path)
            target = define_target(df)
            search_space = create_search_space(df)
            result_df = run_subgroup_discovery(df, target, search_space)

            # Display and visualize results
            display_results(df, result_df)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total runtime: {elapsed_time:.2f} seconds")

                    # Run the optimization pipeline with memory measurement
        _, peak_memory = measure_memory_usage(run_optimization_pipeline, file_path)
        # Print peak memory usage
        print(f"The peak memory usage during the pipeline for {file_path} was: {peak_memory:.2f} MB")


    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor/data/ageSpecificIncidenceRates.csv"
    main(file_path)


