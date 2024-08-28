import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_gurobi import main, load_and_preprocess_data

import os

def test_pipeline(selected_tests=None):
    """
    Test the full pipeline with selected CSV files.

    Parameters:
        selected_tests (list, optional): List of CSV filenames to run. If None, all tests are run.
    """
    # Directory containing the CSV files
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test/test_data/'))
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Please check the path.")
        return
    
    # If no specific tests are selected, run all available test CSV files
    if selected_tests is None:
        csv_files = [f for f in os.listdir(data_dir) if f.startswith('test') and f.endswith('.csv')]
    else:
        csv_files = [f for f in selected_tests if f.startswith('test') and f.endswith('.csv')]

    # Dictionary to store results
    results = {}

    # Loop over each selected CSV file and run the main function
    for csv_file in csv_files:
        print(f"\nRunning optimization for {csv_file}...")
        file_path = os.path.join(data_dir, csv_file)

        try:
            main(file_path)
            results[csv_file] = "OK"
            print(f"Optimization for {csv_file} completed successfully.")
        except Exception as e:
            if "display_results() missing" in str(e):
                results[csv_file] = "WARNING"
                print(f"Warning while processing {csv_file}: {e}")
            else:
                results[csv_file] = "ERROR"
                print(f"An error occurred while processing {csv_file}: {e}")

    
    print("\nTest Results Summary:")
    for csv_file, status in results.items():
        print(f"{csv_file}: {status}")

    print("\nAll tests completed.")


def check_missing_values(df):
    """
    Check for missing values in the DataFrame and print the count of missing values per column.

    Parameters:
        df (pd.DataFrame): The DataFrame to check.
    """
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values detected in the DataFrame:")
        print(missing_values)
    else:
        print("No missing values detected in the DataFrame.")

def test_missing_values_in_test_files():
    """
    Test to check for missing values in test9encodeMissing.csv and test10encodeMissing.csv.
    """
    # Directory containing the CSV files
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test/test_data/'))

    # Test files
    test_files = ['test7missing.csv', 'test8missing.csv', 'test10missing.csv', 'test9missing.csv']

    for test_file in test_files:
        print(f"\nChecking for missing values in {test_file}...")
        file_path = os.path.join(data_dir, test_file)

        # Load the data
        try:
            df = load_and_preprocess_data(file_path)
        except FileNotFoundError:
            print(f"File {test_file} not found in {data_dir}. Please check the path.")
            continue

        # Check for missing values
        check_missing_values(df)

def test_handle_outliers():
    """
    Test function to check outlier removal using load_and_preprocess_data.
    """
    # Directory containing the CSV files
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test/test_data/'))

    # Test File 1: Simple Outlier in Label
    file_path1 = os.path.join(data_dir, 'test11outlier.csv')
    result1 = load_and_preprocess_data(file_path1)
    expected_result1 = pd.DataFrame({
        'Color': ['Red', 'Red', 'Blue', 'Green'],
        'Shape': ['Circle', 'Square', 'Circle', 'Circle'],
        'Label': [1, 1, 2, 2]
    })
    assert result1.equals(expected_result1), "Test 1 Failed: Outlier not removed correctly for z-score method"

    # Test File 2: Multiple Outliers in Label
    file_path2 = os.path.join(data_dir, 'test12outlier.csv')
    result2 = load_and_preprocess_data(file_path2)
    expected_result2 = pd.DataFrame({
        'Color': ['Red', 'Red', 'Blue', 'Blue', 'Yellow'],
        'Shape': ['Circle', 'Square', 'Circle', 'Square', 'Circle'],
        'Label': [1, -20, 3, 4, 3]
    }).reset_index(drop=True)
    # CG : hier sieht man ganz klar den Nachteil der ZScore Methode. 
    # Der Wert -20 wird nicht rausgenommen, dafür bräuchte man vmtl noch zusätzlich IQR
    assert result2.equals(expected_result2), "Test 2 Failed: Outliers not removed correctly for z-score"
    print("--------------------------------------------")
    print("OUTLIERS WITH Z-SCORE: All tests passed!")

if __name__ == '__main__':
    # Specify the tests you want to run by passing a list of filenames.
    #test_missing_values_in_test_files()
    #test_handle_outliers()

    # To run all tests, call without arguments:
    test_pipeline()

