import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_pysubgroup import main  

def test_pipeline():
    """
    Test the full pipeline with multiple CSV files.
    """
    # Directory containing the CSV files
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test/test_data/'))
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Please check the path.")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('test') and f.endswith('.csv')]

    # Dictionary to store results
    results = {}

    # Loop over each CSV file and run the main function
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

    # Print summary
    print("\nTest Results Summary:")
    for csv_file, status in results.items():
        print(f"{csv_file}: {status}")

    print("\nAll tests completed.")

if __name__ == '__main__':
    test_pipeline()
