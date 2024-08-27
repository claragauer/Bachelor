import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_gurobi import main  

def test_pipeline():
    """
    Test the full pipeline with multiple CSV files.
    """
    # Directory containing the CSV files
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Basispfad des aktuellen Skripts
    data_dir = os.path.join(base_dir, 'test_data') 
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('test') and f.endswith('.csv')]

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Please check the path.")
        return
    
    # Loop over each CSV file and run the main function
    for csv_file in csv_files:
        print(f"\nRunning optimization for {csv_file}...")
        file_path = os.path.join(data_dir, csv_file)

        try:
            main(file_path)
            print(f"Optimization for {csv_file} completed successfully.")
        except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")

if __name__ == '__main__':
    test_pipeline()
